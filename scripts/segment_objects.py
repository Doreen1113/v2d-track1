"""Per-frame object (and person) masks for Track 1 episodes.

Detector: open-vocabulary text detection (Grounding DINO, prompt = `object_prompt`
from episodes_metadata.jsonl) on sampled frames to produce seed boxes.
Tracker: SAM 2.1 video predictor, propagated forward and backward from the seed frame.
Several candidate seeds are tracked; the target track is chosen by detection score,
motion (manipulated objects move), visibility and proximity to the main person.

Output: WORK/masks/episode_XXXXXX.npz with
  obj_masks     [T, ceil(h*w/8)] uint8  packed bits (np.packbits over the flattened h x w mask)
  obj_score     [T] float32  SAM 2 object-presence logit (>0 = visible)
  obj_area      [T] int32    mask pixel count at mask resolution
  obj_bbox      [T,4] float32 x0,y0,x1,y1 at FULL resolution (nan if empty)
  person_*      same for the main actor
  mask_h, mask_w (half resolution: 576 x 768), full_h, full_w, seed info (json string)

Usage: python segment_objects.py --episodes 0 1 2   (or --shard i/n)
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import VIDEO_TMPL, WORK, load_meta, pack_masks, read_video  # noqa: E402

MASK_SCALE = 0.5
DET_STRIDE = 10
MAX_SEED_FRAMES = 5
MAX_BOXES = 5
# short category prompts used in addition to the metadata prompt
SHORT_PROMPTS = {
    "hula_hoop": "a hula hoop.", "big_red_bowl": "a red bowl.", "iron": "an iron.",
    "white_desk": "a white desk.", "black_pan": "a frying pan.", "foam_grass_block": "a green foam block.",
    "paint_roller": "a paint roller.", "pink_foam_roll": "a pink foam roller.",
    "short_wood_stool": "a wooden stool.", "white_laptop_cart": "a white laptop cart.",
}


def clean_prompt(p):
    p = p.replace("_", " ").strip().lower()
    if not p.endswith("."):
        p += "."
    return p


class Detector:
    def __init__(self, device):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        mid = "IDEA-Research/grounding-dino-base"
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, images, text, box_thr=0.2, text_thr=0.2):
        out = []
        for i in range(0, len(images), 8):
            batch = images[i:i + 8]
            inp = self.proc(images=batch, text=[text] * len(batch), return_tensors="pt").to(self.device)
            res = self.model(**inp)
            pp = self.proc.post_process_grounded_object_detection(
                res, inp.input_ids, threshold=box_thr, text_threshold=text_thr,
                target_sizes=[im.shape[:2] for im in batch])
            for r in pp:
                out.append((r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy()))
        return out


def box_iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1]); x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def mask_bbox(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return np.full(4, np.nan, np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def track(predictor, state, seeds, T):
    """seeds: list of (obj_id, frame, box) all on the same frame. Returns masks dict, logits dict."""
    predictor.reset_state(state)
    f0 = seeds[0][1]
    for oid, f, box in seeds:
        predictor.add_new_points_or_box(state, frame_idx=f, obj_id=oid, box=np.asarray(box, np.float32))
    h, w = state["video_height"], state["video_width"]
    masks = {oid: np.zeros((T, h, w), bool) for oid, _, _ in seeds}
    for rev in (False, True):
        for fi, oids, logits in predictor.propagate_in_video(state, start_frame_idx=f0, reverse=rev):
            m = (logits[:, 0] > 0).cpu().numpy()
            for k, oid in enumerate(oids):
                masks[oid][fi] = m[k]
    scores = {}
    for oid in masks:
        idx = state["obj_id_to_idx"][oid]
        od = state["output_dict_per_obj"][idx]
        s = np.full(T, -10.0, np.float32)
        for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for fi, o in od[key].items():
                s[fi] = float(o["object_score_logits"].float().max())
        scores[oid] = s
    return masks, scores


def track_stats(m, person_m):
    T = m.shape[0]
    area = m.reshape(T, -1).sum(1)
    vis = area > 20
    if vis.sum() < 3:
        return dict(vis=float(vis.mean()), motion=0.0, prox=0.0, person_iou=0.0, area=0.0)
    k = np.ones((15, 15), np.uint8)
    med_area = np.median(area[vis])
    # motion = how much of the mask leaves the region the object usually occupies.
    # Occlusion by the actor only removes pixels, so static structures score ~0.
    ref = m[vis].sum(0) >= 0.5 * vis.sum()
    ref = cv2.dilate(ref.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    nov = np.array([(m[t] & ~ref).sum() / max(area[t], 1) for t in np.nonzero(vis)[0]])
    motion = float(np.percentile(nov, 95))
    prox, piou = [], []
    for t in np.nonzero(vis)[0][::5]:
        if person_m is None or person_m[t].sum() == 0:
            continue
        pd = cv2.dilate(person_m[t].astype(np.uint8), k) > 0
        prox.append(float((pd & m[t]).any()))
        piou.append((person_m[t] & m[t]).sum() / max((person_m[t] | m[t]).sum(), 1))
    return dict(vis=float(vis.mean()), motion=float(motion), prox=float(np.mean(prox)) if prox else 0.0,
                person_iou=float(np.mean(piou)) if piou else 0.0, area=float(med_area))


def process(ep, meta, det, predictor, tmp_root, out_dir, vis_dir):
    t0 = time.time()
    prompt = clean_prompt(meta[ep]["object_prompt"])
    frames = read_video(VIDEO_TMPL.format(ep), scale=MASK_SCALE)
    T = len(frames)
    H, W = frames[0].shape[:2]
    full_h, full_w = int(round(H / MASK_SCALE)), int(round(W / MASK_SCALE))
    jdir = tempfile.mkdtemp(dir=tmp_root)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(jdir, f"{i:05d}.jpg"), f[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])

    sample = list(range(0, T, DET_STRIDE))
    imgs = [frames[i] for i in sample]
    per_det = det(imgs, "a person.")
    prompts = [prompt]
    short = SHORT_PROMPTS.get(meta[ep]["object"])
    if short and short != prompt:
        prompts.append(short)
    obj_dets = [det(imgs, p, box_thr=0.15, text_thr=0.15) for p in prompts]
    # second pass on crops around the actor so small hand-held objects get more pixels
    crops, offs, actor = [], [], []
    for k, im in enumerate(imgs):
        pb, ps = per_det[k]
        if len(pb) and ps.max() > 0.35:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) if sc > 0.35 else 0 for b, sc in zip(pb, ps)]
            b = pb[int(np.argmax(areas))]
        else:
            b = np.array([W * 0.3, H * 0.2, W * 0.7, H * 0.9])
        actor.append(b)
        bw, bh = b[2] - b[0], b[3] - b[1]
        x0 = int(max(0, b[0] - 1.2 * bw)); x1 = int(min(W, b[2] + 1.2 * bw))
        y0 = int(max(0, b[1] - 0.2 * bh)); y1 = int(min(H, b[3] + 0.2 * bh))
        crops.append(np.ascontiguousarray(im[y0:y1, x0:x1])); offs.append((x0, y0))
    for p in prompts:
        res = det(crops, p, box_thr=0.15, text_thr=0.15)
        shifted = []
        for (b, sc), (x0, y0) in zip(res, offs):
            b = b.copy()
            if len(b):
                b[:, [0, 2]] += x0; b[:, [1, 3]] += y0
            shifted.append((b, sc))
        obj_dets.append(shifted)

    # main person: largest confident person box, typically the actor near the scene centre
    pbest = None
    for fi, (b, s) in zip(sample, per_det):
        for bb, ss in zip(b, s):
            if ss < 0.35:
                continue
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            if pbest is None or area > pbest[0]:
                pbest = (area, fi, bb, ss)
    parea = pbest[0] if pbest is not None else 0.05 * H * W

    # candidate boxes per sampled frame (union over prompts), large background structures dropped
    frame_boxes = {}
    for od in obj_dets:
        for k, fi in enumerate(sample):
            b, s = od[k]
            pb, ps = per_det[k]
            for bb, ss in zip(b, s):
                if ss < 0.15:
                    continue
                if (bb[2] - bb[0]) * (bb[3] - bb[1]) > 2.5 * parea:
                    continue
                if any(box_iou(bb, q) > 0.7 for q, qs in zip(pb, ps) if qs > 0.35):
                    continue
                # interaction prior: boxes touching the (expanded) actor box rank higher
                a = actor[k]
                aw, ah = a[2] - a[0], a[3] - a[1]
                near = not (bb[2] < a[0] - 0.3 * aw or bb[0] > a[2] + 0.3 * aw or
                            bb[3] < a[1] - 0.1 * ah or bb[1] > a[3] + 0.1 * ah)
                pr = float(ss) + (0.15 if near else 0.0)
                lst = frame_boxes.setdefault(fi, [])
                dup = [j for j, (q, qs) in enumerate(lst) if box_iou(bb, q) > 0.6]
                if dup:
                    j = dup[0]
                    lst[j] = (lst[j][0], max(lst[j][1], pr))
                else:
                    lst.append((bb, pr))
    ranked = sorted(frame_boxes, key=lambda fi: -max(x[1] for x in frame_boxes[fi]))
    seed_frames = []
    gap = max(30, T // 10)
    for fi in ranked:
        if all(abs(fi - g) > gap for g in seed_frames):
            seed_frames.append(fi)
        if len(seed_frames) >= MAX_SEED_FRAMES:
            break
    state = predictor.init_state(video_path=jdir, offload_video_to_cpu=True, async_loading_frames=False)

    person_m, person_s = None, np.full(T, -10.0, np.float32)
    if pbest is not None:
        pm, ps = track(predictor, state, [(1, pbest[1], pbest[2])], T)
        person_m, person_s = pm[1], ps[1]

    tracks = []
    for sf in seed_frames:
        boxes = sorted(frame_boxes[sf], key=lambda x: -x[1])[:MAX_BOXES]
        seeds = [(j + 1, sf, bb, sc) for j, (bb, sc) in enumerate(boxes)]
        if not seeds:
            continue
        ms, ss = track(predictor, state, [(o, f, bb) for o, f, bb, _ in seeds], T)
        for o, f, bb, sc in seeds:
            st = track_stats(ms[o], person_m)
            st.update(seed_frame=int(f), seed_box=[float(x / MASK_SCALE) for x in bb], det_score=sc)
            tracks.append((st, ms[o], ss[o]))

    # merge duplicates: keep the higher det score
    keep = []
    for st, m, s in sorted(tracks, key=lambda x: -x[0]["det_score"]):
        dup = False
        for st2, m2, _ in keep:
            inter = (m & m2).reshape(T, -1).sum(1)
            uni = (m | m2).reshape(T, -1).sum(1)
            v = uni > 0
            if v.any() and np.mean(inter[v] / uni[v]) > 0.5:
                st2["support"] = st2.get("support", 1) + 1
                dup = True
                break
        if not dup:
            keep.append((st, m, s))

    def total(st):
        return (st["det_score"] + 1.0 * min(st["motion"] / 0.5, 1.0) + 0.3 * st["prox"] + 0.2 * st["vis"]
                + 0.1 * min(st.get("support", 1) - 1, 3) - 1.0 * (st["person_iou"] > 0.5)
                - 1.0 * (st["area"] > 2.0 * person_area))

    person_area = np.median(person_m.reshape(T, -1).sum(1)) if person_m is not None else 0.05 * H * W
    if keep:
        for st, _, _ in keep:
            st["total"] = float(total(st))
        keep.sort(key=lambda x: -x[0]["total"])
        best_st, obj_m, obj_s = keep[0]
    else:
        best_st, obj_m, obj_s = {}, np.zeros((T, H, W), bool), np.full(T, -10.0, np.float32)
    if person_m is None:
        person_m = np.zeros((T, H, W), bool)

    def bboxes(m):
        return np.stack([mask_bbox(x) for x in m]) / MASK_SCALE

    info = dict(prompts=prompts, person_seed=None if pbest is None else
                dict(frame=int(pbest[1]), box=[float(x / MASK_SCALE) for x in pbest[2]], score=float(pbest[3])),
                candidates=[st for st, _, _ in keep], seed_frames=[int(x) for x in seed_frames],
                tracker="sam2.1-hiera-large", detector="grounding-dino-base")
    np.savez_compressed(
        os.path.join(out_dir, f"episode_{ep:06d}.npz"),
        obj_masks=pack_masks(obj_m), obj_score=obj_s, obj_area=obj_m.reshape(T, -1).sum(1).astype(np.int32),
        obj_bbox=bboxes(obj_m).astype(np.float32),
        person_masks=pack_masks(person_m), person_score=person_s,
        person_area=person_m.reshape(T, -1).sum(1).astype(np.int32), person_bbox=bboxes(person_m).astype(np.float32),
        mask_h=H, mask_w=W, full_h=full_h, full_w=full_w, num_frames=T, info=json.dumps(info))

    # visualisation: 8 frames overlay
    idxs = np.linspace(0, T - 1, 8).astype(int)
    tiles = []
    for t in idxs:
        im = frames[t].copy()
        im[obj_m[t]] = (0.5 * im[obj_m[t]] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        im[person_m[t]] = (0.6 * im[person_m[t]] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)
        im = cv2.resize(im, (384, 288))
        cv2.putText(im, f"ep{ep} f{t} s={obj_s[t]:.1f}", (5, 20), 0, 0.6, (255, 255, 0), 2)
        tiles.append(im)
    sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    cv2.imwrite(os.path.join(vis_dir, f"episode_{ep:06d}_masks.jpg"), sheet[:, :, ::-1])
    shutil.rmtree(jdir, ignore_errors=True)
    print(f"[ep {ep}] T={T} vis={best_st.get('vis')} motion={best_st.get('motion')} "
          f"ncand={len(keep)} time={time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--shard", type=str, default=None, help="i/n")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(WORK, "masks"))
    a = ap.parse_args()
    meta = load_meta()
    eps = a.episodes if a.episodes else sorted(meta)
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        eps = eps[i::n]
    out_dir = a.out_dir
    vis_dir = os.path.join(out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    tmp_root = os.environ.get("TMPDIR", "/tmp")
    dev = "cuda"
    det = Detector(dev)
    from sam2.build_sam import build_sam2_video_predictor_hf
    predictor = build_sam2_video_predictor_hf("facebook/sam2.1-hiera-large", device=dev)
    for ep in eps:
        if a.skip_existing and os.path.exists(os.path.join(out_dir, f"episode_{ep:06d}.npz")):
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            process(ep, meta, det, predictor, tmp_root, out_dir, vis_dir)


if __name__ == "__main__":
    main()
