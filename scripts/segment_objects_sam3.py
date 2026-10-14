"""Per-frame object (and person) masks with SAM 3 (text-prompted video detection + tracking).

Main mask path. SAM 3 is prompted with the episode's `object_prompt` (and a short category
prompt); it detects and tracks every instance of the concept through the video. The target
instance is chosen with the same rule as the Grounding DINO + SAM 2 fallback
(segment_objects.py): detection confidence, motion (mask leaving its usual region; static
scene structures score ~0), visibility, proximity to the actor, size sanity.
The actor is the "person" instance with the largest summed mask area.

Output format is identical to segment_objects.py (WORK/masks/episode_XXXXXX.npz).

Usage: python segment_objects_sam3.py --episodes 0 1   (or --shard i/n) [--out-dir DIR]
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
from segment_objects import MASK_SCALE, SHORT_PROMPTS, clean_prompt, mask_bbox, track_stats  # noqa: E402


def run_prompt(predictor, sid, text, T, h, w):
    predictor.handle_request(dict(type="reset_session", session_id=sid))
    predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=text))
    masks, probs = {}, {}
    for resp in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
        fi = resp["frame_index"]
        out = resp["outputs"]
        ids = np.asarray(out["out_obj_ids"]).tolist()
        bm = out["out_binary_masks"]
        pr = np.asarray(out["out_probs"])
        for k, oid in enumerate(ids):
            m = np.asarray(bm[k]).astype(bool)
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            if oid not in masks:
                masks[oid] = np.zeros((T, h, w), bool)
                probs[oid] = np.zeros(T, np.float32)
            masks[oid][fi] = m
            probs[oid][fi] = float(pr[k])
    return masks, probs


def process(ep, meta, predictor, tmp_root, out_dir, vis_dir):
    t0 = time.time()
    frames = read_video(VIDEO_TMPL.format(ep), scale=MASK_SCALE)
    T = len(frames)
    H, W = frames[0].shape[:2]
    full_h, full_w = int(round(H / MASK_SCALE)), int(round(W / MASK_SCALE))
    jdir = tempfile.mkdtemp(dir=tmp_root)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(jdir, f"{i:05d}.jpg"), f[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    sid = predictor.handle_request(dict(type="start_session", resource_path=jdir))["session_id"]

    # actor
    pm, pp = run_prompt(predictor, sid, "person", T, H, W)
    person_m, person_s = np.zeros((T, H, W), bool), np.full(T, -10.0, np.float32)
    if pm:
        best = max(pm, key=lambda o: pm[o].reshape(T, -1).sum())
        person_m = pm[best]
        person_s = np.where(pp[best] > 0, np.log(pp[best] / np.clip(1 - pp[best], 1e-6, None)), -10.0)
    del pm
    person_area = np.median(person_m.reshape(T, -1).sum(1)) if person_m.any() else 0.05 * H * W

    prompts = [clean_prompt(meta[ep]["object_prompt"])]
    short = SHORT_PROMPTS.get(meta[ep]["object"])
    if short and short not in prompts:
        prompts.append(short)
    tracks = []
    for p in prompts:
        om, op = run_prompt(predictor, sid, p.rstrip("."), T, H, W)
        for oid in om:
            m = om[oid]
            vis = m.reshape(T, -1).any(1)
            if vis.sum() < 3:
                continue
            st = track_stats(m, person_m)
            st.update(prompt=p, obj_id=int(oid), det_score=float(op[oid][vis].mean()))
            tracks.append((st, m, op[oid]))
        del om
    predictor.handle_request(dict(type="close_session", session_id=sid))
    shutil.rmtree(jdir, ignore_errors=True)

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

    if keep:
        for st, _, _ in keep:
            st["total"] = float(total(st))
        keep.sort(key=lambda x: -x[0]["total"])
        best_st, obj_m, obj_p = keep[0]
        obj_s = np.where(obj_p > 0, np.log(np.clip(obj_p, 1e-6, 1) / np.clip(1 - obj_p, 1e-6, None)), -10.0)
        obj_s = obj_s.astype(np.float32)
    else:
        best_st, obj_m, obj_s = {}, np.zeros((T, H, W), bool), np.full(T, -10.0, np.float32)

    def bboxes(m):
        return np.stack([mask_bbox(x) for x in m]) / MASK_SCALE

    info = dict(prompts=prompts, candidates=[st for st, _, _ in keep], tracker="sam3", detector="sam3")
    np.savez_compressed(
        os.path.join(out_dir, f"episode_{ep:06d}.npz"),
        obj_masks=pack_masks(obj_m), obj_score=obj_s, obj_area=obj_m.reshape(T, -1).sum(1).astype(np.int32),
        obj_bbox=bboxes(obj_m).astype(np.float32),
        person_masks=pack_masks(person_m), person_score=person_s.astype(np.float32),
        person_area=person_m.reshape(T, -1).sum(1).astype(np.int32), person_bbox=bboxes(person_m).astype(np.float32),
        mask_h=H, mask_w=W, full_h=full_h, full_w=full_w, num_frames=T, info=json.dumps(info))

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
    print(f"[ep {ep}] T={T} ncand={len(keep)} best={ {k: best_st.get(k) for k in ('prompt', 'motion', 'vis', 'total')} } "
          f"time={time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--shard", type=str, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(WORK, "masks"))
    a = ap.parse_args()
    meta = load_meta()
    eps = a.episodes if a.episodes else sorted(meta)
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        eps = eps[i::n]
    vis_dir = os.path.join(a.out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor(gpus_to_use=[torch.cuda.current_device()])
    for ep in eps:
        if a.skip_existing and os.path.exists(os.path.join(a.out_dir, f"episode_{ep:06d}.npz")):
            continue
        process(ep, meta, predictor, os.environ.get("TMPDIR", "/tmp"), a.out_dir, vis_dir)


if __name__ == "__main__":
    main()
