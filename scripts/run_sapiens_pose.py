"""Sapiens2 308-keypoint top-down pose (body, feet, hands, face) on every frame of the Track 1 episodes.

Person boxes come from the existing SAM 3D Body run (work/body bbox, xyxy full-image pixels); frames without a
valid box fall back to the person-mask box (work/masks), else to the nearest valid box in time. Each box is padded
by 1.25 and widened to the model's 3:4 aspect (Sapiens2 test pipeline: PoseGetBBoxCenterScale + UDP
PoseTopdownAffine, 1024 x 768 input), with horizontal flip test (heatmaps averaged after the keypoint swap).
Heatmaps (256 x 192) are decoded on the GPU as UDPHeatmap.decode does (arg-max, Gaussian modulation with kernel 11,
DARK-UDP second-order refinement) and mapped back to full-image pixels.

Output: <out>/episode_XXXXXX.npz with kp [T,308,2] (x, y full-image pixels), score [T,308] (heat-map peak value),
box [T,4] (xyxy box used), box_src [T] (0 body, 1 mask, 2 nearest), and names.json (308 keypoint names) in <out>.

  python run_sapiens_pose.py --episodes 0 1 2 --out WORK/sap_kp
  python run_sapiens_pose.py --all --shard 0/3 --out WORK/sap_kp
"""
import argparse
import json
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/work/doreen071/v2d"
WORK = f"{ROOT}/work"
VIDEO = f"{ROOT}/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{{:06d}}.mp4"
SAP_REPO = f"{ROOT}/third_party/sapiens2"
CFG_DIR = f"{SAP_REPO}/sapiens/pose/configs/keypoints308/shutterstock_goliath_3po"

IN_W, IN_H = 768, 1024
HM_W, HM_H = 192, 256
MEAN = np.array([123.675, 116.28, 103.53], np.float32)  # RGB
STD = np.array([58.395, 57.12, 57.375], np.float32)


def load_boxes(ep):
    b = np.load(f"{WORK}/body/episode_{ep:06d}.npz")
    T = len(b["pose"])
    valid = b["valid"].astype(bool)
    box = b["bbox"].astype(np.float64).copy()
    src = np.full(T, 2, np.int8)
    src[valid] = 0
    m = np.load(f"{WORK}/masks/episode_{ep:06d}.npz")
    pb = m["person_bbox"].astype(np.float64)[:T]
    ok_m = (~valid[:len(pb)]) & (m["person_area"][:T] > 0) & (pb[:, 2] > pb[:, 0])
    box[:len(pb)][ok_m] = pb[ok_m]  # person_bbox is in full-image pixels
    src[:len(pb)][ok_m] = 1
    good = src < 2
    if not good.any():
        raise ValueError(f"ep {ep}: no person box")
    gi = np.flatnonzero(good)
    for t in np.flatnonzero(~good):
        box[t] = box[gi[np.argmin(np.abs(gi - t))]]
    return box, src


def udp_warp(center, scale):
    """get_udp_warp_matrix with rot 0, output (IN_W, IN_H)."""
    sx = (IN_W - 1) / scale[0]
    sy = (IN_H - 1) / scale[1]
    return np.array([[sx, 0, sx * (-center[0] + 0.5 * scale[0])],
                     [0, sy, sy * (-center[1] + 0.5 * scale[1])]], np.float32)


def box_cs(box, padding=1.25):
    x1, y1, x2, y2 = box
    c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    s = np.array([(x2 - x1) * padding, (y2 - y1) * padding])
    ar = IN_W / IN_H
    if s[0] > s[1] * ar:
        s = np.array([s[0], s[0] / ar])
    else:
        s = np.array([s[1] * ar, s[1]])
    return c, s


def producer(ep, boxes, q, stop):
    cap = cv2.VideoCapture(VIDEO.format(ep))
    t = 0
    while not stop.is_set():
        ok, img = cap.read()
        if not ok or t >= len(boxes):
            break
        c, s = box_cs(boxes[t])
        M = udp_warp(c, s)
        interp = cv2.INTER_AREA if min(M[0, 0], M[1, 1]) < 1.0 else cv2.INTER_CUBIC
        crop = cv2.warpAffine(img, M, (IN_W, IN_H), flags=interp)
        crop = (crop[..., ::-1].astype(np.float32) - MEAN) / STD
        q.put((t, np.ascontiguousarray(crop.transpose(2, 0, 1)), c, s))
        t += 1
    cap.release()
    q.put(None)


def gauss_kernel(k=11):
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8  # OpenCV default for sigma 0
    x = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def decode(hm, gk):
    """hm [B,K,H,W] float32 (GPU) -> kp [B,K,2] heat-map coordinates, score [B,K]. UDPHeatmap.decode on the GPU."""
    B, K, H, W = hm.shape
    flat = hm.reshape(B, K, -1)
    val, idx = flat.max(-1)
    x = (idx % W).float()
    y = (idx // W).float()
    # Gaussian modulation (zero padding as in gaussian_blur, then rescale to the original maximum)
    k = len(gk)
    pad = (k - 1) // 2
    h = hm.reshape(B * K, 1, H, W)
    h = F.conv2d(F.pad(h, (pad, pad, pad, pad)), gk.view(1, 1, 1, k))
    h = F.conv2d(h, gk.view(1, 1, k, 1))
    h = h.reshape(B, K, H, W)
    hmax = h.reshape(B, K, -1).max(-1).values
    h = h * (val / hmax.clamp(min=1e-12))[..., None, None]
    h = torch.log(h.clamp(1e-3, 50.0))
    hp = F.pad(h, (1, 1, 1, 1), mode="replicate")  # edge padding
    xi = x.long() + 1
    yi = y.long() + 1
    bi = torch.arange(B, device=hm.device)[:, None].expand(B, K)
    ki = torch.arange(K, device=hm.device)[None].expand(B, K)

    def at(dy, dx):
        return hp[bi, ki, yi + dy, xi + dx]
    i_ = at(0, 0)
    ix1, ix1_ = at(0, 1), at(0, -1)
    iy1, iy1_ = at(1, 0), at(-1, 0)
    ix1y1, ix1_y1_ = at(1, 1), at(-1, -1)
    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    eps = float(np.finfo(np.float32).eps)
    a, b_, c_, d = dxx + eps, dxy, dxy, dyy + eps
    det = a * d - b_ * c_
    det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
    ox = (d * dx - b_ * dy) / det
    oy = (-c_ * dx + a * dy) / det
    x = x - ox
    y = y - oy
    kp = torch.stack([x, y], -1)
    kp[val <= 0] = -1
    return kp, val


def build_model(size, ckpt, device):
    sys.path.insert(0, SAP_REPO)
    cwd = os.getcwd()
    os.chdir(f"{SAP_REPO}/sapiens/pose")  # configs use relative _base_ paths
    try:
        from sapiens.pose.datasets import parse_pose_metainfo
        from sapiens.pose.models import init_model
        cfg = f"{CFG_DIR}/sapiens2_{size}_keypoints308_shutterstock_goliath_3po-1024x768.py"
        model = init_model(cfg, ckpt, device=device)
        meta = parse_pose_metainfo(dict(from_file="configs/_base_/keypoints308.py"))
    finally:
        os.chdir(cwd)
    names = [meta["keypoint_id2name"][i] for i in range(meta["num_keypoints"])]
    return model, np.asarray(meta["flip_indices"]), names


def run_episode(ep, model, flip_idx, args, gk, dev):
    t0 = time.time()
    boxes, src = load_boxes(ep)
    T = len(boxes)
    q = queue.Queue(maxsize=4 * args.batch)
    stop = threading.Event()
    th = threading.Thread(target=producer, args=(ep, boxes, q, stop), daemon=True)
    th.start()
    kp = np.full((T, 308, 2), np.nan, np.float32)
    sc = np.zeros((T, 308), np.float32)
    fi = torch.as_tensor(flip_idx, device=dev)
    done = False
    n = 0
    while not done:
        items = []
        while len(items) < args.batch:
            it = q.get()
            if it is None:
                done = True
                break
            items.append(it)
        if not items:
            break
        x = torch.as_tensor(np.stack([i[1] for i in items]), device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32):
            if args.no_flip:
                hm = model(x).float()
            else:
                out = model(torch.cat([x, x.flip(-1)], 0)).float()
                hm = out[:len(x)]
                hf = out[len(x):].flip(-1)[:, fi]
                hm = 0.5 * (hm + hf)
            k, v = decode(hm, gk)
        k = k.cpu().numpy()
        v = v.cpu().numpy()
        for j, (t, _, c, s) in enumerate(items):
            # heat-map -> crop (x * (IN-1)/(HM-1)) -> image (crop * s / (IN-1) + c - s/2)
            kp[t, :, 0] = k[j, :, 0] * s[0] / (HM_W - 1) + c[0] - 0.5 * s[0]
            kp[t, :, 1] = k[j, :, 1] * s[1] / (HM_H - 1) + c[1] - 0.5 * s[1]
            sc[t] = v[j]
        n += len(items)
    stop.set()
    th.join(timeout=5)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(f"{args.out}/episode_{ep:06d}.npz", kp=kp, score=sc, box=boxes.astype(np.float32),
                        box_src=src, n_frames=n)
    print(f"ep {ep}: {n}/{T} frames, {time.time() - t0:.0f}s ({n / max(time.time() - t0, 1e-6):.1f} fps), "
          f"median score body {np.median(sc[:, :23]):.3f} hands {np.median(sc[:, 23:65]):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--out", default=f"{WORK}/sap_kp")
    ap.add_argument("--size", default="1b", choices=["0.4b", "0.8b", "1b", "5b"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    if args.ckpt is None:
        from huggingface_hub import hf_hub_download
        args.ckpt = hf_hub_download(f"facebook/sapiens2-pose-{args.size}", f"sapiens2_{args.size}_pose.safetensors")
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, flip_idx, names = build_model(args.size, args.ckpt, "cuda")
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/names.json", "w") as f:
        json.dump(names, f)
    gk = gauss_kernel(11).to(dev)
    eps = list(range(30)) if args.all or not args.episodes else args.episodes
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        eps = eps[i::n]
    for ep in eps:
        if args.skip_existing and os.path.exists(f"{args.out}/episode_{ep:06d}.npz"):
            continue
        run_episode(ep, model, flip_idx, args, gk, dev)


if __name__ == "__main__":
    main()
