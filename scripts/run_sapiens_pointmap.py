"""Sapiens2 pointmap (per-pixel camera-frame XYZ of the person) on every frame, for a torso-lean observation.

Same person crop as run_sapiens_pose.py (box padded 1.25, 3:4, UDP affine to 1024 x 768). The model predicts the
pointmap of the crop in its own canonical camera (focal 768 px at the 1024 x 768 input, principal point at the crop
centre); pointmap / scale is metric. Stored per frame (float16):
  kp_xyz [308,3]   pointmap sampled at the Sapiens2 2D keypoints (work/sap_kp; bilinear, 5x5 median around the point)
  small  [128,96,3] the whole crop pointmap, block-averaged (for later analysis)
plus the crop centre / scale, so crop pixels map back to image pixels (x = u * s / 767 + c - s / 2).

  python run_sapiens_pointmap.py --episodes 0 1 --out WORK/sap_pmap
"""
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sapiens_pose import IN_H, IN_W, SAP_REPO, WORK, load_boxes, producer  # noqa: E402

CFG = f"{SAP_REPO}/sapiens/dense/configs/pointmap/render_people/sapiens2_{{}}_pointmap_render_people-1024x768.py"


def build_model(size, ckpt, device):
    sys.path.insert(0, SAP_REPO)
    cwd = os.getcwd()
    os.chdir(f"{SAP_REPO}/sapiens/dense")
    try:
        from sapiens.dense.models import init_model
        model = init_model(CFG.format(size), ckpt, device=device)
    finally:
        os.chdir(cwd)
    return model


def sample_median(pm, uv, r=2):
    """pm [B,3,H,W], uv [B,K,2] crop pixels -> [B,K,3]: median of the (2r+1)^2 neighbourhood (robust to edges)."""
    B, _, H, W = pm.shape
    offs = torch.stack(torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij"), -1)
    offs = offs.reshape(-1, 2).to(pm)
    pts = uv[:, :, None] + offs[None, None]  # [B,K,M,2]
    g = torch.stack([pts[..., 0] / (W - 1) * 2 - 1, pts[..., 1] / (H - 1) * 2 - 1], -1)
    s = F.grid_sample(pm, g.reshape(B, -1, 1, 2), align_corners=True, padding_mode="border")  # [B,3,K*M,1]
    s = s.reshape(B, 3, uv.shape[1], -1)
    return s.median(-1).values.permute(0, 2, 1)


def run_episode(ep, model, args, dev):
    t0 = time.time()
    boxes, src = load_boxes(ep)
    T = len(boxes)
    sk = np.load(f"{args.sap_kp}/episode_{ep:06d}.npz")["kp"].astype(np.float32)
    q = queue.Queue(maxsize=4 * args.batch)
    stop = threading.Event()
    th = threading.Thread(target=producer, args=(ep, boxes, q, stop), daemon=True)
    th.start()
    kp_xyz = np.full((T, 308, 3), np.nan, np.float16)
    small = np.zeros((T, 128, 96, 3), np.float16)
    cs = np.zeros((T, 4), np.float32)
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
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pm, scale = model(x)
        pm = pm.float() / scale.float().reshape(-1, 1, 1, 1)
        if pm.shape[-2:] != (IN_H, IN_W):
            pm = F.interpolate(pm, size=(IN_H, IN_W), mode="bilinear", align_corners=False)
        uv = []
        for (t, _, c, s) in items:
            u = (sk[t, :, 0] - c[0] + 0.5 * s[0]) * (IN_W - 1) / s[0]
            v = (sk[t, :, 1] - c[1] + 0.5 * s[1]) * (IN_H - 1) / s[1]
            uv.append(np.stack([u, v], -1))
        uv = torch.as_tensor(np.nan_to_num(np.stack(uv)), device=dev, dtype=torch.float32)
        xyz = sample_median(pm, uv).cpu().numpy()
        sm = F.avg_pool2d(pm, 8).permute(0, 2, 3, 1).cpu().numpy()
        for j, (t, _, c, s) in enumerate(items):
            kp_xyz[t] = xyz[j]
            small[t] = sm[j]
            cs[t] = [c[0], c[1], s[0], s[1]]
        n += len(items)
    stop.set()
    th.join(timeout=5)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(f"{args.out}/episode_{ep:06d}.npz", kp_xyz=kp_xyz, small=small, crop_cs=cs, box_src=src,
                        canonical_focal=768.0)
    print(f"ep {ep}: {n}/{T} frames, {time.time() - t0:.0f}s, median depth {np.nanmedian(kp_xyz[:, :17, 2].astype(np.float32)):.2f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--out", default=f"{WORK}/sap_pmap")
    ap.add_argument("--sap-kp", default=f"{WORK}/sap_kp")
    ap.add_argument("--size", default="1b", choices=["0.4b", "0.8b", "1b", "5b"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    if args.ckpt is None:
        from huggingface_hub import hf_hub_download
        args.ckpt = hf_hub_download(f"facebook/sapiens2-pointmap-{args.size}", f"sapiens2_{args.size}_pointmap.safetensors")
    dev = torch.device("cuda")
    model = build_model(args.size, args.ckpt, "cuda")
    eps = list(range(30)) if args.all or not args.episodes else args.episodes
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        eps = eps[i::n]
    for ep in eps:
        if args.skip_existing and os.path.exists(f"{args.out}/episode_{ep:06d}.npz"):
            continue
        run_episode(ep, model, args, dev)


if __name__ == "__main__":
    main()
