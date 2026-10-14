"""Per-frame metric depth with MoGe-2 using the fixed episode intrinsics.

Output: WORK/depth/episode_XXXXXX.npz
  depth  [T, H/4, W/4] float16 metres (z along the OpenCV optical axis), 0 where invalid
  K      3x3 intrinsics at FULL resolution (scale by 1/4 for the stored maps)

Usage: python compute_depth.py --shard i/n
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import VIDEO_TMPL, WORK, load_intrinsics  # noqa: E402

DS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--shard", type=str, default=None)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    eps = a.episodes if a.episodes else list(range(30))
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        eps = eps[i::n]
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
    out_dir = os.path.join(WORK, "depth")
    os.makedirs(out_dir, exist_ok=True)
    for ep in eps:
        p = os.path.join(out_dir, f"episode_{ep:06d}.npz")
        if os.path.exists(p):
            continue
        K, _ = load_intrinsics(ep)
        cap = cv2.VideoCapture(VIDEO_TMPL.format(ep))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fov_x = math.degrees(2 * math.atan(W / (2 * K[0, 0])))
        depths, buf = [], []

        def flush():
            x = torch.from_numpy(np.stack(buf)).cuda().permute(0, 3, 1, 2).float() / 255.0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.infer(x, fov_x=fov_x)
            d = torch.nan_to_num(out["depth"].float(), nan=0.0, posinf=0.0)
            if "mask" in out:
                d = d * out["mask"].float()
            d = torch.nn.functional.interpolate(d[:, None], scale_factor=1 / DS, mode="nearest")[:, 0]
            depths.append(d.cpu().numpy().astype(np.float16))
            buf.clear()

        while True:
            ok, f = cap.read()
            if not ok:
                break
            buf.append(f[:, :, ::-1].copy())
            if len(buf) == a.batch:
                flush()
        if buf:
            flush()
        D = np.concatenate(depths)
        np.savez_compressed(p, depth=D, K=K, downsample=DS)
        print(ep, D.shape, float(np.median(D[D > 0])), flush=True)


if __name__ == "__main__":
    main()
