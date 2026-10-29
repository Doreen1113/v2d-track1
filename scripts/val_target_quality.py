"""How good is a pelvis depth target (compute_moge_depth_target.py / val_depth_target.py output) before any refine?

For each episode the GT pelvis depth over the scored frames is compared with (1) the depth of the body the target was
built from (z_root) and (2) the target (z_target), both after the best single scale to the GT (the target is
scale-free) and after anchoring at the first scored frame (as the scorer's frame-0 alignment does). RMS in % of
depth; a target only helps if it is closer to the GT than the body it corrects.

  python val_target_quality.py /work/doreen071/v2d_val/root/work/depth_target_vb_moge [...more dirs]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhr_utils import MHRForward  # noqa: E402

VAL = "/work/doreen071/v2d_val"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--scored", default="scored_frames.json")
    a = ap.parse_args()
    fw = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)
    sc = {int(k): np.array(v) for k, v in json.load(open(f"{VAL}/gt/{a.scored}")).items()}
    gz = {}
    for ep, fr in sc.items():
        g = np.load(f"{VAL}/gt/episode_{ep:06d}.npz")
        gz[ep] = fw(g["pose"][fr], g["scales"], g["shape"])[1][:, 1, 2]
    for d in a.targets:
        rows = []
        for ep, fr in sorted(sc.items()):
            p = os.path.join(d, f"episode_{ep:06d}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p)
            g = gz[ep]
            out = []
            for x in (z["z_root"][fr], z["z_target"][fr]):
                ok = np.isfinite(x)
                k = np.dot(x[ok], g[ok]) / np.dot(x[ok], x[ok])
                best = np.sqrt(np.mean((k * x[ok] / g[ok] - 1) ** 2)) * 100
                anch = np.sqrt(np.nanmean((x / x[ok][0] - g / g[ok][0]) ** 2)) * 100
                out += [best, anch]
            rows.append([ep] + out)
        r = np.array(rows)
        print(f"{os.path.basename(d)} ({len(r)} eps): best-scale RMS body {r[:, 1].mean():.2f}% target {r[:, 3].mean():.2f}% "
              f"(target better on {int((r[:, 3] < r[:, 1]).sum())}) | anchored at f0: body {r[:, 2].mean():.2f}% "
              f"target {r[:, 4].mean():.2f}%", flush=True)


if __name__ == "__main__":
    main()
