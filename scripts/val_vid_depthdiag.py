"""Pelvis depth trajectory of assembled runs vs the GT (diagnostic for the video-model study).

Per episode and run, on the scored frames: pelvis (joint 1) depth z(t) in the camera frame, normalised by its
median (removes the depth level / body size), compared with the GT's normalised depth:
  rel_rms_pct   RMS of z/med(z) - z_gt/med(z_gt), in % of depth
  lf_rms_pct    the same after a Gaussian low-pass (sigma 45 frames, ~0.1 Hz)
  corr_lf       correlation of the low-passed normalised depths
  z_ratio       median(z) / median(z_gt)

  python val_vid_depthdiag.py moge va_moge vid_gvhmr [--scored scored_frames.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhr_utils import MHRForward  # noqa: E402
from val_vid_traj import gauss  # noqa: E402

VAL = "/work/doreen071/v2d_val"


def ident(z):
    sc, sh = np.asarray(z["scales"], np.float64), np.asarray(z["shape"], np.float64)
    return (np.median(sc, 0) if sc.ndim == 2 else sc), (np.median(sh, 0) if sh.ndim == 2 else sh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--scored", default="scored_frames.json")
    ap.add_argument("--sigma", type=float, default=45.0)
    a = ap.parse_args()
    import torch
    fw = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)
    scored = {int(k): np.asarray(v, int) for k, v in json.load(open(f"{VAL}/gt/{a.scored}")).items()}
    res = {r: {} for r in a.runs}
    for ep in sorted(scored):
        zg = np.load(f"{VAL}/gt/episode_{ep:06d}.npz")
        _, Jg = fw(zg["pose"], *ident(zg))
        fr = scored[ep]
        g = Jg[fr, 1, 2]
        gn = g / np.median(g)
        for r in a.runs:
            zp = np.load(f"{VAL}/runs/{r}/episodes/episode_{ep:06d}.npz")
            _, Jp = fw(zp["pose"], *ident(zp))
            p = Jp[fr, 1, 2]
            pn = p / np.median(p)
            lp, lg = gauss(pn, a.sigma), gauss(gn, a.sigma)
            res[r][ep] = dict(rel_rms_pct=100 * np.sqrt(np.mean((pn - gn) ** 2)),
                              lf_rms_pct=100 * np.sqrt(np.mean((lp - lg) ** 2)),
                              corr_lf=float(np.corrcoef(lp, lg)[0, 1]), z_ratio=float(np.median(p) / np.median(g)))
    keys = ["rel_rms_pct", "lf_rms_pct", "corr_lf", "z_ratio"]
    print("| run | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
    for r in a.runs:
        print(f"| {r} | " + " | ".join(f"{np.mean([v[k] for v in res[r].values()]):.3f}" for k in keys) + " |")
    print("\nper episode rel_rms_pct / lf_rms_pct:")
    for ep in sorted(scored):
        print(ep, "  ".join(f"{r}: {res[r][ep]['rel_rms_pct']:.2f}/{res[r][ep]['lf_rms_pct']:.2f}" for r in a.runs))


if __name__ == "__main__":
    main()
