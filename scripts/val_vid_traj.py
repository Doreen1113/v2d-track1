"""Pelvis depth trajectory from a source run (e.g. a video model) transplanted onto a base run (e.g. V-A).

The base keeps its pose, shape and scales. Per frame the whole base body is shifted rigidly along the camera ray
through its pelvis (joint 1), so its 2D projection does not change and only its depth does:
  p'(t) = p_b(t) * z'(t) / z_b(t)
Modes for the new depth z'(t) (z_b, z_s: base / source pelvis depth in the camera frame):
  abs    z' = z_s                                              (source depth as is)
  rel    z' = median(z_b) * z_s / median(z_s)                  (source trajectory shape, base depth level: keeps the
                                                                base body size and its depth consistent)
  lf     log z' = log z_b - G(log z_b) + G(log z_s) + median(log z_b - log z_s)
                                                               (only the slow part of the source, Gaussian sigma
                                                                --sigma frames, at the base level; the base keeps
                                                                its fast motion)
Both inputs are assembled runs (val_assemble.py output). Medians over the valid source frames.

  python val_vid_traj.py --base /work/doreen071/v2d_val/runs/va_moge/episodes \
      --src /work/doreen071/v2d_val/runs/vid_gvhmr/episodes --mode rel --out /work/doreen071/v2d_val/runs/vid_gvhmr_va_rel/episodes
"""
import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhr_utils import MHRForward, cam_t_to_root_trans  # noqa: E402


def gauss(x, sigma):
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.concatenate([np.repeat(x[:1], r), x, np.repeat(x[-1:], r)])
    return np.convolve(xp, k, mode="valid")


def new_depth(zb, zs, mode, sigma):
    if mode == "abs":
        return zs
    if mode == "rel":
        return np.median(zb) * zs / np.median(zs)
    if mode == "lf":
        lb, ls = np.log(zb), np.log(zs)
        return np.exp(lb - gauss(lb, sigma) + gauss(ls, sigma) + np.median(lb - ls))
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["abs", "rel", "lf"], default="rel")
    ap.add_argument("--sigma", type=float, default=45.0, help="lf: Gaussian sigma in frames (45 ~ 0.1 Hz at 30 fps)")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    a = ap.parse_args()
    import torch
    fw = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)
    eps = a.episodes if a.episodes is not None else sorted(
        int(f[8:14]) for f in os.listdir(a.src) if f.startswith("episode_") and f.endswith(".npz"))
    os.makedirs(a.out, exist_ok=True)
    for ep in eps:
        f = f"episode_{ep:06d}.npz"
        zb, zs = dict(np.load(os.path.join(a.base, f))), np.load(os.path.join(a.src, f))
        T = min(len(zb["pose"]), len(zs["pose"]))
        pose = np.asarray(zb["pose"], np.float64)[:T]
        _, Jb = fw(pose, zb["scales"], zb["shape"])
        _, Js = fw(np.asarray(zs["pose"], np.float64)[:T], zs["scales"], zs["shape"])
        pb, ps = Jb[:, 1], Js[:, 1]
        zn = new_depth(pb[:, 2], ps[:, 2], a.mode, a.sigma)
        pn = pb * (zn / pb[:, 2])[:, None]
        pose = pose.copy()
        pose[:, 0:3] += cam_t_to_root_trans(pn - pb)
        zb.update(pose=pose)
        np.savez(os.path.join(a.out, f), **zb)
        for ext in (".ply", ".obj", ".glb", ".off", ".stl"):
            m = os.path.join(a.base, f"episode_{ep:06d}_object{ext}")
            if os.path.exists(m):
                shutil.copy(m, a.out)
        print(f"ep {ep}: base z {np.median(pb[:, 2]):.3f} src z {np.median(ps[:, 2]):.3f} -> {np.median(zn):.3f} m; "
              f"|dz| mean {np.abs(zn - pb[:, 2]).mean() * 100:.1f} cm", flush=True)


if __name__ == "__main__":
    main()
