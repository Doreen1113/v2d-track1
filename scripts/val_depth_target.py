"""Pelvis depth target for refine_body_v5.py --depth-target, with the options the validation study varies.

Same construction as compute_moge_depth_target.py (per frame: median of D / z_body over the visible, eroded person
pixels, divided by median of D / Z_background over the static background, smoothed, normalised by its median;
z_target = z_root * ratio / median), plus:
  --depth-dir DIR   depth maps in the compute_depth.py format (depth [T, H/4, W/4] metres, K, downsample), e.g. a
                    temporally consistent video depth model (val_vda_depth.py) instead of per-frame MoGe
  --stride N        every N-th valid frame gets a measurement (compute_moge_depth_target.py: 3)
  --visible-only    person pixels exclude a dilated object mask (object-occluded / held-object pixels)
  --no-bg-norm      no division by the background ratio (for depth that is already consistent over time)
  --med / --sigma   temporal running median / Gaussian sigma of the ratio (9 / 4)
Paths (masks, the raw body's intrinsics) come from the validation mirror's work dir (V2D_ROOT/work via common.py).

  python val_depth_target.py --body WORK/body_v5_moge --out WORK/depth_target_x --episodes 0 7 --stride 1 --visible-only
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, unpack_masks  # noqa: E402
from compute_moge_depth_target import smooth_ratio  # noqa: E402
from eval_body_proxies import Renderer, faces  # noqa: E402
from mhr_utils import MHRForward  # noqa: E402


def episode_ratio(ep, a, mhr, F, dev, batch=64):
    b = np.load(os.path.join(a.body, f"episode_{ep:06d}.npz"))
    raw = np.load(os.path.join(WORK, "body", f"episode_{ep:06d}.npz"))
    valid = b["valid"].astype(bool) if "valid" in b.files else raw["valid"].astype(bool)
    K = np.array([float(raw["fx"]), float(raw["fy"]), float(raw["cx"]), float(raw["cy"])])
    T = len(b["pose"])
    mk = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
    mh, mw = int(mk["mask_h"]), int(mk["mask_w"])
    dd = np.load(os.path.join(a.depth_dir, f"episode_{ep:06d}.npz"))
    D = dd["depth"]
    dh, dw = D.shape[1:]
    sy, sx = mh // dh, mw // dw
    ds = float(dd["downsample"])
    Kd = np.array([[K[0] / ds, 0, K[2] / ds], [0, K[1] / ds, K[3] / ds], [0, 0, 1]])
    rend = Renderer(Kd, dh, dw, dev)
    bgs = []
    for f in np.linspace(0, len(D) - 1, 40).round().astype(int):
        pm = unpack_masks(mk["person_masks"], mh, mw, f) | unpack_masks(mk["obj_masks"], mh, mw, f)
        pm = binary_dilation(pm[::sy, ::sx][:dh, :dw], iterations=4)
        zz = D[f].astype(np.float32)
        bgs.append(np.where(pm | ~(zz > 0), np.nan, zz))
    with np.errstate(all="ignore"):
        Zbg = np.nanmedian(np.stack(bgs), 0)
    ratio = np.full(T, np.nan)
    npix = np.zeros(T, np.int32)
    frames = np.flatnonzero(valid)[:: a.stride]
    frames = frames[frames < min(T, len(D), len(mk["person_masks"]))]
    zroot = np.zeros(T)
    for s in range(0, len(frames), batch):
        fr = frames[s:s + batch]
        V, J = mhr(b["pose"][fr], b["scales"], b["shape"])
        zroot[fr] = J[:, 1, 2]
        sil, zb = rend(V, F)
        for k, f in enumerate(fr):
            P = unpack_masks(mk["person_masks"], mh, mw, f)
            O = unpack_masks(mk["obj_masks"], mh, mw, f)
            pe = cv2.erode(P.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            if a.visible_only:
                pe &= ~binary_dilation(O, iterations=6)
            pe = pe[::sy, ::sx][:dh, :dw] & sil[k]
            zr = zb[k]
            Df = D[f].astype(np.float32)
            okp = pe & (Df > 0) & (zr > 0)
            npix[f] = int(okp.sum())
            if okp.sum() <= 30:
                continue
            num = np.median(Df[okp] / zr[okp])
            if a.no_bg_norm:
                ratio[f] = float(num)
                continue
            bg = np.isfinite(Zbg) & (Df > 0) & ~binary_dilation((P | O)[::sy, ::sx][:dh, :dw], iterations=4)
            if bg.sum() > 500:
                ratio[f] = float(num / np.median(Df[bg] / Zbg[bg]))
    rest = np.flatnonzero(zroot == 0)
    if len(rest):
        _, J = mhr(b["pose"][rest], b["scales"], b["shape"])
        zroot[rest] = J[:, 1, 2]
    return ratio, npix, zroot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--depth-dir", default=os.path.join(WORK, "depth"))
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--visible-only", action="store_true")
    ap.add_argument("--no-bg-norm", action="store_true")
    ap.add_argument("--med", type=int, default=9)
    ap.add_argument("--sigma", type=float, default=4.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHRForward(device=dev)
    F = faces()
    os.makedirs(a.out, exist_ok=True)
    for ep in a.episodes:
        ratio_raw, npix, zroot = episode_ratio(ep, a, mhr, F, dev)
        rs = smooth_ratio(ratio_raw, max_gap=max(30, 10 * a.stride), med=a.med, sigma=a.sigma)
        ok = np.isfinite(rs)
        m = np.nanmedian(rs) if ok.any() else 1.0
        z_target = np.where(ok, zroot * rs / m, np.nan)
        np.savez_compressed(os.path.join(a.out, f"episode_{ep:06d}.npz"), ratio_raw=ratio_raw, ratio=rs, npix=npix,
                            z_root=zroot, z_target=z_target, valid_target=ok)
        rel = rs / m
        print(f"ep {ep}: frames with target {ok.sum()}/{len(rs)}  |rel-1| p50 {np.nanpercentile(np.abs(rel[ok] - 1), 50) * 100:.2f}%"
              f"  max shift {np.nanmax(np.abs(z_target - zroot)) * 100:.1f} cm", flush=True)


if __name__ == "__main__":
    main()
