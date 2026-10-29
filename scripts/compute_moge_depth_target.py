"""Per-frame MoGe-vs-body relative depth and a scale-free root-depth target for the sequence refinement.

For every 3rd valid frame the body of a run is rendered at the MoGe depth resolution and compared with its depth on the
visible, eroded person pixels: ratio_t = median(D_moge / z_body) / median(D_moge / Z_background), where the
background term removes MoGe's per-frame scale jitter (Z_background = median static background depth). ratio_t is
smoothed over time (running median 9, Gaussian sigma 4; gaps > 30 frames are left empty) and normalised by its
median, so only the shape of the depth trajectory is used, not MoGe's metric scale. The target root depth is
z_target_t = z_root_t * ratio_t / median(ratio) (moving the body along its pelvis ray changes its depth only).

Output: <out>/episode_XXXXXX.npz with frames, ratio_raw, ratio (smoothed), npix, z_root, z_target, valid_target.

  python compute_moge_depth_target.py --body WORK/body_v5 --out WORK/moge_depth_target_v5
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_dilation, gaussian_filter1d, median_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, unpack_masks  # noqa: E402
from eval_body_proxies import Renderer, faces  # noqa: E402
from mhr_utils import MHRForward  # noqa: E402


def episode_ratio(ep, body_dir, mhr, F, dev, batch=64, stride=3):
    b = np.load(os.path.join(body_dir, f"episode_{ep:06d}.npz"))
    raw = np.load(os.path.join(WORK, "body", f"episode_{ep:06d}.npz"))
    valid = b["valid"].astype(bool) if "valid" in b.files else raw["valid"].astype(bool)
    K = np.array([float(raw["fx"]), float(raw["fy"]), float(raw["cx"]), float(raw["cy"])])
    T = len(b["pose"])
    mk = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
    mh, mw = int(mk["mask_h"]), int(mk["mask_w"])
    dd = np.load(os.path.join(WORK, "depth", f"episode_{ep:06d}.npz"))
    D = dd["depth"]
    dh, dw = D.shape[1:]
    sy, sx = mh // dh, mw // dw
    ds = float(dd["downsample"])  # render directly at the MoGe depth resolution
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
    frames = np.flatnonzero(valid)[::stride]
    frames = frames[frames < min(T, len(D), len(mk["person_masks"]))]
    zroot = np.zeros(T)
    for a in range(0, len(frames), batch):
        fr = frames[a:a + batch]
        V, J = mhr(b["pose"][fr], b["scales"], b["shape"])
        zroot[fr] = J[:, 1, 2]
        sil, zb = rend(V, F)
        for k, f in enumerate(fr):
            P = unpack_masks(mk["person_masks"], mh, mw, f)
            O = unpack_masks(mk["obj_masks"], mh, mw, f)
            pe = cv2.erode(P.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            pe = pe[::sy, ::sx][:dh, :dw] & sil[k]
            zr = zb[k]
            Df = D[f].astype(np.float32)
            okp = pe & (Df > 0) & (zr > 0)
            bg = np.isfinite(Zbg) & (Df > 0) & ~binary_dilation((P | O)[::sy, ::sx][:dh, :dw], iterations=4)
            npix[f] = int(okp.sum())
            if okp.sum() > 30 and bg.sum() > 500:
                ratio[f] = float(np.median(Df[okp] / zr[okp]) / np.median(Df[bg] / Zbg[bg]))
    # z_root for frames without a ratio (invalid ones): forward in one go
    rest = np.flatnonzero(zroot == 0)
    if len(rest):
        _, J = mhr(b["pose"][rest], b["scales"], b["shape"])
        zroot[rest] = J[:, 1, 2]
    return ratio, npix, zroot


def smooth_ratio(r, max_gap=30, med=9, sigma=4.0):
    ok = np.isfinite(r)
    out = np.full_like(r, np.nan)
    if ok.sum() < 10:
        return out
    idx = np.flatnonzero(ok)
    x = np.interp(np.arange(len(r)), idx, r[idx])
    x = median_filter(x, size=med, mode="nearest")
    x = gaussian_filter1d(x, sigma, mode="nearest")
    # distance to the nearest frame with a measurement
    d = np.full(len(r), 10 ** 6)
    last = -10 ** 6
    for t in range(len(r)):
        if ok[t]:
            last = t
        d[t] = t - last
    last = 10 ** 6
    for t in range(len(r) - 1, -1, -1):
        if ok[t]:
            last = t
        d[t] = min(d[t], last - t)
    out[d <= max_gap // 2] = x[d <= max_gap // 2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHRForward(device=dev)
    F = faces()
    os.makedirs(a.out, exist_ok=True)
    for ep in a.episodes:
        ratio_raw, npix, zroot = episode_ratio(ep, a.body, mhr, F, dev)
        rs = smooth_ratio(ratio_raw)
        ok = np.isfinite(rs)
        m = np.nanmedian(rs) if ok.any() else 1.0
        rel = rs / m
        z_target = np.where(ok, zroot * rel, np.nan)
        np.savez_compressed(os.path.join(a.out, f"episode_{ep:06d}.npz"), ratio_raw=ratio_raw, ratio=rs, npix=npix,
                            z_root=zroot, z_target=z_target, valid_target=ok)
        dev_pct = np.nanpercentile(np.abs(rel[ok] - 1), [50, 90]) * 100 if ok.any() else [np.nan, np.nan]
        print(f"ep {ep}: frames with target {ok.sum()}/{len(rs)}  |rel-1| p50 {dev_pct[0]:.2f}% p90 {dev_pct[1]:.2f}%  "
              f"max shift {np.nanmax(np.abs(z_target - zroot)) * 100:.1f} cm", flush=True)


if __name__ == "__main__":
    main()
