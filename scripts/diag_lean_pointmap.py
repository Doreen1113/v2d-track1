"""Diagnostic for a torso-lean observation (towards / away from the camera) from monocular depth of the person.

Per frame, the torso lean is asin(-(upper - lower) . ray / |upper - lower|) (positive: the top is closer to the
camera), with upper / lower the mean of the shoulder / hip points (Sapiens indices 5, 6 / 9, 10), from
  pmap   Sapiens2 pointmap sampled at the Sapiens 2D keypoints (run_sapiens_pointmap.py; crop camera rotated onto the
         real ray of the crop centre)
  moge   MoGe-2 depth (work/depth) at the same pixels, back-projected with the body intrinsics
  model  the refined body's 3D keypoints (joints, not surface)
and their temporal correlations, also on "clean" frames (both wrists farther than 0.5 torso lengths from the hip
centre in the image and no object-mask pixels near the four torso points), since a hand or a held object in front
of the hips replaces the hip surface.

  python diag_lean_pointmap.py --episodes 0 3 --ref WORK/body_vA
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import unpack_masks  # noqa: E402
from eval_sap_proxies import regress  # noqa: E402
from mhr_utils import MHRForward  # noqa: E402

WORK = "/work/doreen071/v2d/work"


def lean(up, lo, center):
    v = up - lo
    r = center / np.linalg.norm(center, axis=-1, keepdims=True)
    return np.degrees(np.arcsin(np.clip(-(v * r).sum(-1) / np.linalg.norm(v, axis=-1), -1, 1)))


def rot_z_to(d):
    z = np.array([0, 0, 1.0])
    v = np.cross(z, d)
    s = np.linalg.norm(v)
    c = d @ z
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / max(s * s, 1e-12))


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def episode(ep, ref, mhr, kmap):
    pm = np.load(f"{WORK}/sap_pmap/episode_{ep:06d}.npz")
    kp = np.load(f"{WORK}/sap_kp/episode_{ep:06d}.npz")["kp"]
    raw = np.load(f"{WORK}/body/episode_{ep:06d}.npz")
    K = np.array([raw["fx"], raw["fy"], raw["cx"], raw["cy"]], float)
    xyz = pm["kp_xyz"].astype(np.float64)
    cs = pm["crop_cs"]
    T = len(xyz)
    d = np.stack([(cs[:, 0] - K[2]) / K[0], (cs[:, 1] - K[3]) / K[1], np.ones(T)], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    X = np.einsum("tij,tkj->tki", np.stack([rot_z_to(x) for x in d]), xyz)
    up, lo = X[:, [5, 6]].mean(1), X[:, [9, 10]].mean(1)
    l_pm = lean(up, lo, (up + lo) / 2)
    dd = np.load(f"{WORK}/depth/episode_{ep:06d}.npz")
    D, ds = dd["depth"], float(dd["downsample"])
    n = min(T, len(D))
    h, w = D.shape[1:]
    M = np.full((n, 4, 3), np.nan)
    for t in range(n):
        for i, k in enumerate((5, 6, 9, 10)):
            u, v = kp[t, k]
            ui, vi = int(round(u / ds)), int(round(v / ds))
            if 0 <= ui < w and 0 <= vi < h:
                p = D[t, max(vi - 1, 0):vi + 2, max(ui - 1, 0):ui + 2].astype(np.float64)
                z = np.nanmedian(np.where(p > 0, p, np.nan))
                M[t, i] = [(u - K[2]) / K[0] * z, (v - K[3]) / K[1] * z, z]
    l_mg = lean(M[:, :2].mean(1), M[:, 2:].mean(1), M.mean(1))
    b = np.load(f"{ref}/episode_{ep:06d}.npz")
    V, J = mhr(b["pose"], b["scales"], b["shape"])
    k3 = regress(kmap, V, J)
    l_md = lean(k3[:, [5, 6]].mean(1), k3[:, [9, 10]].mean(1), k3[:, [5, 6, 9, 10]].mean(1))
    # clean frames
    wr, hp, sh = kp[:, [41, 62]], kp[:, [9, 10]].mean(1), kp[:, [5, 6]].mean(1)
    dh = np.min(np.linalg.norm(wr - hp[:, None], axis=-1), 1) / np.linalg.norm(sh - hp, axis=1)
    mk = np.load(f"{WORK}/masks/episode_{ep:06d}.npz")
    mh, mw = int(mk["mask_h"]), int(mk["mask_w"])
    occ = np.zeros(n, bool)
    for t in range(n):
        O = unpack_masks(mk["obj_masks"], mh, mw, t)
        for k in (5, 6, 9, 10):
            ui, vi = (kp[t, k] * mh / float(mk["full_h"])).astype(int)
            if 0 <= vi < mh and 0 <= ui < mw and O[max(vi - 3, 0):vi + 4, max(ui - 3, 0):ui + 4].any():
                occ[t] = True
    clean = (dh[:n] > 0.5) & ~occ
    a, g, m = l_pm[:n], l_mg[:n], l_md[:n]
    out = dict(ep=ep, mean_pmap=np.nanmean(a), mean_moge=np.nanmean(g), mean_model=np.nanmean(m),
               sd_pmap=np.nanstd(a), sd_moge=np.nanstd(g), sd_model=np.nanstd(m),
               corr_pmap_moge=corr(a, g), corr_pmap_model=corr(a, m), corr_moge_model=corr(g, m),
               clean_frac=float(clean.mean()), corr_hand_pmap=corr(-dh[:n], a),
               clean_corr_pmap_model=corr(a[clean], m[clean]), clean_corr_pmap_moge=corr(a[clean], g[clean]),
               clean_mean_pmap=float(np.nanmean(a[clean])) if clean.any() else float("nan"),
               clean_mean_model=float(np.nanmean(m[clean])) if clean.any() else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+")
    ap.add_argument("--ref", default=f"{WORK}/body_vA")
    a = ap.parse_args()
    mhr = MHRForward(device="cpu")
    kmap = sp.load_npz(f"{WORK}/sap_kp/kp308_mapping.npz").tocsr()
    for ep in a.episodes:
        r = episode(ep, a.ref, mhr, kmap)
        print(" ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)


if __name__ == "__main__":
    main()
