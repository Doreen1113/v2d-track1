"""Fuse the per-frame SAM 3D Body estimates of a second checkpoint (B) into the main ones (A, work/body format),
before the sequence refinement.

B may cover only some frames (e.g. scored frames +- 30, every 3rd frame: run_sam3d_body.py --boxes-from --window
--stride). On the frames where both are valid, B is checked against A:
  - 2D: median distance of each model's own 2D keypoints to ViTPose (conf > 0.5), pixels
  - 3D: mean distance of the root-relative 22 scorer joints between the two models
B frames that are clearly worse in 2D (e_B > --ratio * e_A + --margin) or disagree grossly in 3D (> --max-3d) are
dropped. On the kept frames the residual B - A is formed (root translation, root rotation as a rotation vector of
R_B R_A^T in MHR's native frame, other pose parameters as wrapped angle differences, shape, scales, 2D keypoints),
interpolated linearly over time inside each window segment, and added to A with weight --w-b (0.5 = average).
The hand parameters (pose columns 68:122) stay A's, so B can be a body-decoder-only run (--body-only).
The weight ramps in over --ramp frames at the segment ends, so the fused sequence has no steps there.

  python fuse_body_ensemble.py --a WORK/body --b WORK/body_vith --out WORK/body_ens
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import binary_erosion, label
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhr_utils import MHRForward  # noqa: E402

WORK = "/work/doreen071/v2d/work"
J22 = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
COCO_TO_70 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


def err2d(kp2d, kv):
    e = np.linalg.norm(kp2d[:, COCO_TO_70] - kv[..., :2], axis=-1)
    e = np.where(kv[..., 2] > 0.5, e, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmedian(e, 1)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def ramp_weights(m, n):
    """1 inside the mask, ramping linearly towards 0 over the n frames at each inner end of every run."""
    w = m.astype(np.float64)
    inner = m.copy()
    for k in range(1, n + 1):
        inner = binary_erosion(inner, iterations=1, border_value=1)  # no ramp at the video ends
        w[m & ~inner & (w == 1)] = k / (n + 1)
    return w


def scored_window(ep, T, window, path):
    sc = json.load(open(path))[str(ep)]
    m = np.zeros(T, bool)
    for t in sc:
        m[max(0, t - window):t + window + 1] = True
    return m


def interp_segments(x, good, seg):
    """Linear interpolation of x [T, ...] from the good frames, separately inside each segment (seg labels)."""
    out = np.zeros_like(x)
    have = np.zeros(len(x), bool)
    flat = x.reshape(len(x), -1)
    of = out.reshape(len(x), -1)
    for s in range(1, seg.max() + 1):
        idx = np.flatnonzero(seg == s)
        g = idx[good[idx]]
        if len(g) == 0:
            continue
        for j in range(flat.shape[1]):
            of[idx, j] = np.interp(idx, g, flat[g, j])
        have[idx] = True
    return out, have


def fuse(ep, a_dir, b_dir, out_dir, mhr, args):
    A = dict(np.load(f"{a_dir}/episode_{ep:06d}.npz"))
    B = dict(np.load(f"{b_dir}/episode_{ep:06d}.npz"))
    T = len(A["pose"])
    va, vb = A["valid"].astype(bool), B["valid"].astype(bool)[:T]
    vit = torch.load(f"{WORK}/body_gvhmr/preproc/episode_{ep:06d}_feat.pt", map_location="cpu", weights_only=False)
    kv = vit["kp2d"].numpy().astype(np.float64)
    kv[~vit["detected"].numpy().astype(bool), :, 2] = 0
    both = va & vb
    ea, eb = err2d(A["kp2d"], kv), err2d(B["kp2d"][:T], kv)
    d3 = np.full(T, np.nan)
    idx = np.flatnonzero(both)
    if len(idx):
        _, Ja = mhr(A["pose"][idx], A["scales"][idx], A["shape"][idx])
        _, Jb = mhr(B["pose"][idx], B["scales"][idx], B["shape"][idx])
        d3[idx] = np.linalg.norm((Ja[:, J22] - Ja[:, 1:2]) - (Jb[:, J22] - Jb[:, 1:2]), axis=-1).mean(1)
    worse_b = both & np.isfinite(ea) & np.isfinite(eb) & (eb > args.ratio * ea + args.margin)
    gross = both & (d3 > args.max_3d)
    good = both & ~worse_b & ~gross
    # residuals B - A on the good frames
    Pa = A["pose"].astype(np.float64)
    Pb = B["pose"][:T].astype(np.float64)
    res = np.zeros((T, 136))
    res[good, :3] = Pb[good, :3] - Pa[good, :3]
    if good.any():
        Ra = Rotation.from_euler("xyz", Pa[good, 3:6])
        Rb = Rotation.from_euler("xyz", Pb[good, 3:6])
        res[good, 3:6] = (Rb * Ra.inv()).as_rotvec()
    res[good, 6:] = wrap(Pb[good, 6:] - Pa[good, 6:])
    a, b = (int(v) for v in args.keep_a.split(":"))
    res[:, a:b] = 0.0  # hand parameters stay model A's (B may be a body-decoder-only run)
    r_shape = np.where(good[:, None], B["shape"][:T] - A["shape"], 0.0)
    r_scales = np.where(good[:, None], B["scales"][:T] - A["scales"], 0.0)
    r_kp = np.where(good[:, None, None], B["kp2d"][:T] - A["kp2d"], 0.0)
    # window segments (frames B was meant to cover and A is valid)
    win = scored_window(ep, T, args.window, args.scored) & va if args.window >= 0 else va.copy()
    seg, _ = label(win)
    res_i, have = interp_segments(res, good, seg)
    shp_i, _ = interp_segments(r_shape, good, seg)
    scl_i, _ = interp_segments(r_scales, good, seg)
    kp_i, _ = interp_segments(r_kp, good, seg)
    w = args.w_b * ramp_weights(win & have, args.ramp)
    out = dict(A)
    P = Pa.copy()
    m = w > 0
    P[m, :3] += w[m, None] * res_i[m, :3]
    R = Rotation.from_rotvec(w[m, None] * res_i[m, 3:6]) * Rotation.from_euler("xyz", Pa[m, 3:6])
    d = wrap(R.as_euler("xyz") - Pa[m, 3:6])
    branch = np.abs(d).max(1) > 1.0  # other Euler branch than A (|y| near 90 deg): keep A's root angles
    d[branch] = 0.0
    P[m, 3:6] += d
    P[m, 6:] += w[m, None] * res_i[m, 6:]
    out["pose"] = P.astype(np.float32)
    out["shape"] = (A["shape"] + w[:, None] * shp_i).astype(np.float32)
    out["scales"] = (A["scales"] + w[:, None] * scl_i).astype(np.float32)
    out["kp2d"] = (A["kp2d"] + w[:, None, None] * kp_i).astype(np.float32)
    out["ens_weight_b"] = w.astype(np.float32)
    out["ens_d3"] = d3.astype(np.float32)
    out["ens_good"] = good
    np.savez_compressed(f"{out_dir}/episode_{ep:06d}.npz", **out)
    ef = err2d(out["kp2d"], kv)
    s = dict(ep=ep, b_frames=int(both.sum()), good=int(good.sum()), worse_b=int(worse_b.sum()),
             gross=int(gross.sum()), fused=int(m.sum()), branch=int(branch.sum()),
             e2d_a=float(np.nanmean(ea[both])), e2d_b=float(np.nanmean(eb[both])),
             e2d_fused=float(np.nanmean(ef[both])), d3_med_cm=float(np.nanmedian(d3[both]) * 100),
             d3_p90_cm=float(np.nanpercentile(d3[both], 90) * 100))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=f"{WORK}/body")
    ap.add_argument("--b", default=f"{WORK}/body_vith")
    ap.add_argument("--out", default=f"{WORK}/body_ens")
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--w-b", type=float, default=0.5, help="weight of model b (0.5 = average)")
    ap.add_argument("--window", type=int, default=30, help="B covers scored frames +- window (-1: all frames)")
    ap.add_argument("--scored", default=f"{WORK}/body_v2/scored_frames.json")
    ap.add_argument("--ramp", type=int, default=15)
    ap.add_argument("--keep-a", default="68:122", help="pose columns kept from model A (MHR hand parameters)")
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=3.0, help="pixels")
    ap.add_argument("--max-3d", type=float, default=0.12, help="metres")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mhr = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for ep in args.episodes:
        s = fuse(ep, args.a, args.b, args.out, mhr, args)
        rows.append(s)
        print(" ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()), flush=True)
    keys = ["e2d_a", "e2d_b", "e2d_fused", "d3_med_cm", "d3_p90_cm"]
    print("MEAN " + " ".join(f"{k}={np.mean([r[k] for r in rows]):.2f}" for k in keys) + " "
          + " ".join(f"{k}={sum(r[k] for r in rows)}" for k in ("b_frames", "good", "worse_b", "gross", "fused", "branch")))
    with open(os.path.join(args.out, "fusion_stats.json"), "w") as f:
        json.dump(dict(args=vars(args), episodes=rows), f, indent=1)


if __name__ == "__main__":
    main()
