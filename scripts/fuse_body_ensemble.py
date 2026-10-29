"""Fuse per-frame SAM 3D Body estimates of other runs ("members": second checkpoint, crop-scale or mirrored
test-time passes) into the main ones (A, work/body format), before the sequence refinement.

Members may cover only some frames (e.g. scored frames +- 30: run_sam3d_body.py --boxes-from --window). On the
frames where a member and A are valid, the member is checked against A:
  - 2D: median distance of each run's own 2D keypoints to ViTPose (conf > 0.5), pixels
  - 3D: mean distance of the root-relative 22 scorer joints (full members) and root-rotation difference
Member frames that are clearly worse in 2D (e > --ratio * e_A + --margin) or disagree grossly (> --max-3d,
> --max-rot) are dropped. On the kept frames the residual member - A is formed (root translation, root rotation as
a rotation vector of R_member R_A^T in MHR's native frame, other pose parameters as wrapped angle differences,
shape, scales, 2D keypoints) and interpolated linearly over time inside each window segment. The fused value is the
weighted mean of A (weight 1) and the members (weight --member-weight), per column; root-only members (mirrored
passes, whose joint angles stay mirrored) contribute the root translation / rotation and the 2D keypoints only.
The arm chain (pose columns 30:50 and 132:134: clavicle to wrist) and the fingers (68:122) stay A's: in full mode
SAM 3D Body re-runs the body decoder with wrist / elbow prompts from its hand decoder, so a body-decoder-only B
(--body-only) has worse arms and hands. Only the root, spine, neck / head and legs are fused.
The weight ramps in over --ramp frames at the segment ends, so the fused sequence has no steps there.

  python fuse_body_ensemble.py --a WORK/body --b WORK/body_vith_b --out WORK/body_ens2
  python fuse_body_ensemble.py --a WORK/body --b WORK/body_vith_b TTA/d3_s090 ... --root-only TTA/d3_flip TTA/vh_flip \
      --out WORK/body_ens3
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
ARM_KP = [7, 8] + list(range(21, 67))  # MHR70: elbows, hand keypoints and wrists (21..62), olecranon / cubital fossa


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


def member_residuals(ep, A, b_dir, root_only, kv, ea, mhr, args, seg, win):
    """Interpolated residuals of one member against A; returns (res, shp, scl, kp, weight [T], stats)."""
    B = dict(np.load(f"{b_dir}/episode_{ep:06d}.npz"))
    T = len(A["pose"])
    va, vb = A["valid"].astype(bool), B["valid"].astype(bool)[:T]
    both = va & vb
    eb = err2d(B["kp2d"][:T], kv)
    Pa = A["pose"].astype(np.float64)
    Pb = B["pose"][:T].astype(np.float64)
    d3 = np.full(T, np.nan)
    idx = np.flatnonzero(both)
    rot = np.full(T, np.nan)
    if len(idx):
        rot[idx] = np.degrees((Rotation.from_euler("xyz", Pb[idx, 3:6])
                               * Rotation.from_euler("xyz", Pa[idx, 3:6]).inv()).magnitude())
        if not root_only:
            _, Ja = mhr(A["pose"][idx], A["scales"][idx], A["shape"][idx])
            _, Jb = mhr(B["pose"][idx], B["scales"][idx], B["shape"][idx])
            d3[idx] = np.linalg.norm((Ja[:, J22] - Ja[:, 1:2]) - (Jb[:, J22] - Jb[:, 1:2]), axis=-1).mean(1)
    worse_b = both & np.isfinite(ea) & np.isfinite(eb) & (eb > args.ratio * ea + args.margin)
    gross = both & ((d3 > args.max_3d) | (rot > args.max_rot))
    good = both & ~worse_b & ~gross
    res = np.zeros((T, 136))
    res[good, :3] = Pb[good, :3] - Pa[good, :3]
    if good.any():
        res[good, 3:6] = (Rotation.from_euler("xyz", Pb[good, 3:6])
                          * Rotation.from_euler("xyz", Pa[good, 3:6]).inv()).as_rotvec()
    if not root_only:
        res[good, 6:] = wrap(Pb[good, 6:] - Pa[good, 6:])
    for rng_ in args.keep_a.split(","):  # arm chain and fingers stay model A's (members may be body-decoder-only)
        a, b = (int(v) for v in rng_.split(":"))
        res[:, a:b] = 0.0
    r_shape = np.where(good[:, None], B["shape"][:T] - A["shape"], 0.0)
    r_scales = np.where(good[:, None], B["scales"][:T] - A["scales"], 0.0)
    r_kp = np.where(good[:, None, None], B["kp2d"][:T] - A["kp2d"], 0.0)
    r_kp[:, ARM_KP] = 0.0  # elbows, hands, wrists, olecranon / cubital fossa stay A's (as the arm parameters)
    res_i, have = interp_segments(res, good, seg)
    shp_i, _ = interp_segments(r_shape, good, seg)
    scl_i, _ = interp_segments(r_scales, good, seg)
    kp_i, _ = interp_segments(r_kp, good, seg)
    w = args.member_weight * ramp_weights(win & have, args.ramp)
    st = dict(frames=int(both.sum()), good=int(good.sum()), worse=int(worse_b.sum()), gross=int(gross.sum()),
              e2d=float(np.nanmean(eb[both])) if both.any() else float("nan"),
              rot_med_deg=float(np.nanmedian(rot[both])) if both.any() else float("nan"),
              d3_med_cm=float(np.nanmedian(d3[both]) * 100) if both.any() and not root_only else float("nan"))
    return res_i, shp_i, scl_i, kp_i, w, st


def fuse(ep, a_dir, members, out_dir, mhr, args):
    """members: list of (dir, root_only). Weighted mean of A (weight 1) and the members (weight member_weight,
    ramped at the window ends) per pose column; root rotation averaged in A's tangent space; root-only members
    (mirrored passes) contribute the root translation / rotation and the 2D keypoints only."""
    A = dict(np.load(f"{a_dir}/episode_{ep:06d}.npz"))
    T = len(A["pose"])
    va = A["valid"].astype(bool)
    vit = torch.load(f"{WORK}/body_gvhmr/preproc/episode_{ep:06d}_feat.pt", map_location="cpu", weights_only=False)
    kv = vit["kp2d"].numpy().astype(np.float64)
    kv[~vit["detected"].numpy().astype(bool), :, 2] = 0
    ea = err2d(A["kp2d"], kv)
    win = scored_window(ep, T, args.window, args.scored) & va if args.window >= 0 else va.copy()
    seg, _ = label(win)
    num = np.zeros((T, 136))
    den = np.ones((T, 136))
    num_s, num_c, den_full = np.zeros((T, 45)), np.zeros((T, 68)), np.ones(T)
    num_k, den_k = np.zeros((T, 70, 2)), np.ones(T)
    stats = {}
    for b_dir, root_only in members:
        res_i, shp_i, scl_i, kp_i, w, st = member_residuals(ep, A, b_dir, root_only, kv, ea, mhr, args, seg, win)
        cols = np.zeros(136, bool)
        cols[:6] = True
        if not root_only:
            cols[6:] = True
        num += w[:, None] * res_i * cols
        den += w[:, None] * cols
        if not root_only:
            num_s += w[:, None] * shp_i
            num_c += w[:, None] * scl_i
            den_full += w
        num_k += w[:, None, None] * kp_i
        den_k += w
        stats[os.path.basename(os.path.normpath(b_dir))] = st
    mean = num / den
    Pa = A["pose"].astype(np.float64)
    P = Pa.copy()
    m = (den[:, 0] > 1)
    P[:, :3] += mean[:, :3]
    R = Rotation.from_rotvec(mean[m, 3:6]) * Rotation.from_euler("xyz", Pa[m, 3:6])
    d = wrap(R.as_euler("xyz") - Pa[m, 3:6])
    branch = np.abs(d).max(1) > 1.0  # other Euler branch than A (|y| near 90 deg): keep A's root angles
    d[branch] = 0.0
    P[m, 3:6] += d
    P[:, 6:] += mean[:, 6:]
    out = dict(A)
    out["pose"] = P.astype(np.float32)
    out["shape"] = (A["shape"] + num_s / den_full[:, None]).astype(np.float32)
    out["scales"] = (A["scales"] + num_c / den_full[:, None]).astype(np.float32)
    out["kp2d"] = (A["kp2d"] + num_k / den_k[:, None, None]).astype(np.float32)
    out["ens_members"] = (den[:, 0] - 1).astype(np.float32)
    np.savez_compressed(f"{out_dir}/episode_{ep:06d}.npz", **out)
    ef = err2d(out["kp2d"], kv)
    cov = va & m
    s = dict(ep=ep, fused=int(m.sum()), branch=int(branch.sum()), e2d_a=float(np.nanmean(ea[cov])),
             e2d_fused=float(np.nanmean(ef[cov])), members=stats)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=f"{WORK}/body")
    ap.add_argument("--b", nargs="+", default=[f"{WORK}/body_vith_b"], help="full members (all fused columns)")
    ap.add_argument("--root-only", nargs="*", default=[], help="members used for root translation / rotation "
                    "and 2D keypoints only (mirrored passes, run_sam3d_body.py --flip)")
    ap.add_argument("--out", default=f"{WORK}/body_ens")
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--member-weight", type=float, default=1.0, help="weight of each member relative to A (1 = mean)")
    ap.add_argument("--max-rot", type=float, default=30.0, help="degrees: larger root disagreement drops the frame")
    ap.add_argument("--window", type=int, default=30, help="B covers scored frames +- window (-1: all frames)")
    ap.add_argument("--scored", default=f"{WORK}/body_v2/scored_frames.json")
    ap.add_argument("--ramp", type=int, default=15)
    ap.add_argument("--keep-a", default="30:50,68:122,132:134",
                    help="pose columns kept from model A: arm chain (clavicle..wrist, 30:50, 132:134) and fingers")
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=3.0, help="pixels")
    ap.add_argument("--max-3d", type=float, default=0.12, help="metres")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mhr = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    members = [(b, False) for b in args.b] + [(b, True) for b in args.root_only]
    for ep in args.episodes:
        s = fuse(ep, args.a, members, args.out, mhr, args)
        rows.append(s)
        print(f"ep {ep}: fused {s['fused']} e2d A {s['e2d_a']:.2f} fused {s['e2d_fused']:.2f} | " + " ".join(
            f"{k}: good {v['good']}/{v['frames']} e2d {v['e2d']:.2f} rot {v['rot_med_deg']:.1f}" for k, v in
            s["members"].items()), flush=True)
    print(f"MEAN e2d A {np.mean([r['e2d_a'] for r in rows]):.3f} fused {np.mean([r['e2d_fused'] for r in rows]):.3f}")
    for k in rows[0]["members"]:
        print(f"  {k}: e2d {np.nanmean([r['members'][k]['e2d'] for r in rows]):.3f} root diff to A "
              f"{np.nanmean([r['members'][k]['rot_med_deg'] for r in rows]):.2f} deg, good "
              f"{sum(r['members'][k]['good'] for r in rows)}/{sum(r['members'][k]['frames'] for r in rows)}")
    with open(os.path.join(args.out, "fusion_stats.json"), "w") as f:
        json.dump(dict(args=vars(args), episodes=rows), f, indent=1)


if __name__ == "__main__":
    main()
