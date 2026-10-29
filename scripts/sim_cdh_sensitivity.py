"""CD-H sensitivity to systematic error sources, with a body run as pseudo ground truth.

The scorer fits one Sim(3) on the first scored frame and averages the symmetric Chamfer distance (sum of both
directed means, cm) over the scored frames. Here a body directory (after the assembly smoothing) plays the
ground truth, a perturbed copy plays the prediction, and the same metric is computed on a fixed random vertex
subset. Perturbations:
  focal k          wrong focal (f' = k f) with the same 2D: every frame's body moves from (x, y, z) to (x, y, k z)
                   and turns by the change of its viewing ray (HMR predicts orientation relative to the ray)
  depth_noise s    smooth per-frame root depth error (Gaussian, sd s metres, temporal sigma 15 frames)
  f0_rot d         first scored frame only rotated by d degrees about a horizontal axis (biases the Sim(3))
  tilt_noise d     smooth per-frame lean (sd d degrees) about the feet
  identity DIR     shape / scales of another run (identity uncertainty)
  pose DIR         articulated pose of another run with this run's root trajectory (pose uncertainty)

  python sim_cdh_sensitivity.py --gt WORK/body_v2 --other WORK/body_vith_b --episodes 0 3 12 ...
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_body_proxies import assembled_pose  # noqa: E402
from mhr_utils import DEFAULT_MHR, MHRForward  # noqa: E402

WORK = os.environ.get("V2D_WORK", "/work/doreen071/v2d/work")


def sim3_apply(src0, dst0, pts):
    ms, md = src0.mean(0), dst0.mean(0)
    a, b = src0 - ms, dst0 - md
    u, s, vt = np.linalg.svd(a.T @ b)
    c = np.ones(3)
    c[-1] = -1 if np.linalg.det(u @ vt) < 0 else 1
    R = (u * c) @ vt
    sc = (s * c).sum() / (a * a).sum()
    return sc * (pts - ms) @ R + md


def cdh(pred, gt):
    """pred, gt [T, N, 3]; Sim(3) on frame 0; mean symmetric Chamfer (cm)."""
    p = sim3_apply(pred[0], gt[0], pred.reshape(-1, 3)).reshape(pred.shape)
    out = []
    for t in range(len(gt)):
        d1, _ = cKDTree(gt[t]).query(p[t])
        d2, _ = cKDTree(p[t]).query(gt[t])
        out.append(d1.mean() + d2.mean())
    return float(np.mean(out) * 100)


def rot_about(V, R, c):
    return (V - c) @ R.T + c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=f"{WORK}/body_v2")
    ap.add_argument("--other", default=f"{WORK}/body_vith_b", help="per-frame run for identity / pose swaps")
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--sigma-h", type=float, default=3.0)
    ap.add_argument("--n-verts", type=int, default=2000)
    ap.add_argument("--max-frames", type=int, default=120, help="scored frames used per episode (evenly spaced)")
    ap.add_argument("--scored", default=f"{WORK}/body_v2/scored_frames.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mhr", default=DEFAULT_MHR)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHRForward(path=a.mhr, device=dev)
    scored = {int(k): v for k, v in json.load(open(a.scored)).items()}
    rng = np.random.default_rng(0)
    vsub = np.sort(rng.choice(18439, a.n_verts, replace=False))
    rows = {}
    for ep in a.episodes:
        b = np.load(f"{a.gt}/episode_{ep:06d}.npz")
        raw = np.load(f"{WORK}/body/episode_{ep:06d}.npz")
        K = np.array([float(raw["fx"]), float(raw["fy"]), float(raw["cx"]), float(raw["cy"])])
        pose = assembled_pose(b, a.sigma_h)
        sc = np.array(scored[ep])
        sel = sc[np.linspace(0, len(sc) - 1, min(a.max_frames, len(sc))).round().astype(int)]
        sel = np.unique(np.r_[sc[0], sel])  # keep the first scored frame first
        V, J = mhr(pose[sel], b["scales"], b["shape"])
        gt = V[:, vsub]
        pel = J[:, 1]
        r = {}
        # wrong focal
        for k in (0.9, 0.95, 1.05, 1.1):
            P = gt.copy()
            for i in range(len(sel)):
                u = K[0] * pel[i, 0] / pel[i, 2] + K[2]
                v = K[1] * pel[i, 1] / pel[i, 2] + K[3]
                ray = np.array([(u - K[2]) / K[0], (v - K[3]) / K[1], 1.0])
                ray2 = np.array([(u - K[2]) / (k * K[0]), (v - K[3]) / (k * K[1]), 1.0])
                ray, ray2 = ray / np.linalg.norm(ray), ray2 / np.linalg.norm(ray2)
                ax = np.cross(ray, ray2)
                ang = np.arcsin(np.clip(np.linalg.norm(ax), 0, 1))
                R = Rotation.from_rotvec(ax / max(np.linalg.norm(ax), 1e-12) * ang).as_matrix()
                P[i] = rot_about(P[i], R, pel[i]) + np.array([0, 0, (k - 1) * pel[i, 2]])
            r[f"focal_{k}"] = cdh(P, gt)
        # smooth root depth noise
        for s in (0.02, 0.05):
            vals = []
            for rep in range(3):
                e = gaussian_filter1d(rng.normal(size=len(sel)), 15.0 * len(sel) / max(len(sc), 1) + 1e-3)
                e = e / (e.std() + 1e-9) * s
                vals.append(cdh(gt + np.stack([np.zeros_like(e), np.zeros_like(e), e], 1)[:, None], gt))
            r[f"depth_noise_{s}"] = float(np.mean(vals))
        # first scored frame rotation (pitch about the camera x axis through its centroid)
        for d in (2.0, 5.0):
            P = gt.copy()
            P[0] = rot_about(P[0], Rotation.from_euler("x", d, degrees=True).as_matrix(), P[0].mean(0))
            r[f"f0_rot_{d}"] = cdh(P, gt)
        # smooth per-frame lean about the feet (lowest point)
        for d in (2.0, 4.0):
            P = gt.copy()
            e = gaussian_filter1d(rng.normal(size=(len(sel), 2)), 3.0, axis=0)
            e = e / (e.std(0) + 1e-9) * d
            for i in range(len(sel)):
                foot = P[i][np.argmax(P[i][:, 1])]
                P[i] = rot_about(P[i], Rotation.from_euler("xz", e[i], degrees=True).as_matrix(), foot)
            r[f"tilt_noise_{d}"] = cdh(P, gt)
        # identity / pose of the other run
        if a.other and os.path.exists(f"{a.other}/episode_{ep:06d}.npz"):
            o = np.load(f"{a.other}/episode_{ep:06d}.npz")
            ov = o["valid"].astype(bool)
            shp = np.median(o["shape"][ov], 0) if o["shape"].ndim == 2 else o["shape"]
            scl = np.median(o["scales"][ov], 0) if o["scales"].ndim == 2 else o["scales"]
            Vi, Ji = mhr(pose[sel], scl, shp)
            Vi = Vi - Ji[:, 1:2] + J[:, 1:2]  # same pelvis trajectory
            r["identity_other"] = cdh(Vi[:, vsub], gt)
            if ov[sel].all():
                po = o["pose"][sel].astype(np.float64).copy()
                po[:, :3] = pose[sel, :3]
                Vp, Jp = mhr(po, b["scales"], b["shape"])
                Vp = Vp - Jp[:, 1:2] + J[:, 1:2]
                r["pose_other"] = cdh(Vp[:, vsub], gt)
        rows[ep] = r
        print(ep, " ".join(f"{k}={v:.2f}" for k, v in r.items()), flush=True)
    keys = sorted({k for r in rows.values() for k in r})
    print("MEAN " + " ".join(f"{k}={np.nanmean([rows[e].get(k, np.nan) for e in rows]):.2f}" for k in keys))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
