"""Sanity checks of val_score.py on the validation ground truth.

  identity    GT parameters scored against themselves (expect exactly 0)
  numpy       the torch path against the kit's own numpy _episode_metrics on a few frames (expect ~1e-12)
  focal k     the perturbation of sim_cdh_sensitivity.py: every frame's points move from (x, y, z) to (x, y, k z)
              about the pelvis and turn by the change of the pelvis viewing ray (same 2D, wrong focal)
  depth s     smooth per-frame root depth error, sd s metres (temporal sigma 15 frames), 3 draws
  f0rot d     the first scored frame only, pitched by d degrees about its centroid (biases the Sim(3))
Perturbations act on the arrays the scorer compares (alignment vertices, body surface, joints), as in
sim_cdh_sensitivity.py, so they need no parameter refit.

  python val_check_scorer.py --gt /work/doreen071/v2d_val/gt --out /work/doreen071/v2d_val/results/scorer_check.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import val_score as vs  # noqa: E402

HUMAN_KEYS = ("mhr_vertices", "human_surface_points", "mhr_joints")


def rot_about(P, R, c):
    return (P - c) @ R.T + c


def focal_perturb(arr, K, k):
    out = {key: np.array(v, copy=True) for key, v in arr.items()}
    pel = arr["mhr_joints"][:, 0]  # MHR joint 1 (first of the 22 scored joints)
    fx, fy, cx, cy = K
    for i in range(len(pel)):
        u, v = fx * pel[i, 0] / pel[i, 2] + cx, fy * pel[i, 1] / pel[i, 2] + cy
        r1 = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        r2 = np.array([(u - cx) / (k * fx), (v - cy) / (k * fy), 1.0])
        r1, r2 = r1 / np.linalg.norm(r1), r2 / np.linalg.norm(r2)
        ax = np.cross(r1, r2)
        ang = np.arcsin(np.clip(np.linalg.norm(ax), 0, 1))
        R = Rotation.from_rotvec(ax / max(np.linalg.norm(ax), 1e-12) * ang).as_matrix()
        for key in HUMAN_KEYS:
            out[key][i] = rot_about(arr[key][i], R, pel[i]) + np.array([0, 0, (k - 1) * pel[i, 2]])
    return out


def depth_perturb(arr, s, rng):
    T = len(arr["mhr_joints"])
    e = gaussian_filter1d(rng.normal(size=T), 15.0)
    e = e / (e.std() + 1e-9) * s
    out = {key: np.array(v, copy=True) for key, v in arr.items()}
    for key in HUMAN_KEYS:
        out[key][..., 2] += e[:, None]
    return out


def f0_perturb(arr, d):
    out = {key: np.array(v, copy=True) for key, v in arr.items()}
    R = Rotation.from_euler("x", d, degrees=True).as_matrix()
    c = arr["human_surface_points"][0].mean(0)
    for key in HUMAN_KEYS:
        out[key][0] = rot_about(arr[key][0], R, c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="/work/doreen071/v2d_val/gt")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--roles", default="all", choices=["all", "nohand", "sub"])
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--numpy-frames", type=int, default=8, help="frames for the numpy cross-check (slow on CPU)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    kit = vs.load_kit("CD-H")
    body = vs.build_body(kit, vs.DEFAULT_MHR, a.roles)
    scored = {int(k): np.asarray(v, int) for k, v in json.load(open(os.path.join(a.gt, "scored_frames.json"))).items()}
    eps = a.episodes if a.episodes is not None else sorted(scored)
    rng = np.random.default_rng(0)
    rows = {}
    for ep in eps:
        fr = scored[ep][: a.max_frames] if a.max_frames else scored[ep]
        z = np.load(os.path.join(a.gt, f"episode_{ep:06d}.npz"))
        K = (float(z["fx"]), float(z["fy"]), float(z["cx"]), float(z["cy"]))
        gt, _, _ = vs.episode_arrays(kit, body, a.gt, ep, fr)
        m = lambda p: vs.episode_metrics(kit, p, gt, fr, a.device, 4096, human_only=True)  # noqa: E731
        r = {}
        same = m(gt)
        r["identity_cd_h"], r["identity_acc_h"] = same["cd_h_cm"], same["acc_h_cm"]
        if a.numpy_frames:
            n = a.numpy_frames
            sub = {k_: v[:n] for k_, v in focal_perturb(gt, K, 1.05).items()}
            gsub = {k_: v[:n] for k_, v in gt.items()}
            ref = kit._episode_metrics(sub, gsub, np.asarray(fr[:n]), None)
            tor = vs.episode_metrics(kit, sub, gsub, fr[:n], a.device, 4096)  # full series (all five metrics)
            r["numpy_vs_torch_max_abs"] = max(abs(ref[k_] - tor[k_]) for k_ in ref)
        for k in (0.9, 0.95, 1.05, 1.1):
            q = m(focal_perturb(gt, K, k))
            r[f"focal_{k}_cd_h"], r[f"focal_{k}_acc_h"] = q["cd_h_cm"], q["acc_h_cm"]
        for s in (0.02, 0.05):
            r[f"depth_{s}_cd_h"] = float(np.mean([m(depth_perturb(gt, s, rng))["cd_h_cm"] for _ in range(3)]))
        for d in (2.0, 5.0):
            r[f"f0rot_{d}_cd_h"] = m(f0_perturb(gt, d))["cd_h_cm"]
        rows[ep] = r
        print(ep, " ".join(f"{k_}={v:.4g}" for k_, v in r.items()), flush=True)
    keys = list(next(iter(rows.values())))
    mean = {k_: float(np.mean([rows[e][k_] for e in rows])) for k_ in keys}
    print("MEAN " + " ".join(f"{k_}={v:.4g}" for k_, v in mean.items()))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"roles": a.roles, "mean": mean, "episodes": {str(k_): v for k_, v in rows.items()}},
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
