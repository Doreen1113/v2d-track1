"""Upright / first-frame proxies for one or more body directories (body_v2 npz format with floor_n, floor_d).

Standing frames are detected once on the reference directory (refine_body_v5.detect_standing with its contacts)
and used for every run, so the numbers are comparable. Per episode:
  tilt_med / tilt_p90        spine (hip centre -> neck) angle to the floor normal on standing frames (deg)
  tilt_med_sc                same on standing scored frames
  lean_cam_sd                per-frame SD of the spine lean towards / away from the camera on standing frames (deg)
  lean_cam_med               its median (+ = top of the spine away from the camera)
  hh_sd_cm                   SD of the hip-centre height above the floor on standing frames
  f0_standing, f0_tilt       first scored frame: standing flag and spine tilt
  f0_mpjpe_cm, f0_rot_deg, f0_depth_cm   first scored frame change w.r.t. the reference (22 joints camera frame,
                             torso orientation from hips and neck, pelvis depth)
With --sigma-h the pose is smoothed as in assemble_episode.py first.

  python eval_upright.py --ref WORK/body_v2 WORK/body_v2 WORK/body_v5a --sigma-h 3 --out WORK/body_v5a/upright.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_body_proxies import assembled_pose  # noqa: E402
from refine_body_sequence import J22, Body, forward_np  # noqa: E402
from refine_body_v5 import DEFAULT_CFG, detect_standing, upright_geometry  # noqa: E402

WORK = "/work/doreen071/v2d/work"


def run_kp(body_dir, ep, sigma_h, dev):
    b = np.load(os.path.join(body_dir, f"episode_{ep:06d}.npz"))
    pose = assembled_pose(b, sigma_h)
    body = Body(dev, b["shape"], b["scales"])
    kp, j = forward_np(body, pose.astype(np.float32))
    return b, pose, kp, j


def torso_frame(kp, t):
    """Orthonormal torso frame: x hips left -> right, y hip centre -> neck, z = x cross y."""
    x = kp[t, 10] - kp[t, 9]
    y = kp[t, 69] - kp[t, [9, 10]].mean(0)
    x /= np.linalg.norm(x)
    y -= x * (x @ y)
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ref", default=f"{WORK}/body_v2")
    ap.add_argument("--sigma-h", type=float, default=0.0)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--scored", default=f"{WORK}/body_v2/scored_frames.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scored = {int(k): v for k, v in json.load(open(a.scored)).items()}
    res = {d: {} for d in a.dirs}
    for ep in a.episodes:
        rb, rpose, rkp, rj = run_kp(a.ref, ep, a.sigma_h, dev)
        floor = (rb["floor_n"].astype(np.float64), float(rb["floor_d"]))
        valid = rb["valid"].astype(bool)
        st, _, _ = detect_standing(rkp, floor, rb["contact"].astype(bool), valid, DEFAULT_CFG)
        T = len(rkp)
        sc = np.zeros(T, bool)
        sc[[f for f in scored[ep] if f < T]] = True
        f0 = scored[ep][0]
        n = floor[0]
        camf = np.array([0, 0, 1.0]) - n * n[2]
        camf /= np.linalg.norm(camf)
        for d in a.dirs:
            b, pose, kp, j = run_kp(d, ep, a.sigma_h, dev)
            fl = (b["floor_n"].astype(np.float64), float(b["floor_d"]))  # each run's own floor offset
            g = upright_geometry(kp, (n, fl[1]))
            spine = kp[:, 69] - kp[:, [9, 10]].mean(1)
            su = spine / np.linalg.norm(spine, axis=1, keepdims=True)
            lean = np.degrees(np.arcsin(np.clip(su @ camf, -1, 1)))
            m = {}
            if st.any():
                m["tilt_med"] = float(np.median(g["tilt"][st]))
                m["tilt_p90"] = float(np.percentile(g["tilt"][st], 90))
                m["lean_cam_sd"] = float(np.std(lean[st]))
                m["lean_cam_med"] = float(np.median(lean[st]))
                m["hh_sd_cm"] = float(np.std(g["hh"][st]) * 100)
            if (st & sc).any():
                m["tilt_med_sc"] = float(np.median(g["tilt"][st & sc]))
            m["standing_frac"] = float(st.mean())
            m["f0_standing"] = float(st[f0])
            m["f0_tilt"] = float(g["tilt"][f0])
            m["f0_mpjpe_cm"] = float(np.linalg.norm(j[f0, J22] - rj[f0, J22], axis=-1).mean() * 100)
            dR = torso_frame(kp, f0) @ torso_frame(rkp, f0).T
            m["f0_rot_deg"] = float(np.degrees(Rotation.from_matrix(dR).magnitude()))
            m["f0_depth_cm"] = float((j[f0, 1, 2] - rj[f0, 1, 2]) * 100)
            res[d][ep] = m
        print(ep, " | ".join(" ".join(f"{k}={v:.2f}" for k, v in res[d][ep].items() if k in
                                      ("tilt_med", "lean_cam_sd", "f0_tilt", "f0_mpjpe_cm")) for d in a.dirs), flush=True)
    keys = ["tilt_med", "tilt_p90", "tilt_med_sc", "lean_cam_sd", "lean_cam_med", "hh_sd_cm", "standing_frac",
            "f0_standing", "f0_tilt", "f0_mpjpe_cm", "f0_rot_deg", "f0_depth_cm"]
    print(f"{'metric':14s}" + "".join(f"{os.path.basename(os.path.normpath(d))[-14:]:>15s}" for d in a.dirs))
    summary = {}
    for k in keys:
        vals = [np.nanmean([res[d][e].get(k, np.nan) for e in a.episodes]) for d in a.dirs]
        summary[k] = dict(zip(a.dirs, vals))
        print(f"{k:14s}" + "".join(f"{v:15.3f}" for v in vals))
    # first frame tilt where the first scored frame is standing
    f0s = [e for e in a.episodes if res[a.dirs[0]][e]["f0_standing"] > 0]
    vals = [np.mean([res[d][e]["f0_tilt"] for e in f0s]) if f0s else np.nan for d in a.dirs]
    print(f"{'f0_tilt_stand':14s}" + "".join(f"{v:15.3f}" for v in vals) + f"   ({len(f0s)} episodes)")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(per_episode={d: res[d] for d in a.dirs}, summary=summary, sigma_h=a.sigma_h), f, indent=1)


if __name__ == "__main__":
    main()
