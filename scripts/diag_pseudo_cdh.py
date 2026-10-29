"""Pseudo CD-H between two body runs (one plays the ground truth): how far a change could move CD-H, and which part
of the change matters. Same metric as sim_cdh_sensitivity.py (Sim(3) on the first scored frame over all vertices,
or over the non-hand vertices with --align-nohands; symmetric Chamfer on a fixed random vertex subset).
Hybrids of the reference with parts of the run:
  full       the run as is
  fingers    reference pose with the run's finger parameters (assemble_episode.FINGER_COLS)
  arms+fing  reference pose with the run's arm chain and finger parameters
  nohands    the run with the reference's arm chain and fingers

  python diag_pseudo_cdh.py --run WORK/sap_body --ref WORK/body_vA
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contactlib  # noqa: E402
from assemble_episode import ARM_COLS, FINGER_COLS  # noqa: E402
from eval_body_proxies import assembled_pose  # noqa: E402
from mhr_utils import MHRForward  # noqa: E402
from sim_cdh_sensitivity import sim3_apply  # noqa: E402

WORK = "/work/doreen071/v2d/work"


def cdh(pred, gt, sub, align):
    p = sim3_apply(pred[0][align], gt[0][align], pred.reshape(-1, 3)).reshape(pred.shape)
    out = []
    for t in range(len(gt)):
        a, b = p[t][sub], gt[t][sub]
        out.append(cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())
    return float(np.mean(out) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--sigma-h", type=float, default=3.0)
    ap.add_argument("--n-verts", type=int, default=3000)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mhr = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu")
    scored = {int(k): v for k, v in json.load(open(f"{WORK}/body_v2/scored_frames.json")).items()}
    hands = np.asarray(contactlib.hand_ids_from(mhr.model))
    rng = np.random.default_rng(0)
    rows = []
    for ep in a.episodes:
        b, r = np.load(f"{a.run}/episode_{ep:06d}.npz"), np.load(f"{a.ref}/episode_{ep:06d}.npz")
        fr = np.array([f for f in scored[ep] if f < len(b["pose"])])
        if len(fr) > a.max_frames:
            fr = fr[np.unique(np.linspace(0, len(fr) - 1, a.max_frames).round().astype(int))]
        pb, pr = assembled_pose(b, a.sigma_h)[fr], assembled_pose(r, a.sigma_h)[fr]
        hyb = {"full": pb}
        x = pr.copy(); x[:, FINGER_COLS] = pb[:, FINGER_COLS]; hyb["fingers"] = x
        x = pr.copy(); x[:, FINGER_COLS + ARM_COLS] = pb[:, FINGER_COLS + ARM_COLS]; hyb["arms+fing"] = x
        x = pb.copy(); x[:, FINGER_COLS + ARM_COLS] = pr[:, FINGER_COLS + ARM_COLS]; hyb["nohands"] = x
        Vr, _ = mhr(pr, r["scales"], r["shape"])
        nv = Vr.shape[1]
        sub = rng.choice(nv, a.n_verts, replace=False)
        nonhand = np.setdiff1d(np.arange(nv), hands)
        row = {"ep": ep}
        for k, p in hyb.items():
            V, _ = mhr(p, b["scales"], b["shape"])
            row[k] = cdh(V, Vr, sub, np.arange(nv))
            row[k + "_alignNH"] = cdh(V, Vr, sub, nonhand)
        rows.append(row)
        print(" ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    keys = [k for k in rows[0] if k != "ep"]
    print("MEAN", " ".join(f"{k}={np.mean([r[k] for r in rows]):.2f}" for k in keys))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
