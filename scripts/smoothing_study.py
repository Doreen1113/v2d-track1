"""How much extra temporal smoothing changes the acceleration of the scored quantities.

For a body directory (MHR params) and an assembled episodes directory (object poses), report on the
scored frames of every episode: mean |second difference| of the 22 scored joints / object
translation (cm per frame^2) and the mean displacement caused by the smoothing (cm), for a list of
Gaussian sigmas applied on top of what is stored.

    python smoothing_study.py --body ../../work/body_v2 --episodes-dir <assembled> --sigmas 0 1 2 3 5 8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mhr_utils import MHRForward  # noqa: E402

KIT = Path("/work/doreen071/v2d/kit/v2d_submission_kit")
JOINTS = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]


def scored_frames(sample):
    ids = pd.read_parquet(sample, columns=["row_id"])["row_id"].str.split("_", expand=True)
    ep, fr = ids[1].astype(int), ids[2].astype(int)
    keep = fr != 999999
    return {int(e): np.unique(g.to_numpy()) for e, g in fr[keep].groupby(ep[keep])}


def stretches(frames):
    return [s for s in np.split(frames, np.flatnonzero(np.diff(frames) != 1) + 1) if len(s) > 2]


def acc_cm(x, frames):
    a = [np.linalg.norm(np.diff(x[s], 2, axis=0), axis=-1) for s in stretches(frames)]
    return float(np.concatenate([v.reshape(-1) for v in a]).mean() * 100)


def smooth_pose(pose, sigma):
    if sigma <= 0:
        return pose
    p = pose.copy()
    p[:, 3:] = np.unwrap(p[:, 3:], axis=0)
    return gaussian_filter1d(p, sigma, axis=0, mode="nearest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", type=Path)
    ap.add_argument("--episodes-dir", type=Path)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 1, 2, 3, 5, 8])
    args = ap.parse_args()
    S = scored_frames(KIT / "data/track_1_sample_submission.parquet")
    if args.body:
        mhr = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu")
        res = {s: ([], []) for s in args.sigmas}
        for ep in range(30):
            d = np.load(args.body / f"episode_{ep:06d}.npz")
            sc = d["scales"] if d["scales"].ndim == 1 else np.median(d["scales"], 0)
            sh = d["shape"] if d["shape"].ndim == 1 else np.median(d["shape"], 0)
            f = S[ep]
            lo, hi = max(0, f[0] - 30), min(len(d["pose"]), f[-1] + 31)
            base = None
            for s in args.sigmas:
                _, j = mhr(smooth_pose(d["pose"][lo:hi], s), sc, sh)
                j = j[:, JOINTS]
                ff = f - lo
                if base is None:
                    base = j
                res[s][0].append(acc_cm(j, ff))
                res[s][1].append(float(np.linalg.norm(j[ff] - base[ff], axis=-1).mean() * 100))
        print("body: sigma  joint|acc|  displacement(cm)")
        for s in args.sigmas:
            print(f"  {s:4.1f}  {np.mean(res[s][0]):.4f}  {np.mean(res[s][1]):.2f}")
    if args.episodes_dir:
        res = {s: ([], []) for s in args.sigmas}
        for ep in range(30):
            t = np.load(args.episodes_dir / f"episode_{ep:06d}.npz")["object_translation"]
            f = S[ep]
            for s in args.sigmas:
                ts = gaussian_filter1d(t, s, axis=0, mode="nearest") if s > 0 else t
                res[s][0].append(acc_cm(ts, f))
                res[s][1].append(float(np.linalg.norm(ts[f] - t[f], axis=-1).mean() * 100))
        print("object: sigma  translation|acc|  displacement(cm)")
        for s in args.sigmas:
            print(f"  {s:4.1f}  {np.mean(res[s][0]):.4f}  {np.mean(res[s][1]):.2f}")


if __name__ == "__main__":
    main()
