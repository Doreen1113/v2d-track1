"""Table of eval_body_proxies.py results for several runs (mean over the common episodes, optional per episode).

  python summarize_body_proxies.py DIR_A DIR_B ... [--per-episode KEY]
"""
import argparse
import glob
import json
import os

import numpy as np

KEYS = ["reproj_px", "reproj_px_scored", "iou", "iou_p10", "iou_first", "moge_cv_pct", "moge_drift_pct", "dz_p99_cm",
        "z_acc_cm", "skate_cm", "acc_cm", "acc_root_xy", "acc_root_z", "acc_rel", "first_mpjpe_cm",
        "first_pa_v2v_cm", "first_depth_cm"]


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "episode_*.json")):
        out[int(os.path.basename(f)[8:14])] = json.load(open(f))["metrics"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--per-episode", default=None)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    a = ap.parse_args()
    runs = [load(d) for d in a.dirs]
    eps = sorted(set.intersection(*[set(r) for r in runs]))
    if a.episodes:
        eps = [e for e in eps if e in a.episodes]
    names = [os.path.basename(os.path.normpath(d)) for d in a.dirs]
    print(f"{len(eps)} episodes")
    print(f"{'metric':18s}" + "".join(f"{n[-14:]:>15s}" for n in names))
    for k in KEYS:
        vals = [np.nanmean([r[e].get(k, np.nan) for e in eps]) for r in runs]
        if all(np.isnan(vals)):
            continue
        print(f"{k:18s}" + "".join(f"{v:15.3f}" for v in vals))
    if a.per_episode:
        print(f"\n{a.per_episode} per episode")
        for e in eps:
            print(f"{e:3d}" + "".join(f"{r[e].get(a.per_episode, np.nan):10.3f}" for r in runs))


if __name__ == "__main__":
    main()
