"""Table of before/after metrics written by refine_body_sequence.py (<v2>/metrics/episode_XXXXXX.json).

  python summarize_body_v2.py [--v2 DIR] [--tag SUFFIX]
"""
import argparse
import glob
import json
import os

import numpy as np

KEYS = [("depth_dz_p99_cm", "dz99"), ("depth_acc_cm", "zacc"), ("reproj_vit_px", "2D"),
        ("skate_cm_per_frame", "skate"), ("acc22_cm", "acc22")]
RUNS = ["raw", "v1", "v2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="/work/doreen071/v2d/work/body_v2")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.v2, "metrics", f"episode_??????{args.tag}.json")))
    rows = [json.load(open(f)) for f in files]
    head = "ep  " + " | ".join(f"{short:>17s}" for _, short in KEYS) + " | contact L/R  floor_ratio"
    print(" " * 4 + " | ".join(f"{'raw/v1/v2':>17s}" for _ in KEYS))
    print(head)
    agg = {r: {k: [] for k, _ in KEYS} for r in RUNS}
    for d in rows:
        cells = []
        for k, _ in KEYS:
            vals = [d[r][k] for r in RUNS]
            for r, v in zip(RUNS, vals):
                agg[r][k].append(v)
            cells.append("/".join(f"{v:5.2f}" for v in vals))
        c = d.get("contact_frac", [np.nan, np.nan])
        print(f"{d['episode']:2d}  " + " | ".join(f"{c_:>17s}" for c_ in cells)
              + f" | {c[0]:.2f}/{c[1]:.2f}  {d.get('floor_moge_over_body', np.nan):.3f}")
    print("mean " + " | ".join("/".join(f"{np.nanmean(agg[r][k]):5.2f}" for r in RUNS) for k, _ in KEYS))


if __name__ == "__main__":
    main()
