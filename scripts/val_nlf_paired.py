"""Paired comparison of validation runs against reference runs: mean CD-H, paired mean difference +- SE, better /
worse episode counts, for both sets (scored, frame-drop) and both role choices.

  python val_nlf_paired.py --refs moge va_moge --runs nlf nlfraw_ray_va ... \
      [--res /work/doreen071/v2d_val/nlf/results] [--ref-res /work/doreen071/v2d_val/results]
"""
import argparse
import json
import os

import numpy as np


def load(d, name, suffix, metric):
    p = os.path.join(d, f"{name}{suffix}.json")
    if not os.path.exists(p):
        return None
    return {int(e): v[metric] for e, v in json.load(open(p))["episodes"].items() if metric in v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", default=["moge", "va_moge"])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--res", default="/work/doreen071/v2d_val/nlf/results")
    ap.add_argument("--ref-res", default="/work/doreen071/v2d_val/results")
    ap.add_argument("--metric", default="cd_h_cm")
    a = ap.parse_args()
    for setname, s in (("scored (12)", ""), ("frame-drop (7)", "_framedrop")):
        for roles, r in (("all", ""), ("nohand", "_nohand")):
            suf = s + r
            refs = {n: load(a.ref_res, n, suf, a.metric) or load(a.res, n, suf, a.metric) for n in a.refs}
            print(f"\n#### {setname}, roles {roles}\n")
            print("| run | CD-H | " + " | ".join(f"vs {n}: mean +- SE (better/worse)" for n in a.refs) + " |")
            print("|---|---|" + "---|" * len(a.refs))
            for n in a.refs:
                v = refs[n]
                if v:
                    print(f"| {n} | {np.mean(list(v.values())):.2f} |" + " |" * len(a.refs))
            for run in a.runs:
                v = load(a.res, run, suf, a.metric)
                if not v:
                    continue
                cells = []
                for n in a.refs:
                    ref = refs[n]
                    eps = sorted(set(v) & set(ref or {}))
                    d = np.array([v[e] - ref[e] for e in eps])
                    cells.append(f"{d.mean():+.2f} +- {d.std(ddof=1) / np.sqrt(len(d)):.2f} ({(d < 0).sum()}/{(d > 0).sum()})"
                                 if len(d) > 1 else "-")
                print(f"| {run} | {np.mean(list(v.values())):.2f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
