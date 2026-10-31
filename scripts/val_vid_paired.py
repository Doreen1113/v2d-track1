"""Paired CD-H differences of validation runs against reference runs, for both role choices and both episode sets.

  python val_vid_paired.py vid_gvhmr vid_gvhmr_va_rel --refs moge va_moge [--metric cd_h_cm]
For every run R and reference A it reads results/R{,_nohand,_framedrop,_framedrop_nohand}.json and prints
mean(R), mean(R - A) +- standard error over episodes, and the better / worse counts.
"""
import argparse
import json
import os

import numpy as np

RES = "/work/doreen071/v2d_val/results"
SETS = [("", "scored 12, all"), ("_nohand", "scored 12, nohand"), ("_framedrop", "framedrop 7, all"),
        ("_framedrop_nohand", "framedrop 7, nohand")]


def load(name, metric):
    p = os.path.join(RES, name + ".json")
    if not os.path.exists(p):
        return None
    return {int(e): v[metric] for e, v in json.load(open(p))["episodes"].items() if metric in v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--refs", nargs="+", default=["moge", "va_moge"])
    ap.add_argument("--metric", default="cd_h_cm")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    if a.markdown:
        print("| run | set | mean | " + " | ".join(f"vs {r}: diff ± SE (better/worse)" for r in a.refs) + " |")
        print("|---|---|---|" + "---|" * len(a.refs))
    for run in a.runs:
        for suf, label in SETS:
            b = load(run + suf, a.metric)
            if b is None:
                continue
            cells = []
            for ref in a.refs:
                r = load(ref + suf, a.metric)
                if r is None:
                    cells.append("n/a")
                    continue
                eps = sorted(set(b) & set(r))
                d = np.array([b[e] - r[e] for e in eps])
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
                cells.append(f"{d.mean():+.2f} ± {se:.2f} ({(d < 0).sum()}/{(d > 0).sum()})")
            m = np.mean(list(b.values()))
            if a.markdown:
                print(f"| {run} | {label} | {m:.2f} | " + " | ".join(cells) + " |")
            else:
                print(f"{run:28s} {label:20s} mean {m:6.2f}  " + "  ".join(
                    f"vs {ref}: {c}" for ref, c in zip(a.refs, cells)))


if __name__ == "__main__":
    main()
