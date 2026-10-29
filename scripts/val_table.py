"""Markdown table of val_score.py results: one column per run, one row per sequence, plus the mean.

  python val_table.py RESULTS_DIR/moge.json RESULTS_DIR/gt.json ... [--metric cd_h_cm acc_h_cm] [--names a b ...]
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--metric", nargs="+", default=["cd_h_cm", "acc_h_cm"])
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--diff", action="store_true", help="also print each run minus the first run")
    a = ap.parse_args()
    runs = [json.load(open(p)) for p in a.results]
    names = a.names or [os.path.splitext(os.path.basename(p))[0] for p in a.results]
    eps = sorted({int(e) for r in runs for e in r["episodes"]})
    seq = {}
    for r in runs:
        for e, v in r["episodes"].items():
            seq[int(e)] = v.get("sequence", "")
    for m in a.metric:
        fmt = "{:.4f}" if m.startswith("acc") else "{:.2f}"
        print(f"\n### {m}\n")
        print("| ep | sequence | " + " | ".join(names) + " |")
        print("|---|---|" + "---|" * len(names))
        cols = []
        for r in runs:
            cols.append([r["episodes"].get(str(e), {}).get(m, np.nan) for e in eps])
        cols = np.array(cols, float)
        for i, e in enumerate(eps):
            cells = [fmt.format(c) for c in cols[:, i]]
            if a.diff:
                cells = [cells[0]] + [f"{fmt.format(c)} ({c - cols[0, i]:+.2f})" for c in cols[1:, i]]
            print(f"| {e} | {seq.get(e, '')} | " + " | ".join(cells) + " |")
        both = ~np.isnan(cols).any(0)
        mean = cols[:, both].mean(1)
        cells = [fmt.format(c) for c in mean]
        if a.diff:
            cells = [cells[0]] + [f"{fmt.format(c)} ({c - mean[0]:+.2f})" for c in mean[1:]]
        print(f"| **mean** ({both.sum()} eps) | | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
