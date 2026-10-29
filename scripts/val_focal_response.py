"""CD-H / ACC-H as a function of the focal used by the body run, on the validation set.

Each run is given as NAME=RESULT_JSON:INTR_DIR; the focal ratio of an episode is fx(INTR_DIR) / fx(GT). Per episode a
parabola CD-H(ratio) is fitted through the runs and its minimum reported (the focal the pipeline would like), plus
the CD-H slope per +1% of focal around the GT focal.

  python val_focal_response.py moge=R/moge.json:intrinsics gt=R/gt.json:intrinsics_gt \
      gt095=R/gt095.json:intrinsics_gt095 gt105=R/gt105.json:intrinsics_gt105
"""
import argparse
import json
import os

import numpy as np

ROOT_WORK = "/work/doreen071/v2d_val/root/work"
GT = "/work/doreen071/v2d_val/gt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--metric", default="cd_h_cm")
    a = ap.parse_args()
    runs = []
    for spec in a.runs:
        name, rest = spec.split("=", 1)
        res, intr = rest.rsplit(":", 1)
        runs.append((name, json.load(open(res))["episodes"], intr))
    eps = sorted({int(e) for _, r, _ in runs for e in r})
    print(f"| ep | sequence | " + " | ".join(f"{n} f/f_gt, {a.metric}" for n, _, _ in runs)
          + " | best f/f_gt | slope per +1% |")
    print("|---|---|" + "---|" * (len(runs) + 2))
    best, slopes = [], []
    for e in eps:
        fgt = json.load(open(os.path.join(ROOT_WORK, "intrinsics_gt", f"episode_{e:06d}.json")))["fx"]
        xs, ys, cells = [], [], []
        for n, r, intr in runs:
            if str(e) not in r:
                cells.append("-")
                continue
            f = json.load(open(os.path.join(ROOT_WORK, intr, f"episode_{e:06d}.json")))["fx"]
            xs.append(f / fgt)
            ys.append(r[str(e)][a.metric])
            cells.append(f"{f / fgt:.3f}: {ys[-1]:.2f}")
        seq = next((r[str(e)].get("sequence", "") for _, r, _ in runs if str(e) in r), "")
        if len(xs) >= 3:
            c = np.polyfit(xs, ys, 2)
            xb = -c[1] / (2 * c[0]) if c[0] > 0 else float("nan")
            sl = (2 * c[0] * 1.0 + c[1]) * 0.01
            best.append(xb)
            slopes.append(sl)
            tail = f"{xb:.3f} | {sl:+.3f}"
        else:
            tail = "- | -"
        print(f"| {e} | {seq} | " + " | ".join(cells) + f" | {tail} |")
    if best:
        print(f"\nmedian best f/f_gt {np.nanmedian(best):.3f} (mean {np.nanmean(best):.3f}); "
              f"median CD-H slope at f_gt {np.median(slopes):+.3f} cm per +1% focal")


if __name__ == "__main__":
    main()
