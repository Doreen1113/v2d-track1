"""Per episode, keep the fix_penetration_micro result only where it is nearly free.

On rim and handle grasps (bowls, pinched thin parts) clearing every hand point needs object pushes of
tens of cm whose constraints flip side from frame to frame; no smoothing weight keeps those pushes from
multiplying the object acceleration. Such episodes keep the finger-only result instead.

    python select_pen_fix.py --fingers EPS --micro EPS_PEN0 --reports a.json b.json --out OUT
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingers", type=Path, required=True, help="episodes after fix_penetration_fingers.py")
    ap.add_argument("--micro", type=Path, required=True, help="episodes after fix_penetration_micro.py")
    ap.add_argument("--reports", type=Path, nargs="+", required=True, help="micro JSON reports")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-acc", type=float, default=0.002,
                    help="largest accepted rise of the mean object |acc| on scored frames, cm/frame^2")
    ap.add_argument("--max-disp", type=float, default=0.5, help="largest mean shift on scored frames, cm")
    a = ap.parse_args()
    rows = {}
    for r in a.reports:
        for row in json.load(open(r)):
            rows[row["episode"]] = row
    a.out.mkdir(parents=True, exist_ok=True)
    choice = {}
    for p in sorted(a.fingers.glob("episode_??????.npz")):
        ep = int(p.stem.split("_")[1])
        row = rows.get(ep)
        ok = row is not None and (
            row["obj_acc_mean_after_cm"] - row["obj_acc_mean_before_cm"] <= a.max_acc
            and row["disp_scored_mean_cm"] <= a.max_disp)
        src = a.micro if ok else a.fingers
        for f in src.glob(f"episode_{ep:06d}*"):
            shutil.copy2(f, a.out / f.name)
        choice[ep] = {"micro": ok, **({k: row[k] for k in ("pen_cm_before", "pen_cm_after",
                      "disp_scored_mean_cm", "obj_acc_mean_before_cm", "obj_acc_mean_after_cm")} if row else {})}
        print(ep, "micro" if ok else "fingers", json.dumps(choice[ep]))
    json.dump(choice, open(a.out.parent / "pen_choice.json", "w"), indent=1)


if __name__ == "__main__":
    main()
