"""Replace pose columns of a refined body run (default: the MHR finger parameters 68:122) by the per-frame
SAM 3D Body values (hand-decoder fingers), interpolated over invalid frames. Everything else is unchanged, so the
output works with assemble_episode.py and make_floor_ratio.py like the input (metrics/ is linked).

  python swap_hand_params.py --body WORK/body_v5 --hands-from WORK/body --out WORK/body_v5h
"""
import argparse
import os
import shutil

import numpy as np

from assemble_episode import fill_invalid, unwrap_euler

WORK = "/work/doreen071/v2d/work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--hands-from", default=f"{WORK}/body")
    ap.add_argument("--cols", default="68:122", help="comma-separated column ranges of pose")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    a = ap.parse_args()
    cols = [c for r in a.cols.split(",") for c in range(*map(int, r.split(":")))]
    os.makedirs(a.out, exist_ok=True)
    for ep in a.episodes:
        b = dict(np.load(f"{a.body}/episode_{ep:06d}.npz"))
        r = np.load(f"{a.hands_from}/episode_{ep:06d}.npz")
        rv = r["valid"].astype(bool)
        raw = unwrap_euler(fill_invalid(r["pose"], rv), list(range(3, 136)))
        pose = b["pose"].astype(np.float64).copy()
        # keep each column on the refined run's 2*pi branch
        d = raw[:, cols] - pose[:, cols]
        raw[:, cols] -= 2 * np.pi * np.round(np.median(d, 0) / (2 * np.pi))
        pose[:, cols] = raw[:, cols]
        b["pose"] = pose.astype(np.float32)
        np.savez_compressed(f"{a.out}/episode_{ep:06d}.npz", **b)
    src = os.path.join(a.body, "metrics")
    dst = os.path.join(a.out, "metrics")
    if os.path.isdir(src) and not os.path.exists(dst):
        shutil.copytree(src, dst)
    print(f"wrote {len(a.episodes)} episodes to {a.out} (columns {a.cols} from {a.hands_from})")


if __name__ == "__main__":
    main()
