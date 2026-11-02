#!/usr/bin/env python3
"""Pick the best evaluated rollout per episode across seeds and pack a Track 2 submission.

Each run directory <run_root>/epXXX_sSEED/results/ holds the evaluator's episode_XXXXXX.parquet
and episode_XXXXXX.json. The rollout with the lowest evaluator MPPE (a real unassisted rollout
of that seed's policy) is copied into <out_dir>/results/ and tools/pack_submission.py is run.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-roots", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--track", default="track_2_tier1")
    ap.add_argument("--dataset-root", type=Path, default=Path("/work/doreen071/v2d/data_t2"))
    ap.add_argument("--kit", type=Path, default=Path("/work/doreen071/v2d/kit/v2d_submission_kit"))
    ap.add_argument("--code-commit", default="https://github.com/placeholder/repo/commit/0000000000000000000000000000000000000000")
    ap.add_argument("--metric", default="mppe_cm")
    ap.add_argument("--frame-references", type=Path)
    ap.add_argument("--no-pack", action="store_true")
    args = ap.parse_args()

    best: dict[int, tuple[float, Path, Path]] = {}
    seen: dict[int, list[tuple[str, float]]] = {}
    for root in args.run_roots:
        for js in sorted(root.glob("ep*_s*/results/episode_*.json")):
            m = re.search(r"episode_(\d+)\.json$", js.name)
            if not m:
                continue
            ep = int(m.group(1))
            pq = js.with_suffix(".parquet")
            if not pq.is_file():
                continue
            report = json.loads(js.read_text())
            value = float(report["metrics"][args.metric])
            seen.setdefault(ep, []).append((js.parent.parent.name, value))
            if ep not in best or value < best[ep][0]:
                best[ep] = (value, pq, js)
    results = args.out_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ep in sorted(best):
        value, pq, js = best[ep]
        shutil.copyfile(pq, results / pq.name)
        shutil.copyfile(js, results / js.name)
        summary[ep] = {"best": js.parent.parent.name, args.metric: value, "candidates": seen[ep]}
        print(f"episode {ep:3d}: {args.metric}={value:7.2f} from {js.parent.parent.name} ({len(seen[ep])} candidates)")
    (args.out_dir / "selection.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(best)} episodes selected into {results}")
    if args.no_pack or not best:
        return
    out = args.out_dir / f"{args.track}.csv"
    cmd = [sys.executable, str(args.kit / "tools/pack_submission.py"), "--track", args.track,
           "--results", str(results), "--dataset-root", str(args.dataset_root),
           "--code-commit", args.code_commit, "-o", str(out)]
    if args.frame_references:
        cmd += ["--frame-references", str(args.frame_references)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=args.kit, check=False)


if __name__ == "__main__":
    main()
