#!/usr/bin/env python3
"""Write task lists for train_node.sh: one line per (episode, seed) with the step budget.

Usage: make_tasks.py --hmd HMD --steps N --seeds 42 43 --out tasks.txt [--episodes ...] [--chunks K]
With --chunks K the tasks are split round-robin into tasks_0.txt ... tasks_{K-1}.txt next to --out.
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--hmd", type=Path, required=True)
ap.add_argument("--steps", type=int, required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[42])
ap.add_argument("--episodes", type=int, nargs="*")
ap.add_argument("--chunks", type=int, default=1)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
eps = json.loads((a.hmd / "episodes.json").read_text())
lines = []
for seed in a.seeds:
    for e, m in sorted(eps.items(), key=lambda kv: -kv[1]["frames"]):
        if a.episodes and int(e) not in a.episodes:
            continue
        lines.append(f"{int(e)} {m['sequence_id']} {seed} {a.steps}")
if a.chunks == 1:
    a.out.write_text("\n".join(lines) + "\n"); print(a.out, len(lines))
else:
    for k in range(a.chunks):
        p = a.out.with_name(f"{a.out.stem}_{k}{a.out.suffix}")
        p.write_text("\n".join(lines[k::a.chunks]) + "\n"); print(p, len(lines[k::a.chunks]))
