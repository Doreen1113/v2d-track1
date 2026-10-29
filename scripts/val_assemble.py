"""Assemble a body run on the validation set into scoreable episode NPZs (assemble_episode.py, Track 1 settings).

The body goes through assemble_episode.assemble exactly as for a submission (invalid-frame interpolation, Euler
unwrap, Gaussian pose smoothing sigma_h, median identity). The object is a placeholder (1 cm cube 3 m in front of the
camera) unless --objects points at real per-episode object estimates; val_score.py reports CD-O / ACC-O / PEN for it
but they only mean something with real objects.

  python val_assemble.py --body /work/doreen071/v2d_val/root/work/body_v5_moge \
      --out /work/doreen071/v2d_val/runs/moge/episodes --sigma-h 3
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assemble_episode as asm  # noqa: E402

DUMMY = Path("/work/doreen071/v2d_val/root/work/objects_dummy")


def dummy_objects(episodes):
    import trimesh
    DUMMY.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        p = DUMMY / f"episode_{ep:06d}_pose.json"
        if not p.exists():
            p.write_text(json.dumps({"s": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 3.0]}))
        m = DUMMY / f"episode_{ep:06d}_object.ply"
        if not m.exists():
            trimesh.creation.box((0.01, 0.01, 0.01)).export(m)
    return DUMMY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--sigma-h", type=float, default=3.0)
    ap.add_argument("--sigma-fingers", type=float, default=None)
    ap.add_argument("--sigma-arms", type=float, default=None)
    ap.add_argument("--objects", type=Path, default=None, help="real object estimates (assemble_episode format)")
    ap.add_argument("--sigma-o", type=float, default=12.0)
    a = ap.parse_args()
    eps = a.episodes if a.episodes is not None else sorted(
        int(f[8:14]) for f in os.listdir(a.body) if f.startswith("episode_") and f.endswith(".npz") and len(f) == 18)
    obj = a.objects or dummy_objects(eps)
    for ep in eps:
        T = asm.assemble(ep, a.body, obj, a.out, a.sigma_h, a.sigma_o, list(range(3, 136)), 1.0,
                         sigma_f=a.sigma_fingers, sigma_a=a.sigma_arms)
        print(f"episode {ep}: {T} frames -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
