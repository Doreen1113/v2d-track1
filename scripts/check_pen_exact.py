"""Scorer-exact hand-object penetration check of assembled Track 1 episodes.

Runs the leaderboard's own code path on every scored frame: the scorer's float64 MHR forward
(metric_code/track_1/PEN.py::_MHRBody, with an equivalent asset rebuilt from the official MHR
v1.0.1 model), the scorer's rotation projection, and the scorer's _penetration_depth (winding-number
inside test, exact point-triangle depth) against the object mesh exactly as the packer submits it
(v2dlb/mesh_budget.py). Hand points are every vertex skinned to a wrist subtree, a superset of the
scorer's 512 hand points, so zero here implies zero on the leaderboard.

    python check_pen_exact.py --episodes-dir EPS [--episodes ...] [--workers N] [--json OUT]

Per episode it prints the PEN value (cm, alignment scale 1), the number of scored frames with any
hand point inside, the deepest point (mm) and the smallest clearance of any hand point within
5 mm of the surface (mm).
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

MESH_SUFFIXES = (".glb", ".obj", ".ply", ".stl", ".off")
_STATE = {}


def _init(threads):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import torch  # noqa: F401  (only to read the official model's buffers)
    from check_scorer_mhr import asset_from_torchscript
    from fix_penetration import hand_vertex_ids, load_pen, sample_budget
    from mhr_utils import DEFAULT_MHR
    import contactlib as cl
    torch.set_num_threads(threads)
    pen = load_pen()
    asset = asset_from_torchscript(DEFAULT_MHR)
    _STATE["pen"] = pen
    _STATE["body"] = pen._MHRBody(asset)
    model = torch.jit.load(DEFAULT_MHR, map_location="cpu")
    _STATE["hand_ids"] = hand_vertex_ids(model)
    _STATE["budget"] = sample_budget(Path(cl.SAMPLE))
    _STATE["scored"] = cl.scored()


def check_episode(path):
    from v2dlb.mesh_budget import budget_mesh
    pen, body, hand_ids = _STATE["pen"], _STATE["body"], _STATE["hand_ids"]
    path = Path(path)
    ep = int(path.stem.split("_")[1])
    data = dict(np.load(path))
    frames = _STATE["scored"][ep]
    mesh = next(p for s in MESH_SUFFIXES if (p := path.with_name(path.stem + "_object" + s)).exists())
    mv, mf = budget_mesh(str(mesh), *_STATE["budget"])
    mf = mf.astype(np.int64)
    pose = np.asarray(data["pose"], float)[frames]
    params = np.concatenate([pose, np.broadcast_to(data["scales"], (len(pose), 68))], 1)
    hands = []
    for a in range(0, len(params), 32):
        v, _ = body(params[a:a + 32], data["shape"], chunk=32)
        hands.append(v[:, hand_ids])
    hands = np.concatenate(hands)
    R = pen._rotations(np.asarray(data["object_rotation"], float)[frames])
    t = np.asarray(data["object_translation"], float)[frames]
    s = float(np.asarray(data["object_scale"]).reshape(()))
    local = np.einsum("tpi,tij->tpj", hands - t[:, None], R) / s
    flat = local.reshape(-1, 3)
    depth = pen._penetration_depth(flat, mv, mf).reshape(local.shape[:2])
    value = float(np.mean(pen.sdf_interpenetration_distance_per_frame(-depth * s)))
    # clearance of outside points close to the surface (scorer-style exact distance)
    lo, hi = mv.min(0) - 0.005 / s, mv.max(0) + 0.005 / s
    near = np.flatnonzero(((flat >= lo) & (flat <= hi)).all(1) & (depth.reshape(-1) == 0))
    clear = np.inf
    if len(near):
        area = np.linalg.norm(np.cross(mv[mf[:, 1]] - mv[mf[:, 0]], mv[mf[:, 2]] - mv[mf[:, 0]]), axis=1)
        f = mf[area > 0]
        dist = pen._in_parallel(pen._point_triangle_distance, np.ascontiguousarray(flat[near]),
                                mv[f[:, 0]], mv[f[:, 1]], mv[f[:, 2]])
        clear = float(dist.min() * s)
    return {"episode": ep, "pen_cm": value, "frames_inside": int((depth > 0).any(1).sum()),
            "points_inside": int((depth > 0).sum()), "max_depth_mm": float(depth.max() * s * 1000),
            "min_clearance_mm": clear * 1000, "scored_frames": int(len(frames))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-dir", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8, help="numpy threads per worker")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    eps = a.episodes if a.episodes is not None else sorted(
        int(p.stem.split("_")[1]) for p in a.episodes_dir.glob("episode_??????.npz"))
    paths = [a.episodes_dir / f"episode_{e:06d}.npz" for e in eps]
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.threads,)) as pool:
        for r in pool.map(check_episode, paths):
            rows.append(r)
            print(f"episode {r['episode']:2d}: PEN {r['pen_cm']:.7f} cm  frames inside {r['frames_inside']:3d}/"
                  f"{r['scored_frames']}  points {r['points_inside']:5d}  max depth {r['max_depth_mm']:.3f} mm  "
                  f"min clearance {r['min_clearance_mm']:.3f} mm", flush=True)
    mean = float(np.mean([r["pen_cm"] for r in rows]))
    print(f"mean PEN over {len(rows)} episodes: {mean:.7f} cm; episodes with any point inside: "
          f"{sum(r['frames_inside'] > 0 for r in rows)}")
    if a.json:
        json.dump({"rows": rows, "mean_pen_cm": mean}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
