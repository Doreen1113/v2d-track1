"""Remove hand-object interpenetration from assembled Track 1 episodes, in place.

For every frame whose hand vertices fall inside the (submission-budgeted) object mesh, the object
is translated by the smallest offset that moves every hand vertex out of it (iterated a few
times). The penetration test is the scorer's own (metric_code/track_1/PEN.py), run on the same
decimated mesh the packer submits.

    python fix_penetration.py --episodes-dir OUT [--episodes 0 1 ...]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mhr_utils import DEFAULT_MHR, MHRForward  # noqa: E402

KIT = Path("/work/doreen071/v2d/kit/v2d_submission_kit")
sys.path.insert(0, str(KIT))
from v2dlb.mesh_budget import budget_mesh  # noqa: E402

WRISTS = (41, 77)  # MHR joints whose subtrees are the left / right hands
MESH_SUFFIXES = (".glb", ".obj", ".ply", ".stl", ".off")


def load_pen():
    spec = importlib.util.spec_from_file_location("pen", KIT / "metric_code/track_1/PEN.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hand_vertex_ids(model):
    b = {k: v.cpu().numpy() for k, v in model.named_buffers()}
    c = "character_torch."
    parents = b[c + "skeleton.joint_parents"]
    hand = set(WRISTS)
    for j in range(len(parents)):
        p = parents[j]
        while p >= 0:
            if p in WRISTS:
                hand.add(j)
                break
            p = parents[p]
    nv = b[c + "mesh.rest_vertices"].shape[0]
    w = np.zeros((nv, len(parents)))
    np.add.at(w, (b[c + "linear_blend_skinning.vert_indices_flattened"],
                  b[c + "linear_blend_skinning.skin_indices_flattened"]),
              b[c + "linear_blend_skinning.skin_weights_flattened"])
    return np.flatnonzero(w[:, sorted(hand)].sum(1) > 0.5)


def scored_frames(sample_path):
    import pandas as pd
    ids = pd.read_parquet(sample_path, columns=["row_id"])["row_id"]
    parts = ids.str.split("_", expand=True)
    ep, fr = parts[1].astype(int), parts[2].astype(int)
    keep = fr != 999999
    return {int(e): np.unique(g.to_numpy()) for e, g in fr[keep].groupby(ep[keep])}


def sample_budget(sample_path):
    import pandas as pd
    from v2dlb import mhr_submission as ms
    ids = pd.read_parquet(sample_path, columns=["row_id"])["row_id"]
    parts = ids.str.split("_", expand=True)
    first = parts[1] == parts[1].iloc[0]
    order = ms.array_order()
    arr = parts[3][first].astype(int)
    return (int((arr == order.index("object_mesh_faces")).sum()),
            int((arr == order.index("object_mesh_vertices")).sum()))


def depths(pen, local, mv, mf):
    """Scorer penetration depth, evaluated only for points inside the mesh's bounding box."""
    flat = local.reshape(-1, 3)
    lo, hi = mv.min(0) - 1e-6, mv.max(0) + 1e-6
    near = np.flatnonzero(((flat >= lo) & (flat <= hi)).all(1))
    out = np.zeros(len(flat))
    if len(near):
        out[near] = pen._penetration_depth(flat[near], mv, mf)
    return out.reshape(local.shape[:2])


def candidate_directions():
    """26 unit directions of the 3x3x3 grid (mesh coordinates)."""
    g = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                  if (x, y, z) != (0, 0, 0)], float)
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def min_clearing_step(pen, pts, d, mv, mf, start, iters):
    """Smallest step along d (mesh units) that takes every point out of the mesh, or inf."""
    def clear(step):
        return not (depths(pen, (pts - d * step)[None], mv, mf) > 0).any()

    hi = max(start, 1e-4)
    for _ in range(iters):
        if clear(hi):
            break
        hi *= 2
    else:
        return np.inf
    lo = 0.0
    for _ in range(10):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if clear(mid) else (mid, hi)
    return hi


def required_push(pen, local, depth, R, s, mv, mf, iters, search=True):
    """Per penetrating frame: the smallest scene-space object offset that clears the hands.

    Candidate directions are the direction from the penetrating hand points to the mesh centre
    plus the 26 grid directions; for each, the clearing step is found by doubling then bisection
    with the scorer's own depth test, and the shortest one is kept. Only the hand points near
    the mesh are tested, which keeps this cheap.
    """
    centre = mv.mean(0)
    dirs = candidate_directions() if search else np.zeros((0, 3))
    lo_box, hi_box = mv.min(0), mv.max(0)
    push = np.zeros((len(local), 3))
    for f in np.flatnonzero((depth > 0).any(1)):
        pad = 2 * float(depth[f].max()) + 0.05 * np.linalg.norm(hi_box - lo_box)
        near = ((local[f] >= lo_box - pad) & (local[f] <= hi_box + pad)).all(1)
        pts = local[f][near]
        d0 = centre - local[f][depth[f] > 0].mean(0)
        d0 /= np.linalg.norm(d0) + 1e-12
        best, best_d = np.inf, d0
        for d in np.vstack([d0[None], dirs]):
            step = min_clearing_step(pen, pts, d, mv, mf, float(depth[f].max()),
                                     iters if np.isinf(best) else 3)
            # a direction needing more than the current best is abandoned after few doublings
            if step < best:
                best, best_d = step, d
        if np.isinf(best):
            best = min_clearing_step(pen, local[f], d0, mv, mf, float(depth[f].max()), iters + 6)
            best_d = d0
        # hand moves by -d*step in mesh units  <=>  object moves by +s*R d*step in the scene
        push[f] = R[f] @ best_d * (best * s)
    return push


def smooth_envelope(push, sigma):
    """Temporally smooth offset that still reaches every frame's required push.

    Each frame takes the largest required push within +-2 sigma (a vector max filter), then a
    Gaussian blur; the blur can only lower the peak near isolated frames, which the outer loop
    tops up on the next round.
    """
    T = len(push)
    w = int(round(2 * sigma))
    norm = np.linalg.norm(push, axis=1)
    env = np.zeros_like(push)
    for f in range(T):
        a, b = max(0, f - w), min(T, f + w + 1)
        env[f] = push[a + int(np.argmax(norm[a:b]))]
    return gaussian_filter1d(env, sigma, axis=0, mode="nearest")


def fix_episode(path, pen, mhr, hand_ids, budget, margin, iters, frames=None, sigma=0.0, rounds=6,
                search=False):
    data = dict(np.load(path))
    mesh = next(p for s in MESH_SUFFIXES if (p := path.with_name(path.stem + "_object" + s)).exists())
    mv, mf = budget_mesh(mesh, *budget)
    mf = mf.astype(np.int64)
    T = len(data["pose"])
    frames = np.arange(T) if frames is None else np.asarray(frames)
    verts, _ = mhr(data["pose"][frames], data["scales"], data["shape"])
    hands = verts[:, hand_ids]
    full_t = data["object_translation"].copy()
    R, s = data["object_rotation"][frames], float(data["object_scale"])

    def local_depth(t):
        local = np.einsum("tpi,tij->tpj", hands - t[:, None], R) / s
        return local, depths(pen, local, mv, mf)

    local, depth = local_depth(full_t[frames])
    before = float(depth.clip(min=0).mean())
    if sigma <= 0:
        push = required_push(pen, local, depth, R, s, mv, mf, iters, search)
        nz = np.linalg.norm(push, axis=1) > 0
        push[nz] += push[nz] / np.linalg.norm(push[nz], axis=1, keepdims=True) * margin
        full_t[frames] += push
    else:
        offset = np.zeros((T, 3))
        for _ in range(rounds):
            local, depth = local_depth(full_t[frames] + offset[frames])
            if not (depth > 0).any():
                break
            push = required_push(pen, local, depth, R, s, mv, mf, iters, search)
            nz = np.linalg.norm(push, axis=1) > 0
            push[nz] += push[nz] / np.linalg.norm(push[nz], axis=1, keepdims=True) * margin
            full = np.zeros((T, 3))
            full[frames] = push
            offset += smooth_envelope(full, sigma)
        full_t += offset
    local, depth = local_depth(full_t[frames])
    data["object_translation"] = full_t
    np.savez(path, **data)
    return before, float(depth.clip(min=0).mean()), int((depth > 0).any(1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-dir", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--sample", type=Path, default=KIT / "data/track_1_sample_submission.parquet")
    ap.add_argument("--margin", type=float, default=0.002, help="extra clearance, metres")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="if >0, apply the push as a temporally smooth offset (Gaussian sigma, frames)")
    ap.add_argument("--search", action="store_true",
                    help="search 26 extra push directions for the shortest one (slow)")
    args = ap.parse_args()
    pen = load_pen()
    mhr = MHRForward(DEFAULT_MHR, device="cuda" if torch.cuda.is_available() else "cpu")
    hand_ids = hand_vertex_ids(mhr.model)
    budget = sample_budget(args.sample)
    scored = scored_frames(args.sample)
    eps = args.episodes if args.episodes is not None else sorted(
        int(p.stem.split("_")[1]) for p in args.episodes_dir.glob("episode_??????.npz"))
    for ep in eps:
        b, a, n = fix_episode(args.episodes_dir / f"episode_{ep:06d}.npz", pen, mhr, hand_ids,
                              budget, args.margin, args.iters, scored.get(ep), args.sigma, search=args.search)
        print(f"episode {ep}: mean hand depth {b * 100:.4f} -> {a * 100:.4f} (mesh units x100), "
              f"frames still penetrating {n}", flush=True)


if __name__ == "__main__":
    main()
