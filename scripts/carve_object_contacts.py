"""Make hand-object penetration exactly zero by denting the submitted object mesh at the hand contacts.

The packer reduces every object mesh to the submission budget (v2dlb/mesh_budget.py), so this works on that reduced
mesh: what is checked here is exactly what is scored. The hand points (every vertex skinned to a wrist subtree, a
superset of the scorer's 512 hand points) of all scored frames are taken into the object frame. Every point inside
the mesh (the scorer's own winding-number test and depth) pushes the vertices of its closest triangle inwards along
their normals by its depth plus a margin; points outside but closer than --accept get the same push up to the
margin. The push is spread over the surface with a Gaussian of width --sigma around each contact (the largest
weighted push wins per vertex, only vertices facing the same way as the touched triangle, along the vertex normals of the original mesh, at most --step per round), so the dents
stay smooth and no triangle folds over. This repeats until no point is inside and every point is at least --accept
from the surface. Decimated meshes that are open or have overlapping shells cannot always be dented that way; their
scorer-inside region (|winding number| > 0.5) is then rebuilt as one closed surface (marching cubes, simplified to the
budget) and carved instead. Needs scikit-image (PYTHONPATH=/work/doreen071/v2d/envs/mc_lib).

The body and the object poses are untouched, so CD-H, ACC-H and ACC-O do not change. The mesh only moves where the
hands touch it, by the few mm of the deepest contacts (reported per episode). Episodes without contact keep their
original mesh file; an episode that needs more than --max-move is left unchanged and reported.

    python carve_object_contacts.py --src EPS --out OUT [--episodes ...]   (OUT: copy of EPS, carved meshes as .ply)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_pen_exact as cpe  # noqa: E402


def closest_triangles(p, a, b, c, chunk=4096):
    """Closest point on any triangle (a, b, c: [F, 3]) for each query point p [N, 3]: (distance, triangle, point)."""
    dist, idx, pts = [], [], []
    for s in range(0, len(p), chunk):
        q = p[s:s + chunk, None]
        ab, ac, ap = b - a, c - a, q - a
        d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
        bp = q - b
        d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
        cp = q - c
        d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        eps = 1e-30
        # interior point by default, then the Voronoi regions of the edges and vertices (Ericson 5.1.5)
        den = (va + vb + vc).clamp_min(eps)
        v, w = vb / den, vc / den
        x = a + v[..., None] * ab + w[..., None] * ac
        m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        t = ((d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(eps))[..., None]
        x = torch.where(m[..., None], b + t * (c - b), x)
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = (d2 / (d2 - d6).clamp_min(eps))[..., None]
        x = torch.where(m[..., None], a + t * ac, x)
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = (d1 / (d1 - d3).clamp_min(eps))[..., None]
        x = torch.where(m[..., None], a + t * ab, x)
        x = torch.where(((d6 >= 0) & (d5 <= d6))[..., None], c.expand_as(x), x)
        x = torch.where(((d3 >= 0) & (d4 <= d3))[..., None], b.expand_as(x), x)
        x = torch.where(((d1 <= 0) & (d2 <= 0))[..., None], a.expand_as(x), x)
        d = (q - x).norm(dim=-1)
        dm, im = d.min(1)
        dist.append(dm)
        idx.append(im)
        pts.append(x[torch.arange(len(im), device=x.device), im])
    return torch.cat(dist), torch.cat(idx), torch.cat(pts)


def vertex_normals(v, f):
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], n)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-12)
    vol = np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum()
    return vn if vol >= 0 else -vn


def hand_points_local(path):
    pen, body, hand_ids = cpe._STATE["pen"], cpe._STATE["body"], cpe._STATE["hand_ids"]
    ep = int(path.stem.split("_")[1])
    data = dict(np.load(path))
    frames = cpe._STATE["scored"][ep]
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
    return np.einsum("tpi,tij->tpj", hands - t[:, None], R).reshape(-1, 3) / s, s


def strip_padding(v, f):
    real = ~(f == 0).all(1)
    f = f[real]
    n = int(f.max()) + 1
    assert np.allclose(v[n:], v[0]), "unexpected vertices after the referenced ones"
    return v[:n].copy(), f


def winding_numbers(v, f, pts, dev, chunk=8192):
    """Generalised winding number of a triangle soup at many points (float32, GPU), for remeshing only."""
    a, b, c = (torch.as_tensor(v[f[:, k]], device=dev, dtype=torch.float32) for k in range(3))
    out = []
    for s in range(0, len(pts), chunk):
        q = torch.as_tensor(pts[s:s + chunk], device=dev, dtype=torch.float32)[:, None]
        A, B, C = a - q, b - q, c - q
        la, lb, lc = A.norm(dim=-1), B.norm(dim=-1), C.norm(dim=-1)
        det = (A * torch.cross(B, C, dim=-1)).sum(-1)
        den = la * lb * lc + (A * B).sum(-1) * lc + (B * C).sum(-1) * la + (C * A).sum(-1) * lb
        out.append((2 * torch.atan2(det, den)).sum(1) / (4 * np.pi))
    return torch.cat(out).cpu().numpy()


def remesh(v, f, res, budget, dev):
    """Closed, consistently oriented surface of the scorer's inside region (|winding| > 0.5) within the budget."""
    import fast_simplification
    import trimesh
    from skimage.measure import marching_cubes
    lo, hi = v.min(0), v.max(0)
    h = (hi - lo).max() / res
    lo, hi = lo - 2 * h, hi + 2 * h
    n = np.ceil((hi - lo) / h).astype(int) + 1
    grid = np.stack(np.meshgrid(*[lo[i] + h * np.arange(n[i]) for i in range(3)], indexing="ij"), -1)
    w = np.abs(winding_numbers(v, f, grid.reshape(-1, 3), dev)).reshape(n)
    mv, mf, _, _ = marching_cubes(w, 0.5, spacing=(h, h, h))
    mv, mf = mv + lo, mf.astype(np.int64)
    faces, verts = budget
    target = min(faces, 2 * verts - 8) - 64
    if len(mf) > target:
        mv, mf = fast_simplification.simplify(mv, mf, target_count=target)
    m = trimesh.Trimesh(mv, mf, process=True)
    if m.volume < 0:
        m.invert()
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), bool(m.is_watertight)


def carve_mesh(v, f, P, s, a, dev):
    """Dent (v, f) until no point of P is inside and all are at least --accept away; returns (v, stats, ok)."""
    pen = cpe._STATE["pen"]
    v0 = v.copy()
    margin, accept, reach = a.margin / 1000 / s, a.accept / 1000 / s, a.reach / 1000 / s
    sigma, step = a.sigma / 1000 / s, a.step / 1000 / s
    normals = torch.as_tensor(vertex_normals(v, f), device=dev)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fn = torch.as_tensor(fn / np.linalg.norm(fn, axis=1, keepdims=True).clip(1e-12), device=dev)
    # the surface only moves inwards, so points farther than `reach` outside can never become inside or near
    Pt = torch.as_tensor(P, device=dev)
    ft = torch.as_tensor(f, device=dev)
    lo, hi = v.min(0) - reach, v.max(0) + reach
    cand = np.flatnonzero(((P >= lo) & (P <= hi)).all(1))
    st = {"candidates": int(len(cand))}
    if not len(cand):
        st.update(inside_before=0, max_depth_before_mm=0.0, near_left=0, iterations=0, inside_after=0, unresolved=0,
                  moved_vertices=0, vertices=int(len(v)), max_move_mm=0.0, mean_move_moved_mm=0.0)
        return v, st, True
    best = None  # pushing the last near-surface points can fold thin parts; keep the best state with nothing inside
    for it in range(a.iters + 1):
        depth = pen._penetration_depth(P[cand], v, f)
        vt = torch.as_tensor(v, device=dev)
        dist, tri, near_pt = closest_triangles(Pt[cand], vt[ft[:, 0]], vt[ft[:, 1]], vt[ft[:, 2]])
        dist = dist.cpu().numpy()
        inside = depth > 0
        need = np.where(inside, depth + margin, np.where(dist < accept, margin - dist, 0.0))
        if it == 0:
            st.update(inside_before=int(inside.sum()), max_depth_before_mm=float(depth.max(initial=0) * s * 1000))
            keep = inside | (dist < reach)
            kt = torch.as_tensor(keep, device=dev)
            cand, need, near_pt, tri = cand[keep], need[keep], near_pt[kt], tri[kt]
            depth, inside = depth[keep], inside[keep]
        hit = need > 0
        n_near = int((~inside & hit).sum())
        if not inside.any() and (best is None or n_near < best[1]):
            best = (v.copy(), n_near, it)
        if not hit.any() or it == a.iters:
            break
        hit_t = torch.as_tensor(hit, device=dev)
        x, nd, xn = near_pt[hit_t], torch.as_tensor(need[hit], device=dev), fn[tri[hit_t]]
        move = torch.zeros(len(v), device=dev)
        for c in range(0, len(x), 8192):
            k = torch.exp(-torch.cdist(vt, x[c:c + 8192]).square() / (2 * sigma ** 2))
            # only the side of the surface that was touched: the far side of a thin part faces the other way
            k = k * (normals @ xn[c:c + 8192].T > 0.3)
            move = torch.maximum(move, (k * nd[c:c + 8192]).max(1).values)
        v = (vt - normals * move.clamp_max(step)[:, None]).cpu().numpy()
    near_left = 0
    if hit.any() and best is not None:
        v, near_left, it = best
        depth = np.zeros(1)
        hit = np.zeros(1, bool)
    disp = np.linalg.norm(v - v0, axis=1) * s * 1000
    moved = disp > 1e-6
    ok = not hit.any() and disp.max(initial=0) <= a.max_move
    st["near_left"] = near_left
    st.update(iterations=it, inside_after=int((depth > 0).sum()), unresolved=int(hit.sum()),
              moved_vertices=int(moved.sum()), vertices=int(len(v)), max_move_mm=float(disp.max(initial=0)),
              mean_move_moved_mm=float(disp[moved].mean()) if moved.any() else 0.0)
    return v, st, ok


def csg_carve(v0, f0, P, s, a, dev):
    """Object minus a ball of radius r around every hand point, rebuilt as one surface within the budget.

    The field min(|winding| - 0.5, (distance to the nearest hand point - r) / r) is positive exactly inside the
    scorer's object region and outside every ball; its zero level is extracted with marching cubes on a grid of
    --voxel mm, simplified to the budget and checked with the scorer's depth. r grows by 1 mm until nothing is inside
    and every point is at least --accept away (simplification can cut back into the dents). Returns (v, f, stats, ok)."""
    import fast_simplification
    import trimesh
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes
    pen = cpe._STATE["pen"]
    accept = a.accept / 1000 / s
    pad = (a.csg_rmax + 3 * a.voxel) / 1000 / s
    lo, hi = v0.min(0) - pad, v0.max(0) + pad
    h = max(a.voxel / 1000 / s, (hi - lo).max() / a.csg_res)
    n = np.ceil((hi - lo) / h).astype(int) + 1
    axes = [lo[i] + h * np.arange(n[i]) for i in range(3)]
    yz = np.stack(np.meshgrid(axes[1], axes[2], indexing="ij"), -1).reshape(-1, 3 - 1)
    w = np.empty(n, np.float32)
    for i, x in enumerate(axes[0]):
        pts = np.concatenate([np.full((len(yz), 1), x), yz], 1)
        w[i] = np.abs(winding_numbers(v0, f0, pts, dev, chunk=16384)).reshape(n[1], n[2])
    obj = w - 0.5
    Pin = P[((P >= lo) & (P <= hi)).all(1)]
    # distance to the nearest hand point: voxel distance transform, exact (KD tree) wherever it can matter
    idx = np.clip(np.round((Pin - lo) / h).astype(int), 0, n - 1)
    seed = np.ones(n, bool)
    seed[tuple(idx.T)] = False
    d = distance_transform_edt(seed).astype(np.float32) * h
    near = np.flatnonzero((d < (a.csg_rmax / 1000 / s + 2 * h)).ravel() & (obj.ravel() > -0.25))
    if len(near):
        g = np.stack(np.unravel_index(near, n), 1) * h + lo
        d.ravel()[near] = cKDTree(Pin).query(g, workers=a.threads)[0]
    budget_f, budget_v = cpe._STATE["budget"]
    target = min(budget_f, 2 * budget_v - 8) - 64
    st = {"voxel_mm": float(h * s * 1000), "grid": [int(x) for x in n], "hand_points": int(len(Pin))}
    v, f, ok = v0, f0, False
    r = a.csg_r
    rb = np.zeros(len(Pin))  # per-point radius for points that simplification keeps cutting back into, mm (0: none)
    tree = cKDTree(Pin)
    for _ in range(a.csg_tries):
        rr = r / 1000 / s
        field = np.minimum(obj, (d - rr) / rr)
        bad = np.flatnonzero(rb > 0)
        if len(bad):
            bt = cKDTree(Pin[bad])
            rbl = rb[bad] / 1000 / s
            lo_b, hi_b = Pin[bad].min(0) - rbl.max() - 2 * h, Pin[bad].max(0) + rbl.max() + 2 * h
            i0, i1 = np.floor((lo_b - lo) / h).astype(int).clip(0, n - 1), np.ceil((hi_b - lo) / h).astype(int).clip(0, n - 1) + 1
            sub = np.stack(np.meshgrid(*[np.arange(i0[k], i1[k]) for k in range(3)], indexing="ij"), -1).reshape(-1, 3)
            g = sub * h + lo
            dd, ii = bt.query(g, k=min(16, len(bad)), distance_upper_bound=rbl.max() + 2 * h, workers=a.threads)
            dd, ii = dd.reshape(len(g), -1), ii.reshape(len(g), -1)
            valid = ii < len(bad)
            val = np.where(valid, dd - rbl[np.minimum(ii, len(bad) - 1)], np.inf).min(1)
            m = np.isfinite(val)
            sub, val = sub[m], val[m]
            cur = field[tuple(sub.T)]
            field[tuple(sub.T)] = np.minimum(cur, val / rr)
        mv, mf, _, _ = marching_cubes(field, 0.0, spacing=(h, h, h))
        mv, mf = mv + lo, mf.astype(np.int64)
        if len(mf) > target:
            mv, mf = fast_simplification.simplify(mv, mf, target_count=target)
        m = trimesh.Trimesh(mv, mf, process=True)
        if m.volume < 0:
            m.invert()
        v, f = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
        depth = pen._penetration_depth(Pin, v, f)
        vt, ft = torch.as_tensor(v, device=dev), torch.as_tensor(f, device=dev)
        dist = closest_triangles(torch.as_tensor(Pin, device=dev), vt[ft[:, 0]], vt[ft[:, 1]], vt[ft[:, 2]])[0]
        dist = dist.cpu().numpy()
        wrong = (depth > 0) | (dist < accept)
        n_in, n_near = int((depth > 0).sum()), int(((depth <= 0) & (dist < accept)).sum())
        st.setdefault("tries", []).append({"r_mm": r, "boosted": int(len(bad)), "max_boost_mm": float(rb.max()),
                                           "faces": int(len(f)), "inside": n_in, "near": n_near,
                                           "min_clear_mm": float(dist[depth <= 0].min(initial=1) * s * 1000)})
        if not wrong.any():
            ok = True
            break
        if n_in + n_near <= a.csg_polish:
            vp, pst, pok = carve_mesh(v, f, Pin, s, a, dev)
            st["tries"][-1].update(polish_ok=bool(pok), polish_max_move_mm=pst["max_move_mm"])
            if pok:
                v, ok = vp, True
                break
        if r < a.csg_rmax and n_in + n_near > a.csg_boost:
            r += 1.0
            continue
        # grow the radius only around the points that are still wrong, and their neighbours within 2 mm
        nb = np.unique(np.concatenate(tree.query_ball_point(Pin[wrong], 2.0 / 1000 / s, workers=a.threads)))
        rb[nb] = np.minimum(np.maximum(rb[nb], r) + a.csg_bump, a.csg_rboost_max)
    st.update(r_mm=r if ok else None, closed=bool(trimesh.Trimesh(v, f, process=False).is_watertight),
              faces=int(len(f)), vertices=int(len(v)))
    return v, f, st, ok


def surface_change_mm(v0, f0, v1, f1, s, n=20000):
    """Symmetric Chamfer (mean of both directions) between two surfaces, mm at object scale."""
    import trimesh
    from scipy.spatial import cKDTree
    p0 = trimesh.sample.sample_surface(trimesh.Trimesh(v0, f0, process=False), n, seed=0)[0]
    p1 = trimesh.sample.sample_surface(trimesh.Trimesh(v1, f1, process=False), n, seed=0)[0]
    d = cKDTree(p1).query(p0)[0].mean() + cKDTree(p0).query(p1)[0].mean()
    return float(d / 2 * s * 1000)


def carve(path, out_dir, a, dev):
    import trimesh
    from v2dlb.mesh_budget import budget_mesh
    mesh = next(p for s in cpe.MESH_SUFFIXES if (p := path.with_name(path.stem + "_object" + s)).exists())
    P, s = hand_points_local(path)
    v0, f0 = strip_padding(*budget_mesh(str(mesh), *cpe._STATE["budget"]))
    f0 = f0.astype(np.int64)
    row = {"episode": int(path.stem.split("_")[1]), "mode": "direct"}
    v, st, ok = carve_mesh(v0, f0, P, s, a, dev)
    row.update(st)
    f = f0
    if not ok and a.csg and st["inside_before"] + st["unresolved"] > 0:
        v, f, cst, ok = csg_carve(v0, f0, P, s, a, dev)
        row.update({"mode": "csg"}, **{"csg_" + k: x for k, x in cst.items()})
    if not ok and a.remesh and st["inside_before"] + st["unresolved"] > 0:
        vr, f, closed = remesh(v0, f0, a.remesh, cpe._STATE["budget"], dev)
        v, st, ok = carve_mesh(vr, f, P, s, a, dev)
        row.update({"mode": "remesh", "remesh_closed": closed, "remesh_change_mm": surface_change_mm(v0, f0, vr, f, s)},
                   **{"remesh_" + k: x for k, x in st.items()})
    changed = ok and (row["mode"] != "direct" or row["moved_vertices"] > 0)
    row.update(carved=bool(changed), failed=not ok,
               surface_change_mm=surface_change_mm(v0, f0, v, f, s) if changed else 0.0)
    for m in path.parent.glob(path.stem + "_object.*"):
        if not changed:
            shutil.copy2(m, out_dir / m.name)
    if changed:
        trimesh.Trimesh(v, f, process=False).export(out_dir / (path.stem + "_object.ply"))
    shutil.copy2(path, out_dir / path.name)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--margin", type=float, default=0.5, help="clearance target, mm")
    ap.add_argument("--accept", type=float, default=0.2, help="smallest accepted clearance, mm")
    ap.add_argument("--reach", type=float, default=20.0, help="points within this of the mesh are tracked, mm")
    ap.add_argument("--sigma", type=float, default=12.0, help="width of the dent around a contact, mm")
    ap.add_argument("--step", type=float, default=3.0, help="largest vertex move per round, mm")
    ap.add_argument("--max-move", type=float, default=25.0, help="largest vertex move accepted, mm")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--remesh", type=int, default=160,
                    help="if the budgeted mesh cannot be carved (open, overlapping shells), rebuild its inside region "
                         "with marching cubes at this grid size and carve that (0: never)")
    ap.add_argument("--csg", type=int, default=1, help="if the direct dent fails, subtract balls around the hand "
                    "points from the object's inside region and rebuild it (0: off)")
    ap.add_argument("--voxel", type=float, default=1.0, help="CSG grid spacing, mm")
    ap.add_argument("--csg-res", type=int, default=640, help="largest CSG grid size along the longest side")
    ap.add_argument("--csg-r", type=float, default=1.0, help="first ball radius around each hand point, mm")
    ap.add_argument("--csg-polish", type=int, default=60,
                    help="dent the CSG result directly when at most this many points are inside or near")
    ap.add_argument("--csg-tries", type=int, default=12)
    ap.add_argument("--csg-boost", type=int, default=400,
                    help="once at most this many points are wrong, grow the radius only around them")
    ap.add_argument("--csg-bump", type=float, default=2.0, help="radius growth around wrong points per try, mm")
    ap.add_argument("--csg-rboost-max", type=float, default=15.0, help="largest radius around wrong points, mm")
    ap.add_argument("--csg-rmax", type=float, default=5.0, help="largest ball radius tried, mm")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cpe._init(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float64)
    eps = a.episodes if a.episodes is not None else sorted(
        int(p.stem.split("_")[1]) for p in a.src.glob("episode_??????.npz"))
    rows = []
    for ep in eps:
        row = carve(a.src / f"episode_{ep:06d}.npz", a.out, a, dev)
        rows.append(row)
        print(json.dumps(row), flush=True)
    rep = a.report or a.out.parent / f"{a.out.name}_carve_report.json"
    json.dump(rows, open(rep, "w"), indent=1)


if __name__ == "__main__":
    main()
