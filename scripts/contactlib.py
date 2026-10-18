"""Hand-object geometry helpers: signed-distance grids of object meshes and hand vertex sets.

The sign test is the scorer's own (winding number of metric_code/track_1/PEN.py, |w| > 0.5 is
inside) and distances are exact point-triangle distances, both evaluated once on a regular grid in
mesh coordinates; queries are trilinear (numpy or differentiable torch). Outside the grid the
value at the clamped point plus the distance to the grid is returned (an upper bound that keeps
the gradient pointing towards the object).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KIT = "/work/doreen071/v2d/kit/v2d_submission_kit"
SAMPLE = os.path.join(KIT, "data/track_1_sample_submission.parquet")
if KIT not in sys.path:
    sys.path.insert(0, KIT)

_PEN = None


def pen_module():
    global _PEN
    if _PEN is None:
        from fix_penetration import load_pen
        _PEN = load_pen()
    return _PEN


def budget(path):
    """The mesh exactly as the packer submits it (welded, simplified, padded)."""
    from fix_penetration import sample_budget
    from v2dlb.mesh_budget import budget_mesh
    from pathlib import Path
    mv, mf = budget_mesh(str(path), *sample_budget(Path(SAMPLE)))
    return mv, mf.astype(np.int64)


def penetration_depth(points, mv, mf):
    """Scorer penetration depth (mesh units) of points [N,3] in mesh coordinates."""
    pen = pen_module()
    out = np.zeros(len(points))
    lo, hi = mv.min(0) - 1e-6, mv.max(0) + 1e-6
    near = np.flatnonzero(((points >= lo) & (points <= hi)).all(1))
    if len(near):
        out[near] = pen._penetration_depth(np.ascontiguousarray(points[near]), mv, mf)
    return out


def _closest_winding(p, A, B, C):
    """p [n,3] torch -> (closest surface point [n,3], winding number [n]) for triangles A, B, C."""
    import torch
    p = p[:, None]
    ab, ac = B - A, C - A

    def dot(x, y):
        return (x * y).sum(-1)

    a, b, c = A[None] - p, B[None] - p, C[None] - p
    la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
    num = dot(a, torch.cross(b, c, dim=-1))
    den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb
    w = torch.atan2(num, den).sum(1) / (2 * np.pi)
    del a, b, c, la, lb, lc, num, den
    ap, bp, cp = p - A[None], p - B[None], p - C[None]
    d1, d2, d3, d4, d5, d6 = dot(ab, ap), dot(ac, ap), dot(ab, bp), dot(ac, bp), dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    denom = va + vb + vc
    sd = torch.where(denom != 0, denom, torch.ones_like(denom))
    v = torch.where(denom != 0, vb / sd, torch.zeros_like(sd))
    ww = torch.where(denom != 0, vc / sd, torch.zeros_like(sd))
    cl = A[None] + ab[None] * v[..., None] + ac[None] * ww[..., None]

    def safe(n, m):
        return torch.nan_to_num(n / m)
    t = safe(d4 - d3, (d4 - d3) + (d5 - d6))
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    cl = torch.where(m[..., None], B[None] + (C - B)[None] * t[..., None], cl)
    t = safe(d2, d2 - d6)
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    cl = torch.where(m[..., None], A[None] + ac[None] * t[..., None], cl)
    m = (d6 >= 0) & (d5 <= d6)
    cl = torch.where(m[..., None], C[None].expand_as(cl), cl)
    t = safe(d1, d1 - d3)
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    cl = torch.where(m[..., None], A[None] + ab[None] * t[..., None], cl)
    m = (d3 >= 0) & (d4 <= d3)
    cl = torch.where(m[..., None], B[None].expand_as(cl), cl)
    m = (d1 <= 0) & (d2 <= 0)
    cl = torch.where(m[..., None], A[None].expand_as(cl), cl)
    k = (p - cl).norm(dim=-1).argmin(1)
    return cl[torch.arange(len(k), device=k.device), k], w


class ExactSDF:
    """Exact signed distance to a mesh (scorer's inside test), differentiable in the query points."""

    def __init__(self, mv, mf, device="cuda", chunk=2_000_000):
        import torch
        area = np.linalg.norm(np.cross(mv[mf[:, 1]] - mv[mf[:, 0]], mv[mf[:, 2]] - mv[mf[:, 0]]), axis=1)
        f = mf[area > 0]
        self.A, self.B, self.C = (torch.tensor(mv[f[:, k]], dtype=torch.float64, device=device) for k in range(3))
        self.per = max(1, chunk // len(f))

    def __call__(self, p):
        import torch
        cls, ws = [], []
        with torch.no_grad():
            q = p.detach().double()
            for s in range(0, len(q), self.per):
                c, w = _closest_winding(q[s:s + self.per], self.A, self.B, self.C)
                cls.append(c)
                ws.append(w)
        if not cls:
            return p.new_zeros(0)
        c = torch.cat(cls).to(p.dtype)
        w = torch.cat(ws)
        d = (p - c).norm(dim=-1)
        return torch.where(w.abs() > 0.5, -d, d)


def signed_distance(points, v0, v1, v2, chunk=2_000_000):
    """Exact signed distance (negative inside, winding |w| > 0.5 as in the scorer) on the GPU.

    Same formulas as metric_code/track_1/PEN.py (_winding_numbers, _point_triangle_distance),
    evaluated in float64 torch, chunked over points.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A, B, C = (torch.tensor(np.asarray(v), dtype=torch.float64, device=dev) for v in (v0, v1, v2))
    per = max(1, chunk // len(A))
    out = np.empty(len(points))
    for s in range(0, len(points), per):
        p = torch.tensor(points[s:s + per], dtype=torch.float64, device=dev)
        c, w = _closest_winding(p, A, B, C)
        d = (p - c).norm(dim=-1)
        out[s:s + per] = torch.where(w.abs() > 0.5, -d, d).cpu().numpy()
    return out


class SDFGrid:
    """Signed distance (mesh units, negative inside) of a triangle mesh on a regular grid."""

    def __init__(self, mv, mf, res=96, pad=0.3):
        lo, hi = mv.min(0), mv.max(0)
        ext = hi - lo
        m = float(ext.max())
        lo = lo - pad * m
        hi = hi + pad * m
        self.cell = (hi - lo).max() / (res - 1)
        self.shape = np.ceil((hi - lo) / self.cell).astype(int) + 1
        self.origin = lo
        axes = [lo[k] + self.cell * np.arange(self.shape[k]) for k in range(3)]
        P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        area = np.linalg.norm(np.cross(mv[mf[:, 1]] - mv[mf[:, 0]], mv[mf[:, 2]] - mv[mf[:, 0]]), axis=1)
        f = mf[area > 0]
        d = signed_distance(P, mv[f[:, 0]], mv[f[:, 1]], mv[f[:, 2]])
        self.grid = d.reshape(self.shape).astype(np.float32)
        self._t = None

    # ---------------------------------------------------------------- numpy
    def query(self, p):
        from scipy.ndimage import map_coordinates
        p = np.asarray(p, float)
        sh = p.shape
        p = p.reshape(-1, 3)
        g = (p - self.origin) / self.cell
        gc = np.clip(g, 0, self.shape - 1)
        val = map_coordinates(self.grid, gc.T, order=1, mode="nearest")
        out = val + np.linalg.norm(g - gc, axis=1) * self.cell
        return out.reshape(sh[:-1])

    # ---------------------------------------------------------------- torch
    def torch_grid(self, device):
        import torch
        if self._t is None or self._t[0].device != torch.device(device):
            self._t = (torch.tensor(self.grid, device=device)[None, None],
                       torch.tensor(self.origin, dtype=torch.float32, device=device),
                       torch.tensor(self.shape - 1, dtype=torch.float32, device=device))
        return self._t

    def query_hybrid(self, p, exact, band):
        """Grid value, replaced by the exact signed distance where the grid is below band."""
        import torch
        sd = self.query_torch(p)
        near = (sd.detach() < band).reshape(-1)
        if near.any():
            flat = p.reshape(-1, 3)
            ex = torch.zeros_like(sd.reshape(-1))
            ex = ex.index_put((near.nonzero()[:, 0],), exact(flat[near]))
            sd = torch.where(near, ex, sd.reshape(-1)).reshape(sd.shape)
        return sd

    def query_torch(self, p):
        """p [..., 3] (torch, mesh units) -> signed distance [...], differentiable in p."""
        import torch
        G, o, n = self.torch_grid(p.device)
        sh = p.shape
        g = (p.reshape(-1, 3) - o) / self.cell
        gc = torch.minimum(torch.maximum(g, torch.zeros_like(n)), n)
        # grid_sample expects (x, y, z) = (W, H, D) order for a [1,1,D,H,W] volume
        q = (gc / n * 2 - 1)[:, [2, 1, 0]].reshape(1, -1, 1, 1, 3)
        val = torch.nn.functional.grid_sample(G, q, align_corners=True, mode="bilinear",
                                              padding_mode="border").reshape(-1)
        out = val + torch.linalg.norm(g - gc, dim=1) * self.cell
        return out.reshape(sh[:-1])


def hand_ids_from(model):
    from fix_penetration import hand_vertex_ids
    return hand_vertex_ids(model)


SCORED_CACHE = "/work/doreen071/v2d/work/scored_frames.json"


def scored():
    """Scored frames per episode (sample parquet), cached as JSON for envs without pyarrow."""
    import json
    if os.path.exists(SCORED_CACHE):
        return {int(k): np.asarray(v) for k, v in json.load(open(SCORED_CACHE)).items()}
    from fix_penetration import scored_frames
    from pathlib import Path
    s = scored_frames(Path(SAMPLE))
    json.dump({str(k): v.tolist() for k, v in s.items()}, open(SCORED_CACHE, "w"))
    return s


def save_sdf(path, mv, mf, sdf):
    np.savez(path, mv=mv, mf=mf, grid=sdf.grid, origin=sdf.origin, cell=sdf.cell)


def load_sdf(path):
    d = np.load(path)
    g = SDFGrid.__new__(SDFGrid)
    g.grid, g.origin, g.cell = d["grid"], d["origin"], float(d["cell"])
    g.shape = np.array(g.grid.shape)
    g._t = None
    return d["mv"], d["mf"].astype(np.int64), g


def main():
    """Precompute budgeted meshes + SDF grids:  python contactlib.py OUT_DIR MESH:EP [MESH:EP ...]"""
    import time
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    scored()
    for item in sys.argv[2:]:
        mesh, ep = item.rsplit(":", 1)
        t0 = time.time()
        mv, mf = budget(mesh)
        sdf = SDFGrid(mv, mf, res=128)
        save_sdf(os.path.join(out, f"episode_{int(ep):06d}_sdf.npz"), mv, mf, sdf)
        print(f"ep {ep}: grid {tuple(sdf.shape)} cell {sdf.cell:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
