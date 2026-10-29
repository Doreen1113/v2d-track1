"""Make hand-object penetration exactly zero with the smallest smooth object offset.

Run after fix_penetration_fingers.py. The contacts that remain are mostly palm (wrist-skinned)
vertices resting slightly inside a surface and finger tips hooked over thin parts (rims, handles),
typically 1-5 mm deep. Here the object is translated by a small, temporally smooth offset D_t so
that every hand vertex skinned to a wrist subtree (a superset of the scorer's 512 hand points) is
outside the submission-budgeted object mesh on every scored frame, by the scorer's own inside test
(winding number |w| > 0.5), with a small clearance margin. Frames that no small translation can
clear (a rim or handle pinched between fingers and thumb or palm) first get their finger and
wrist-flexion parameters relaxed by a short gradient descent on the exact signed distance, pulled
towards the original pose; those parameters do not move any of the scorer's 22 joints.

Per scored frame with penetration, the smallest clearing translation is found by an exact search
(directions x radii, inside test on the GPU) and refined by least-distance programming on the
signed distance linearised at that clear configuration. Frames near contact contribute their
linearised non-penetration constraints at the current offset. All offsets then solve one convex QP
    min  sum_t |D_t|^2 + w_v |D_t - D_{t-1}|^2 + w_a |D_{t-1} - 2 D_t + D_{t+1}|^2
    s.t. every linearised constraint,
which is re-linearised until the exact check passes on every scored frame. The rotation, scale and
mesh are untouched.

    python fix_penetration_micro.py --src EPS --out OUT [--episodes ...]   (OUT: copy of EPS, fixed)
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import nnls
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contactlib as cl  # noqa: E402
from mhr_utils import DEFAULT_MHR, MHRForward  # noqa: E402

MESH_SUFFIXES = (".glb", ".obj", ".ply", ".stl", ".off")
SCORED_JOINTS = (1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42)

# ----------------------------------------------------------------------------- geometry
class Mesh:
    """Budgeted object mesh on the GPU: exact signed distance (float64) and inside test (float32)."""

    def __init__(self, mv, mf, dev):
        area = np.linalg.norm(np.cross(mv[mf[:, 1]] - mv[mf[:, 0]], mv[mf[:, 2]] - mv[mf[:, 0]]), axis=1)
        f = mf[area > 0]
        self.dev = dev
        self.A64, self.B64, self.C64 = (torch.tensor(mv[f[:, k]], dtype=torch.float64, device=dev)
                                        for k in range(3))
        self.A, self.B, self.C = (x.float() for x in (self.A64, self.B64, self.C64))
        self.lo, self.hi = mv.min(0), mv.max(0)
        self.per = max(64, 2_000_000 // len(f))

    @torch.no_grad()
    def sd(self, p):
        """p [N,3] (mesh units) -> signed distance [N] (negative inside), outward unit normal [N,3].
        Same formulas as the scorer, in float32 (the margin dwarfs the rounding)."""
        sd = np.empty(len(p))
        nrm = np.empty((len(p), 3))
        for s in range(0, len(p), self.per):
            q = torch.tensor(p[s:s + self.per], dtype=torch.float32, device=self.dev)
            c, w = cl._closest_winding(q, self.A, self.B, self.C)
            d = q - c
            dist = d.norm(dim=-1)
            sign = 1.0 - 2.0 * (w.abs() > 0.5).float()
            sd[s:s + self.per] = (sign * dist).cpu().numpy()
            nrm[s:s + self.per] = (d / dist.clamp_min(1e-12)[:, None] * sign[:, None]).cpu().numpy()
        return sd, nrm

    @torch.no_grad()
    def sd32(self, q):
        """q torch [N,3] float32 (mesh units) -> signed distance torch [N] (negative inside)."""
        out = torch.empty(len(q), device=self.dev)
        for s in range(0, len(q), self.per):
            c, w = cl._closest_winding(q[s:s + self.per], self.A, self.B, self.C)
            sign = 1.0 - 2.0 * (w.abs() > 0.5).float()
            out[s:s + self.per] = sign * (q[s:s + self.per] - c).norm(dim=-1)
        return out

    @torch.no_grad()
    def inside32(self, q):
        """q torch [N,3] float32 (mesh units) -> bool [N], winding |w| > 0.5."""
        out = torch.empty(len(q), dtype=torch.bool, device=self.dev)
        for s in range(0, len(q), self.per):
            p = q[s:s + self.per, None]
            a, b, c = self.A[None] - p, self.B[None] - p, self.C[None] - p
            la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
            num = (a * torch.cross(b, c, dim=-1)).sum(-1)
            den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
            out[s:s + self.per] = torch.atan2(num, den).sum(1).abs() > np.pi  # |w| > 0.5
        return out


def fibonacci(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def ldp(G, h):
    """min |x| s.t. G x >= h (Lawson-Hanson least-distance programming via NNLS); None if infeasible."""
    m, n = G.shape
    if m == 0:
        return np.zeros(n)
    E = np.vstack([G.T, h[None]])
    f = np.zeros(n + 1)
    f[n] = 1.0
    try:
        u, _ = nnls(E, f, maxiter=50 * max(m, 10))
    except RuntimeError:
        return None
    r = E @ u - f
    if np.linalg.norm(r) < 1e-12 or abs(r[n]) < 1e-14:
        return None
    return -r[:n] / r[n]


def dedupe(G, c, bins):
    """Keep, per normal-direction bin, only the tightest of the constraints G x <= c (|G_k| = 1)."""
    if len(G) <= 8:
        return G, c
    key = np.argmax(G @ bins.T, axis=1)
    order = np.lexsort((c, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    keep = order[first]
    return G[keep], c[keep]


# ----------------------------------------------------------------------------- per-frame exit
def min_exit(mesh, L, R, s, dirs, margin, r_max, band, r_min=0.0, steps=48):
    """Smallest object translation clearing the hand points of one frame.

    L [N,3]: hand points in mesh coordinates at the current offset, R, s: object pose. Returns the
    extra scene-space translation x [3] and the constraints (G [k,3], h [k]) linearised at the
    cleared configuration, in the form G . x' <= h for the extra translation x', or (None, None).
    Along every direction the clearing radius is found by sphere tracing: a point at depth e stays
    inside for any further shift smaller than e (signed distance is 1-Lipschitz), so each step
    advances by the largest depth (plus half the margin) and cannot pass far beyond the first clear
    radius; the clearance margin itself is then imposed by the least-distance refinement.
    """
    sd0, _ = mesh.sd(L)
    sd0 = sd0 * s
    if (sd0 >= margin).all():
        return np.zeros(3), None
    dev = mesh.dev
    dir_loc = torch.tensor(dirs @ R / s, dtype=torch.float32, device=dev)  # local shift per metre
    Lt = torch.tensor(L, dtype=torch.float32, device=dev)
    sd0_t = torch.tensor(sd0, dtype=torch.float32, device=dev)
    r = torch.full((len(dirs),), float(max(r_min, -sd0.min())), device=dev)  # depth: a lower bound
    done = torch.zeros(len(dirs), dtype=torch.bool, device=dev)
    alive = torch.ones(len(dirs), dtype=torch.bool, device=dev)
    for _ in range(steps):
        idx = torch.nonzero(alive & ~done)[:, 0]
        if not len(idx):
            break
        need = torch.nonzero(sd0_t < r[idx].max())[:, 0]
        q = (Lt[need][None] - r[idx][:, None, None] * dir_loc[idx][:, None]).reshape(-1, 3)
        ins = mesh.inside32(q)
        depth = torch.zeros(len(q), device=dev)
        if ins.any():
            depth[ins] = -mesh.sd32(q[ins]) * s  # winding first; distances only for inside points
        short = depth.reshape(len(idx), len(need)).max(1).values if len(need) else torch.zeros(len(idx), device=dev)
        clear = short <= 0
        done[idx[clear]] = True
        r[idx[~clear]] += short[~clear] + 0.5 * margin
        alive &= r <= r_max
        if done.any() and (r[alive & ~done].min() if (alive & ~done).any() else np.inf) >= r[done].min():
            break
    if not done.any():
        return None, None
    rr = torch.where(done, r, torch.full_like(r, np.inf))
    di = int(torch.argmin(rr))
    x = dirs[di] * float(rr[di])
    # refine with least-distance programming on the signed distance linearised at a clear state
    best = None
    for it in range(6):
        sd, nr = mesh.sd(L - (R.T @ x) / s)
        sd = sd * s
        if (sd > 0).all():
            if best is None or np.linalg.norm(x) < np.linalg.norm(best[0]) or not best[2]:
                best = (x.copy(), (sd, nr), bool((sd >= 0.6 * margin).all()))
            if best[2] and it > 0:
                break
        else:
            break
        m = sd < band + np.linalg.norm(x)
        g = nr[m] @ R.T
        y = ldp(-g, margin - sd[m] - g @ x)  # sd_i(x) - g_i.(y - x) >= margin
        if y is None:
            break
        x = y
    if best is None:
        return None, None
    x, (sd, nr), _ = best
    m = sd < band + np.linalg.norm(x)
    g = nr[m] @ R.T
    return x, (g, sd[m] + g @ x - margin)


def lin_constraints(sd, nr, R, band, margin):
    """Constraints g . x' <= h on an extra translation x' for points of a non-penetrating frame."""
    m = sd < band
    if not m.any():
        return None
    g = nr[m] @ R.T
    return g, sd[m] - np.minimum(margin, np.maximum(sd[m], 0.0))


# ----------------------------------------------------------------------------- smooth QP
def time_hessian(T, w_n, w_v, w_a):
    I = sp.eye(T, format="csr")
    D1 = sp.diags([-np.ones(T - 1), np.ones(T - 1)], [0, 1], shape=(T - 1, T))
    D2 = sp.diags([np.ones(T - 2), -2 * np.ones(T - 2), np.ones(T - 2)], [0, 1, 2], shape=(T - 2, T))
    return (w_n * I + w_v * D1.T @ D1 + w_a * D2.T @ D2).tocsc()


def solve_offsets(T, frames, G, c, w_n=1.0, w_v=0.0, w_a=1000.0, sigma=1e-6, alpha=1.6,
                  max_iter=50000, tol=1e-6, x0=None):
    """min 1/2 x'Px s.t. G_k . x[frames_k] <= c_k (ADMM). Units: c in mm. Returns [T,3]."""
    K = len(frames)
    if K == 0:
        return np.zeros((T, 3))
    P = sp.kron(time_hessian(T, w_n, w_v, w_a), sp.eye(3), format="csc")
    rows = np.repeat(np.arange(K), 3)
    cols = (np.asarray(frames)[:, None] * 3 + np.arange(3)[None]).reshape(-1)
    A = sp.csr_matrix((np.asarray(G, float).reshape(-1), (rows, cols)), shape=(K, 3 * T))
    At = A.T.tocsr()
    b = np.asarray(c, float)
    n = 3 * T
    AtA = (At @ A).tocsc()
    rho = 0.1

    def factor(r):
        return splu((P + sigma * sp.eye(n) + r * AtA).tocsc())

    lu = factor(rho)
    x = np.zeros(n) if x0 is None else np.asarray(x0, float).reshape(-1).copy()
    z = np.minimum(A @ x, b)
    y = np.zeros(K)
    for it in range(max_iter):
        xt = lu.solve(sigma * x + At @ (rho * z - y))
        zt = A @ xt
        x = alpha * xt + (1 - alpha) * x
        zr = alpha * zt + (1 - alpha) * z
        znew = np.minimum(zr + y / rho, b)
        y = y + rho * (zr - znew)
        z = znew
        if it % 25 == 0:
            Ax = A @ x
            r_p = np.abs(Ax - z).max()
            r_d = np.abs(P @ x + At @ y).max()
            viol = (Ax - b).max()
            sp_ = max(np.abs(Ax).max(), np.abs(z).max(), 1e-9)
            sd_ = max(np.abs(P @ x).max(), np.abs(At @ y).max(), 1e-9)
            if viol <= tol and r_p <= tol + 1e-6 * sp_ and r_d <= 1e-4 * sd_:
                break
            ratio = np.sqrt((r_p / sp_) / max(r_d / sd_, 1e-30))
            if ratio > 5 or ratio < 0.2:
                rho = float(np.clip(rho * ratio, 1e-6, 1e6))
                lu = factor(rho)
    return x.reshape(T, 3)


# ----------------------------------------------------------------------------- hands
def subtree(parents, root):
    out = set()
    for j in range(len(parents)):
        p = parents[j]
        while p >= 0:
            if p == root:
                out.add(j)
                break
            p = parents[p]
    return out


def finger_params(model):
    """Pose parameters that only drive joints strictly below the wrist twist joints (41 / 77): the
    fingers and the wrist flexion (joints 42 / 78 rotate in place). None moves a scored joint."""
    b = {k: v.cpu().numpy() for k, v in model.named_buffers()}
    P = b["character_torch.parameter_transform.parameter_transform"]
    par = b["character_torch.skeleton.joint_parents"]
    J = len(par)
    A = np.abs(P[:J * 7].reshape(J, 7, -1)).sum(1)
    out = []
    for w in (41, 77):
        sub = subtree(par, w)
        out += [p for p in range(136) if A[:, p].sum() > 0 and set(np.flatnonzero(A[:, p])) <= sub]
    return np.array(sorted(out))


# ----------------------------------------------------------------------------- episode
class Fixer:
    def __init__(self, args):
        self.a = args
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch._C._jit_set_profiling_executor(False)  # no re-specialisation for every batch shape
        self.mhr = MHRForward(DEFAULT_MHR, device=self.dev, dtype=torch.float64)
        for prm in self.mhr.model.parameters():
            prm.requires_grad_(False)
        self.hand_ids = cl.hand_ids_from(self.mhr.model)
        self.fidx = finger_params(self.mhr.model)
        self.scored = cl.scored()
        self.dirs = fibonacci(args.dirs)
        self.bins = fibonacci(64)

    def hands(self, pose, scales, shape):
        verts, joints = self.mhr(pose, scales, shape)
        return verts[:, self.hand_ids], joints[:, list(SCORED_JOINTS)]

    def relax_sqp(self, ks, fr, pose, data, mesh, Rf, tf, s, D):
        """Per frame k in ks, the smallest joint change of the finger / wrist-flexion parameters and the
        object position that puts every hand point outside the mesh with the margin.

        Sequential least-distance programming: the signed distance of every hand point near the surface
        is linearised in the parameters (finite-difference Jacobian of the MHR hand vertices) and in the
        object shift; min |W^1/2 (dtheta, dshift)| subject to those constraints is solved by NNLS, applied
        and re-linearised with the exact signed distance. Parameter change is weighted by hand_reg
        (mm^2 per rad^2), object shift by 1 per mm^2. Returns the new poses [len(ks), 136]."""
        a = self.a
        margin = a.margin
        nf = len(self.fidx)
        eps = 1e-3
        cur = pose[fr[ks]].copy()
        wt, wd = np.sqrt(a.hand_reg), 1.0
        for j, k in enumerate(ks):
            f = fr[k]
            theta = cur[j].copy()
            shift = np.zeros(3)  # extra object shift, mm (scene)
            for it in range(a.hand_reps):
                P = np.repeat(theta[None], nf + 1, 0)
                P[np.arange(1, nf + 1), self.fidx] += eps
                V, _ = self.hands(P, data["scales"], data["shape"])
                Lw = (V[0] - tf[k] - D[f] - shift / 1000.0) @ Rf[k] / s
                box = np.flatnonzero(((Lw >= mesh.lo - 0.01 / s) & (Lw <= mesh.hi + 0.01 / s)).all(1))
                if not len(box):
                    break
                sd, nr = mesh.sd(Lw[box])
                sd = sd * s * 1000.0  # mm
                if (sd >= 0.6 * margin).all():
                    break
                m = sd < a.band
                idx, sd, g = box[m], sd[m], nr[m] @ Rf[k].T  # scene outward normals
                Jp = (V[1:, idx] - V[0][None, idx]) / eps * 1000.0  # [nf, n, 3] mm / rad
                a_th = np.einsum("fni,ni->nf", Jp, g)  # d sd / d theta
                # sd_i + a_th . dth - g_i . dsh >= target_i ;  z = (sqrt(wt) dth, sqrt(wd) dsh)
                target = np.where(sd < 0, margin, np.minimum(margin, sd))
                G = np.hstack([a_th / wt, -g / wd])
                z = ldp(G, target - sd)
                if z is None:
                    break
                dth, dsh = z[:nf] / wt, z[nf:] / wd
                trust = min(1.0, 0.05 / max(np.abs(dth).max(), 1e-12), 3.0 / max(np.linalg.norm(dsh), 1e-12))
                theta[self.fidx] += trust * dth  # trust region: 0.05 rad, 3 mm per step
                shift += trust * dsh
            cur[j] = theta
        return cur

    def fix(self, src, dst):
        a = self.a
        t_start = time.time()
        ep = int(src.stem.split("_")[1])
        data = dict(np.load(src))
        mesh_path = next(p for s in MESH_SUFFIXES if (p := src.with_name(src.stem + "_object" + s)).exists())
        mv, mf = cl.budget(mesh_path)
        mesh = Mesh(mv, mf, self.dev)
        fr = np.asarray(self.scored[ep])
        T = len(data["pose"])
        R = np.asarray(data["object_rotation"], float)
        u, _, vt = np.linalg.svd(R)
        R = u @ vt  # the scorer's projection onto rotations
        t0 = np.asarray(data["object_translation"], float)
        s = float(np.asarray(data["object_scale"]).reshape(()))
        pose = np.asarray(data["pose"], float).copy()
        pose0 = pose.copy()
        H, J0 = self.hands(pose[fr], data["scales"], data["shape"])
        H0 = H.copy()
        Rf, tf = R[fr], t0[fr]
        margin, band = a.margin / 1000.0, a.band / 1000.0
        reach = a.r_max / 1000.0 + band + 0.01  # points farther than this never matter
        lo, hi = mesh.lo - reach / s, mesh.hi + reach / s

        def local(k, D):
            return (H[k] - tf[k] - D) @ Rf[k] / s

        def active(ks):
            """per frame: hand points inside the padded box and within `reach` of the surface"""
            cand = []
            for k in ks:
                L = local(k, 0.0)
                cand.append(np.flatnonzero(((L >= lo) & (L <= hi)).all(1)))
            pts = np.concatenate([local(k, 0.0)[i] for k, i in zip(ks, cand)]) if len(ks) else np.zeros((0, 3))
            sd, _ = mesh.sd(pts)
            out, o = {}, 0
            for k, i in zip(ks, cand):
                out[k] = i[sd[o:o + len(i)] * s < reach]
                o += len(i)
            return out

        act = active(range(len(fr)))

        def evaluate(D, ks=None):
            ks = [k for k in (act if ks is None else ks) if len(act[k])]
            pts = np.concatenate([local(k, D[fr[k]])[act[k]] for k in ks]) if ks else np.zeros((0, 3))
            sd, nr = mesh.sd(pts)
            out, o = {}, 0
            for k in ks:
                n = len(act[k])
                out[k] = (sd[o:o + n] * s, nr[o:o + n])
                o += n
            return out

        D = np.zeros((T, 3))
        ev = evaluate(D)
        pen0 = sum(float(np.maximum(-v[0], 0).sum()) for v in ev.values()) / (len(fr) * H.shape[1])
        nbad0 = sum(int((v[0] < 0).any()) for v in ev.values())
        relaxed, far, stuck = set(), set(), {}
        timing = {"exit": 0.0, "relax": 0.0, "qp": 0.0, "eval": 0.0}
        cons = {}  # scored index -> list of (G, h, D_at):  G . (D_t - D_at) <= h
        hist = []
        for it in range(a.rounds):
            bad = [k for k, (sd, _) in ev.items() if (sd < a.accept / 1000.0).any()]
            hist.append(len(bad))
            if not bad:
                break
            for k, (sd, nr) in ev.items():
                if k in bad:
                    continue
                c_ = lin_constraints(sd, nr, Rf[k], band, margin)
                if c_ is not None:
                    cons.setdefault(k, []).append((*dedupe(*c_, self.bins), D[fr[k]].copy()))
            found, pinch = {}, []
            tt = time.time()
            for k in bad:
                x, c_ = min_exit(mesh, local(k, D[fr[k]])[act[k]], Rf[k], s, self.dirs, margin,
                                 a.r_hand / 1000.0, band)
                if x is None:
                    pinch.append(k)
                else:
                    found[k] = c_
            timing["exit"] += time.time() - tt
            def exit_all(k, Hk, r_done=0.0):
                """smallest clearing shift for hand points Hk of scored frame k: (x, constraints, idx)"""
                L0 = (Hk - tf[k]) @ Rf[k] / s
                pad = (a.r_last / 1000.0 + band) / s
                idx = np.flatnonzero(((L0 >= mesh.lo - pad) & (L0 <= mesh.hi + pad)).all(1))
                L = (Hk[idx] - tf[k] - D[fr[k]]) @ Rf[k] / s
                for r_lim in (a.r_max, a.r_last):
                    x, c_ = min_exit(mesh, L, Rf[k], s, self.dirs, margin, r_lim / 1000.0, band, r_done)
                    if x is not None:
                        return x, c_, idx
                    r_done = r_lim / 1000.0
                return None, None, idx

            tt = time.time()
            pinch = [k for k in pinch if stuck.get(k, 0) < 2]  # nothing within r_last twice: give up
            todo = [k for k in pinch if k not in relaxed]
            cand = {}
            if todo and a.hand_reps > 0:
                new = self.relax_sqp(np.array(todo), fr, pose, data, mesh, Rf, tf, s, D)
                Hnew, _ = self.hands(new, data["scales"], data["shape"])
                cand = {k: (new[j], Hnew[j]) for j, k in enumerate(todo)}
                relaxed.update(todo)
            timing["relax"] += time.time() - tt
            tt = time.time()
            for k in pinch:
                x, c_, idx = exit_all(k, H[k], a.r_hand / 1000.0)
                if k in cand:
                    x2, c2, idx2 = exit_all(k, cand[k][1])
                    r1 = np.inf if x is None else np.linalg.norm(x)
                    r2 = np.inf if x2 is None else np.linalg.norm(x2)
                    if r2 < r1 - a.hand_gain / 1000.0:  # the finger change must buy a clearly smaller shift
                        pose[fr[k]] = cand[k][0]
                        H[k] = cand[k][1]
                        cons[k] = []  # linearisations of the old hand pose
                        x, c_, idx = x2, c2, idx2
                        act[k] = np.union1d(act[k], idx)
                if x is None:
                    stuck[k] = stuck.get(k, 0) + 1
                    continue
                found[k] = c_
                if np.linalg.norm(x) > a.r_hand / 1000.0:
                    act[k] = np.union1d(act[k], idx)
                if np.linalg.norm(x) > a.r_max / 1000.0:
                    far.add(int(fr[k]))
            timing["exit"] += time.time() - tt
            for k, c_ in found.items():
                if c_ is not None:
                    cons.setdefault(k, []).append((*dedupe(*c_, self.bins), D[fr[k]].copy()))
            frames, G, h = [], [], []
            for k, lst in cons.items():
                for (g, c_, Dat) in lst[-a.keep:]:
                    frames.append(np.full(len(g), fr[k]))
                    G.append(g)
                    h.append((c_ + g @ Dat) * 1000.0)  # G . D <= h, mm
            tt = time.time()
            if frames:
                D = solve_offsets(T, np.concatenate(frames), np.concatenate(G), np.concatenate(h),
                                  w_n=1.0, w_v=a.w_v, w_a=a.w_a, x0=D * 1000.0) / 1000.0
            timing["qp"] += time.time() - tt
            tt = time.time()
            ev = evaluate(D)
            timing["eval"] += time.time() - tt
        bad = [k for k, (sd, _) in ev.items() if (sd <= 0).any()]
        pen1 = sum(float(np.maximum(-v[0], 0).sum()) for v in ev.values()) / (len(fr) * H.shape[1])
        data["object_translation"] = t0 + D
        data["pose"] = pose
        np.savez(dst, **data)
        # report
        disp = np.linalg.norm(D, axis=1) * 100
        acc, acc_b, acc_a = [], [], []
        for part in np.split(fr, np.flatnonzero(np.diff(fr) != 1) + 1):
            if len(part) >= 3:
                acc.append(np.linalg.norm(D[part][:-2] - 2 * D[part][1:-1] + D[part][2:], axis=1) * 100)
                x0_, x1_ = t0[part], t0[part] + D[part]
                acc_b.append(np.linalg.norm(x0_[:-2] - 2 * x0_[1:-1] + x0_[2:], axis=1) * 100)
                acc_a.append(np.linalg.norm(x1_[:-2] - 2 * x1_[1:-1] + x1_[2:], axis=1) * 100)
        acc, acc_b, acc_a = (np.concatenate(v) for v in (acc, acc_b, acc_a))
        Hn, J1 = self.hands(pose[fr], data["scales"], data["shape"])
        dpar = np.abs(pose - pose0).max(1)
        return {
            "episode": ep, "scored": int(len(fr)), "pen_frames_before": nbad0, "pen_frames_after": len(bad),
            "pen_cm_before": pen0 * 100, "pen_cm_after": pen1 * 100, "rounds": hist,
            "disp_mean_cm": float(disp.mean()), "disp_scored_mean_cm": float(disp[fr].mean()),
            "disp_max_cm": float(disp.max()), "moved_frames": int((disp > 1e-3).sum()),
            "acc_delta_mean_cm": float(acc.mean()), "acc_delta_max_cm": float(acc.max()),
            "obj_acc_mean_before_cm": float(acc_b.mean()), "obj_acc_mean_after_cm": float(acc_a.mean()),
            "obj_acc_max_before_cm": float(acc_b.max()), "obj_acc_max_after_cm": float(acc_a.max()),
            "hand_frames": int((dpar > 0).sum()), "hand_param_max_rad": float(dpar.max()),
            "hand_vert_disp_max_cm": float(np.linalg.norm(Hn - H0, axis=-1).max() * 100),
            "joint_disp_max_cm": float(np.linalg.norm(J1 - J0, axis=-1).max() * 100),
            "far_frames": sorted(far), "unresolved_frames": [int(fr[k]) for k in bad],
            "timing": {k: round(v, 1) for k, v in timing.items()},
            "seconds": round(time.time() - t_start, 1),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="episodes dir (fingers already fixed)")
    ap.add_argument("--out", type=Path, required=True, help="output episodes dir")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--margin", type=float, default=0.5, help="clearance target, mm")
    ap.add_argument("--accept", type=float, default=0.1, help="accepted clearance in the exact check, mm")
    ap.add_argument("--band", type=float, default=4.0, help="points within this of the surface give constraints, mm")
    ap.add_argument("--r-max", type=float, default=30.0, help="largest per-frame translation searched, mm")
    ap.add_argument("--r-last", type=float, default=80.0,
                    help="last resort for frames neither a small shift nor the fingers can clear, mm")
    ap.add_argument("--r-hand", type=float, default=8.0,
                    help="frames needing more translation than this get their fingers opened first, mm")
    ap.add_argument("--hand-reps", type=int, default=8, help="hand SQP iterations (0: never touch the hand)")
    ap.add_argument("--hand-reg", type=float, default=100.0, help="cost of finger parameter change, mm^2 per rad^2")
    ap.add_argument("--hand-gain", type=float, default=2.0,
                    help="finger changes are kept only if they reduce the needed object shift by this, mm")
    ap.add_argument("--w-a", type=float, default=1000.0, help="weight of offset acceleration (frames^-4)")
    ap.add_argument("--w-v", type=float, default=10.0, help="weight of offset velocity")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--keep", type=int, default=4, help="linearisations kept per frame")
    ap.add_argument("--dirs", type=int, default=96)
    ap.add_argument("--report", type=Path, default=None, help="JSON report (default OUT/../<name>_report.json)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    eps = a.episodes if a.episodes is not None else sorted(
        int(p.stem.split("_")[1]) for p in a.src.glob("episode_??????.npz"))
    fx = Fixer(a)
    rows = []
    for ep in eps:
        src = a.src / f"episode_{ep:06d}.npz"
        for m in a.src.glob(f"episode_{ep:06d}_object.*"):
            shutil.copy2(m, a.out / m.name)
        row = fx.fix(src, a.out / src.name)
        rows.append(row)
        print(json.dumps(row), flush=True)
    rep = a.report or a.out.parent / f"{a.out.name}_micro_report.json"
    json.dump(rows, open(rep, "w"), indent=1)


if __name__ == "__main__":
    main()
