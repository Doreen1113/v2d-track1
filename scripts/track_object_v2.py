"""Object trajectory v2: whole-trajectory render-and-compare with static segments.

Starting from an initial per-frame trajectory (the v1 track, mapped onto the shared class mesh
by select_class_mesh.py), all frames are optimised jointly:

  sum_f  data_f(pose_f)  +  lambda_t * |accel(t)|^2  +  lambda_r * |accel(R)|^2

data_f = soft-silhouette IoU against the object mask (actor pixels ignored) + a distance-to-mask
term (gives gradients for thin objects such as the hoop) + a robust depth term against the
scale-stabilised MoGe depth. Frames where the object rests (static segments detected from the
masks and the image appearance, objlib.static_segments) share ONE pose, so the object is
exactly still there. The episode scale is optimised as well (anchored by the depth term),
except for thin objects whose depth is unreliable. Frames that stay poorly fitted are
re-initialised (flipped / neighbouring poses, single-frame fits) and the joint optimisation is
repeated.

Inputs : WORK/objects_v2/init/episode_XXXXXX.npz  (R [T,3,3], t [T,3], s, mesh)
Outputs: WORK/objects_v2/track/episode_XXXXXX.npz (R, t, s, valid, iou, seg_id, ...)
         WORK/objects_v2/track/vis/episode_XXXXXX.jpg

Usage: python track_object_v2.py --episodes 0 1 2   (or --shard i/n)
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import objlib  # noqa: E402
from objlib import V2, Episode, FrameTargets, Renderer, accel_penalty, data_loss, so3_exp  # noqa: E402

THIN = {"hula_hoop"}


def build_units(T, seg_id):
    """Map frames to optimisation units: one unit per static segment, one per other frame."""
    unit = np.zeros(T, int)
    u = -1
    prev = None
    for f in range(T):
        if seg_id[f] >= 0 and seg_id[f] == prev:
            unit[f] = u
        else:
            u += 1
            unit[f] = u
        prev = seg_id[f] if seg_id[f] >= 0 else None
    return unit, u + 1


def robust_unit_pose(R, t, frames, w):
    """Weighted chordal mean rotation and weighted median translation of the given frames."""
    from scipy.spatial.transform import Rotation
    w = np.asarray(w, float) + 1e-6
    M = (R[frames] * w[:, None, None]).sum(0)
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    tm = np.array([np.median(t[frames, k]) for k in range(3)])
    return Rm, tm


class Trajectory(torch.nn.Module):
    def __init__(self, R0, t0, unit, nu, s, device):
        super().__init__()
        self.unit = torch.tensor(unit, device=device)
        self.Rb = torch.tensor(R0, dtype=torch.float32, device=device)  # [nu,3,3]
        self.tb = torch.tensor(t0, dtype=torch.float32, device=device)  # [nu,3]
        self.w = torch.nn.Parameter(torch.zeros(nu, 3, device=device))
        # translation update: (dx, dy) in metres, then a log-depth step along the viewing ray, so
        # large depth corrections (wrong initial scale / depth) are reachable in few steps
        self.dt = torch.nn.Parameter(torch.zeros(nu, 2, device=device))
        self.dlz = torch.nn.Parameter(torch.zeros(nu, device=device))
        self.ls = torch.nn.Parameter(torch.zeros((), device=device))
        self.s0 = s

    def unit_poses(self):
        t = self.tb + torch.nn.functional.pad(self.dt, (0, 1))
        return so3_exp(self.w) @ self.Rb, t * torch.exp(self.dlz)[:, None]

    def frames(self, idx=None):
        R, t = self.unit_poses()
        u = self.unit if idx is None else self.unit[idx]
        return R[u], t[u]

    def scale(self):
        return self.s0 * torch.exp(self.ls)


def optimise(traj, E, V, F, ren, tg_all, iters, lam_t, lam_r, fit_scale, thin, chunk=64, lr=(0.01, 0.004, 0.004),
             log=None):
    params = [{"params": [traj.w], "lr": lr[0]}, {"params": [traj.dt], "lr": lr[1]},
              {"params": [traj.dlz], "lr": lr[2]}]
    if fit_scale:
        params.append({"params": [traj.ls], "lr": 0.003})
    opt = torch.optim.Adam(params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr[1] * 0.05)
    T = E.T
    vis = np.flatnonzero(E.visible)
    c0 = 0.5 * (V.min(0).values + V.max(0).values)
    sub_v = torch.randperm(V.shape[0], device=V.device)[:600]
    for it in range(iters):
        opt.zero_grad()
        perm = vis[np.random.permutation(len(vis))]
        tot = 0.0
        for a in range(0, len(perm), chunk):
            idx = torch.tensor(perm[a:a + chunk], device=V.device)
            R, t = traj.frames(idx)
            verts = traj.scale() * torch.einsum("vj,bij->bvi", V, R) + t[:, None]
            cz = (traj.scale() * R @ c0 + t)[:, 2]
            sil, z = ren(verts, F)
            tg = tg_all.subset(idx)
            uv = objlib.project_points(verts[:, sub_v], ren.K) if thin else None
            l = data_loss(sil, z, tg, zc=cz, thin=thin, uv=uv).sum()
            l.backward()
            tot += float(l)
        R, t = traj.frames()
        reg = lam_t * accel_penalty(t * 100.0) + lam_r * accel_penalty(R.reshape(T, 9))
        if fit_scale:
            reg = reg + 50.0 * traj.ls ** 2  # weak prior towards the initial scale
        reg.backward()
        opt.step()
        sched.step()
        if log is not None and (it % 20 == 0 or it == iters - 1):
            log(f"  it {it:3d} data {tot / max(len(vis), 1):.4f} reg {float(reg):.3f} s {float(traj.scale()):.4f}")


def hard_ious(E, V_np, F_np, R, t, s, frames=None):
    frames = range(E.T) if frames is None else frames
    out = np.zeros(E.T)
    for f in frames:
        if E.visible[f]:
            out[f] = objlib.frame_iou(E, f, s * V_np @ R[f].T + t[f], F_np)
    return out


def single_fit(E, V, F, ren, f, R_init, t_init, s, iters=40, thin=False):
    """Single-frame pose fit from several starting rotations; returns best (R, t, loss)."""
    tg = FrameTargets(E, [f])
    B = len(R_init)
    Rb = torch.tensor(np.stack(R_init), dtype=torch.float32, device=V.device)
    tb = torch.tensor(np.stack(t_init), dtype=torch.float32, device=V.device)
    w = torch.zeros(B, 3, device=V.device, requires_grad=True)
    dxy = torch.zeros(B, 2, device=V.device, requires_grad=True)
    dlz = torch.zeros(B, device=V.device, requires_grad=True)
    opt = torch.optim.Adam([{"params": [w], "lr": 0.02}, {"params": [dxy], "lr": 0.01}, {"params": [dlz], "lr": 0.01}])
    tgB = tg.subset(torch.zeros(B, dtype=torch.long, device=V.device))
    c0 = 0.5 * (V.min(0).values + V.max(0).values)
    sub_v = torch.randperm(V.shape[0], device=V.device)[:600]
    for _ in range(iters):
        R = so3_exp(w) @ Rb
        tt = (tb + torch.nn.functional.pad(dxy, (0, 1))) * torch.exp(dlz)[:, None]
        verts = s * torch.einsum("vj,bij->bvi", V, R) + tt[:, None]
        cz = (s * R @ c0 + tt)[:, 2]
        sil, z = ren(verts, F)
        uv = objlib.project_points(verts[:, sub_v], ren.K) if thin else None
        l = data_loss(sil, z, tgB, zc=cz, thin=thin, uv=uv)
        opt.zero_grad()
        l.sum().backward()
        opt.step()
    with torch.no_grad():
        R = (so3_exp(w) @ Rb).cpu().numpy()
        tt = ((tb + torch.nn.functional.pad(dxy, (0, 1))) * torch.exp(dlz)[:, None]).cpu().numpy()
    return R, tt, l.detach().cpu().numpy()


def reinit_bad(E, V, F, V_np, F_np, ren, R, t, s, iou, thin, log):
    """Re-fit poorly matching frames from flipped / neighbouring poses."""
    from scipy.spatial.transform import Rotation
    bad = np.flatnonzero(E.visible & (iou < 0.45))
    if len(bad) == 0:
        return R, t, 0
    good = np.flatnonzero(E.visible & (iou >= 0.6))
    n_fixed = 0
    flips = [np.eye(3)] + [Rotation.from_rotvec(ax * ang).as_matrix() for ax in np.eye(3)
                            for ang in (np.pi, np.pi / 2, -np.pi / 2)]
    for f in bad[:: max(1, len(bad) // 150)]:  # cap the cost
        cands_R, cands_t = [], []
        srcs = [R[f]]
        if len(good):
            g = good[np.argmin(np.abs(good - f))]
            srcs.append(R[g])
        for Rs in srcs:
            for Fl in flips:
                cands_R.append(Rs @ Fl)
                cands_t.append(t[f])
        Rc, tc, _ = single_fit(E, V, F, ren, f, cands_R, cands_t, s, thin=thin)
        ious = np.array([objlib.frame_iou(E, f, s * V_np @ Rc[k].T + tc[k], F_np) for k in range(len(Rc))])
        # among near-best fits take the one closest in rotation to the previous frame: symmetric
        # objects (hoop, bowl, rolls) have equivalent poses with very different rotation matrices
        near = np.flatnonzero(ious >= ious.max() - 0.03)
        Rprev = R[max(f - 1, 0)]
        ang = [np.arccos(np.clip((np.trace(Rprev.T @ Rc[k]) - 1) / 2, -1, 1)) for k in near]
        k = int(near[int(np.argmin(ang))])
        if ious[k] > iou[f] + 0.1:
            R[f], t[f] = Rc[k], tc[k]
            iou[f] = ious[k]
            n_fixed += 1
    log(f"  re-init: {n_fixed}/{len(bad)} poorly fitted frames improved")
    return R, t, n_fixed


def track(ep, args, device="cuda"):
    t_start = time.time()

    def log(m):
        print(f"[ep {ep}] {m}", flush=True)

    init = np.load(os.path.join(V2, "init", f"episode_{ep:06d}.npz"))
    mesh_path = str(init["mesh"])
    cls = objlib.class_of(ep)
    thin = cls in THIN
    E = Episode(ep)
    gray = objlib.load_gray_half(ep)
    segs, seg_id = objlib.static_segments(E, gray=gray)
    del gray
    log(f"T={E.T} static segments {segs}")
    V_np, F_np = objlib.load_mesh(mesh_path, args.faces)
    V = torch.tensor(V_np, dtype=torch.float32, device=device)
    F = torch.tensor(F_np, dtype=torch.int64, device=device)
    ren = Renderer(E.K, E.H, E.W, device)
    R_in, t_in, s = init["R"].astype(np.float64), init["t"].astype(np.float64), float(init["s"])
    iou0 = hard_ious(E, V_np, F_np, R_in, t_in, s)
    log(f"init median IoU {np.median(iou0[E.visible]):.3f}")

    tg_all = FrameTargets(E, np.arange(E.T))
    R_cur, t_cur = R_in.copy(), t_in.copy()
    iou = iou0
    for rnd in range(args.rounds):
        unit, nu = build_units(E.T, seg_id)
        Ru = np.zeros((nu, 3, 3)); tu = np.zeros((nu, 3))
        for u in range(nu):
            fr = np.flatnonzero(unit == u)
            if len(fr) == 1:
                Ru[u], tu[u] = R_cur[fr[0]], t_cur[fr[0]]
            else:
                Ru[u], tu[u] = robust_unit_pose(R_cur, t_cur, fr, np.clip(iou[fr], 0.05, 1) ** 2)
        traj = Trajectory(Ru, tu, unit, nu, s, device)
        optimise(traj, E, V, F, ren, tg_all, args.iters, args.lam_t, args.lam_r,
                 fit_scale=(not thin and not args.fix_scale), thin=thin, log=log)
        with torch.no_grad():
            R, t = traj.frames()
            R_cur, t_cur = R.cpu().numpy().astype(np.float64), t.cpu().numpy().astype(np.float64)
            s = float(traj.scale())
        iou = hard_ious(E, V_np, F_np, R_cur, t_cur, s)
        log(f"round {rnd}: median IoU {np.median(iou[E.visible]):.3f} s={s:.4f}")
        if rnd < args.rounds - 1:
            R_cur, t_cur, n = reinit_bad(E, V, F, V_np, F_np, ren, R_cur, t_cur, s, iou, thin, log)
            if n == 0 and rnd > 0:
                break
    valid = E.visible & (iou > 0.3)
    out = os.path.join(V2, "track")
    os.makedirs(os.path.join(out, "vis"), exist_ok=True)
    np.savez(os.path.join(out, f"episode_{ep:06d}.npz"), R=R_cur, t=t_cur, s=s, valid=valid, iou=iou,
             iou_init=iou0, seg_id=seg_id, segs=np.array(segs).reshape(-1, 2), mesh=mesh_path,
             depth_scale=E.depth_scale)
    save_vis(E, V_np, F_np, s, R_cur, t_cur, iou, seg_id, os.path.join(out, "vis", f"episode_{ep:06d}.jpg"))
    log(f"done: median IoU {np.median(iou0[E.visible]):.3f} -> {np.median(iou[E.visible]):.3f} "
        f"static {np.mean(seg_id >= 0):.2f} time {time.time() - t_start:.0f}s")


def save_vis(E, V, F, s, R, t, iou, seg_id, path, n=12):
    import cv2
    from common import VIDEO_TMPL, read_frame
    K = E.K_full.copy(); K[:2] /= 2
    tiles = []
    for f in np.linspace(0, E.T - 1, n).astype(int):
        im = read_frame(VIDEO_TMPL.format(E.ep), f)
        im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2))
        h, w = im.shape[:2]
        sil = objlib.hard_silhouette(s * V @ R[f].T + t[f], F, K, h, w)
        im[sil] = (0.5 * im[sil] + [0, 127, 0]).astype(np.uint8)
        cnt, _ = cv2.findContours(E.obj_half[f].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cnt, -1, (255, 0, 0), 1)
        ys, xs = np.nonzero(sil | E.obj_half[f])
        if len(xs):
            cx, cy = int(xs.mean()), int(ys.mean())
            im = im[max(0, cy - 120):cy + 120, max(0, cx - 160):cx + 160]
        im = cv2.resize(im, (320, 240))
        cv2.putText(im, f"f{f} iou {iou[f]:.2f}{' S' if seg_id[f] >= 0 else ''}", (4, 18), 0, 0.55,
                    (255, 255, 0), 2)
        tiles.append(im)
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, n, 4)])
    cv2.imwrite(path, sheet[:, :, ::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--shard", type=str, default=None)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--faces", type=int, default=2000)
    ap.add_argument("--lam-t", type=float, default=0.01, help="weight of squared translation accel (cm/frame^2)")
    ap.add_argument("--lam-r", type=float, default=3.0, help="weight of squared rotation-matrix accel")
    ap.add_argument("--fix-scale", action="store_true")
    a = ap.parse_args()
    eps = a.episodes if a.episodes else list(range(30))
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        eps = eps[i::n]
    torch.manual_seed(0)
    np.random.seed(0)
    for ep in eps:
        if not os.path.exists(os.path.join(V2, "init", f"episode_{ep:06d}.npz")):
            print(f"[ep {ep}] no init", flush=True)
            continue
        track(ep, a)


if __name__ == "__main__":
    main()
