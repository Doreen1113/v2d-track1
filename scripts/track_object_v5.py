"""Object trajectory v5: the v4 fit plus support-surface calibration and rest constraints.

Silhouettes fix an object's pose only up to a scaling about the camera centre; v2-v4 resolve it
with MoGe depth (plus hand contact), which leaves a per-episode depth/size bias of up to 15-20 %
(the object then floats above or sinks below the surface it rests on). The body gives an
independent, metric reference: its floor plane (refine_body, feet contacts). For every static
segment (object at rest, shared pose) the height of the lowest mesh point above the body floor
is measured; segments are classified as floor-resting (|h| < 0.3 m) or table-resting
(0.6 < h < 1.0 m, the round table of the recordings) and

  1. the episode is rescaled about the camera centre (k on scale and translation): floor
     segments pin k exactly (lowest point on the floor); table-only episodes are corrected when
     their table height is more than 8 cm off the common table height AND the body-frame size
     disagrees with the class median in the same direction (geometric mean of the two
     estimates; the same physical object appears in the three episodes of a class);
  2. the joint fit of track_object_v4.py is repeated from the rescaled trajectory with the scale
     fixed, a reduced MoGe depth weight, and two support terms: resting frames keep their lowest
     point at the surface height (0 for the floor, the episode's own median for the table) and
     no frame puts the object below the floor.

Inputs : --init  (objects_v4: episode_XXXXXX_track.npz / _object.ply, MoGe frame)
         --sdf   (objects_v4/sdf, budgeted mesh + SDF grid of the same mesh)
         --body  (floor plane floor_n / floor_d per episode, hands), --ratio (MoGe/body)
Outputs: --out/episode_XXXXXX_{track.npz,pose.json,object.ply,metrics.json}, --out/vis/,
         --out/calibration.json

Usage: python track_object_v5.py --episodes 27 15 ...       (--dry: calibration only, no GPU)
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contactlib as cl  # noqa: E402
import objlib  # noqa: E402
from common import WORK, load_meta  # noqa: E402
from objlib import Episode, FrameTargets, Renderer, accel_penalty, data_loss  # noqa: E402
from track_object_v2 import Trajectory, build_units, hard_ious, robust_unit_pose, save_vis  # noqa: E402
from track_object_v4 import CONTACT_TOL, THIN, body_hands, exact_penetration, hand_terms, held_mask  # noqa: E402

FLOOR_BAND = (-0.30, 0.30)
TABLE_BAND = (0.60, 1.00)
TABLE_H = 0.80          # common resting height of the table-resting episodes (median over classes)
TABLE_TOL = 0.08
FLOOR_DEADBAND = 0.02
K_RANGE = (0.75, 1.30)


def lowest_height(V, R, t, s, n, d, r):
    """Height (body frame, m) of the lowest mesh point above the floor n.x = d, per frame.

    V, R, t, s in the MoGe frame; r = MoGe/body ratio (the plane n.x = r d in the MoGe frame).
    """
    H = np.empty(len(t))
    for f in range(len(t)):
        H[f] = ((s * V @ R[f].T + t[f]) @ n - r * d).min() / r
    return H


def load_floor(body_dir, ep):
    b = np.load(os.path.join(body_dir, f"episode_{ep:06d}.npz"))
    return np.asarray(b["floor_n"], float), float(b["floor_d"])


def class_scales(init_dir, ratio, meta):
    out = {}
    for ep, m in meta.items():
        p = os.path.join(init_dir, f"episode_{ep:06d}_track.npz")
        if os.path.exists(p):
            out.setdefault(m["object"], {})[ep] = float(np.load(p)["scale"]) / ratio[str(ep)]
    return out


def calibrate(ep, init_dir, body_dir, ratio, meta, cls_scales, V=None):
    """Segment classification and the episode rescale k; returns a dict (k, targets, ...)."""
    tr = np.load(os.path.join(init_dir, f"episode_{ep:06d}_track.npz"))
    if V is None:
        V, _ = objlib.load_mesh(os.path.join(init_dir, f"episode_{ep:06d}_object.ply"), 2000)
    r = ratio[str(ep)]
    n, d = load_floor(body_dir, ep)
    R, t, s, seg = tr["R"], tr["t"], float(tr["scale"]), tr["seg_id"]
    h = lowest_height(V, R, t, s, n, d, r)
    kind = np.zeros(len(t), int)  # 0 none, 1 floor, 2 table
    segs = []
    for sg in range(seg.max() + 1):
        f = np.flatnonzero(seg == sg)
        hm = float(np.median(h[f]))
        kd = 1 if FLOOR_BAND[0] < hm < FLOOR_BAND[1] else 2 if TABLE_BAND[0] < hm < TABLE_BAND[1] else 0
        kind[f] = kd
        segs.append(dict(seg=sg, frames=[int(f[0]), int(f[-1])], height_cm=hm * 100, kind=kd))
    k, how = 1.0, "none"
    a = h + d  # n.x of the lowest point (body frame); height after rescale = k a - d
    if (kind == 1).any():
        hf = float(np.median(h[kind == 1]))
        if abs(hf) > FLOOR_DEADBAND:
            k = float(np.median(d / a[kind == 1]))
            how = f"floor: median height {hf * 100:.1f} cm"
        else:
            how = f"floor within deadband ({hf * 100:.1f} cm)"
    elif (kind == 2).any():
        ht = float(np.median(h[kind == 2]))
        if abs(ht - TABLE_H) > TABLE_TOL:
            k_t = float(np.median((TABLE_H + d) / a[kind == 2]))
            cls = meta[ep]["object"]
            others = [v for e, v in cls_scales[cls].items() if e != ep]
            k_c = float(np.median(others)) / cls_scales[cls][ep] if others else 1.0
            if (k_t - 1) * (k_c - 1) > 0:
                k = float(np.exp(0.5 * (np.log(k_t) + np.log(k_c))))
                how = f"table: height {ht * 100:.1f} cm (k_table {k_t:.3f}), class size (k_class {k_c:.3f})"
            else:
                how = f"table: height {ht * 100:.1f} cm off but class size disagrees (k_table {k_t:.3f}, k_class {k_c:.3f})"
        else:
            how = f"table height {ht * 100:.1f} cm, no change"
    k = float(np.clip(k, *K_RANGE))
    h_new = k * a - d
    target = np.full(len(t), np.nan)
    target[kind == 1] = 0.0
    if (kind == 2).any():
        target[kind == 2] = float(np.median(h_new[kind == 2]))
    return dict(ep=ep, cls=meta[ep]["object"], k=k, how=how, segments=segs, kind=kind, target=target,
                h_before=h, h_after=h_new, scale_body_before=s / r, scale_body_after=k * s / r,
                floor_frac=float((kind == 1).mean()), table_frac=float((kind == 2).mean()))


def optimise(traj, E, V, F, ren, tg_all, sdf, Hm, held, tol_cm, floor, iters, a, w_pen, thin, log,
             lr_scale=1.0, lam_t=None, chunk=64):
    """track_object_v4.optimise with a fixed scale, reduced depth weight and the support terms."""
    n_m, d_m, target, r = floor  # plane in the MoGe frame (n.x = d_m), targets in body metres
    lam_t = a.lam_t if lam_t is None else lam_t
    lr = (0.01 * lr_scale, 0.004 * lr_scale, 0.004 * lr_scale)
    params = [{"params": [traj.w], "lr": lr[0]}, {"params": [traj.dt], "lr": lr[1]},
              {"params": [traj.dlz], "lr": lr[2]}]
    opt = torch.optim.Adam(params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr[1] * 0.05)
    T = E.T
    vis = np.flatnonzero(E.visible)
    c0 = 0.5 * (V.min(0).values + V.max(0).values)
    sub_v = torch.randperm(V.shape[0], device=V.device)[:600]
    heldt = torch.tensor(held, dtype=torch.float32, device=V.device)
    has_t = torch.tensor(np.isfinite(target), device=V.device)
    tgt = torch.tensor(np.nan_to_num(target), dtype=torch.float32, device=V.device)
    for it in range(iters):
        opt.zero_grad()
        perm = vis[np.random.permutation(len(vis))]
        tot = 0.0
        for b in range(0, len(perm), chunk):
            idx = torch.tensor(perm[b:b + chunk], device=V.device)
            R, t = traj.frames(idx)
            verts = traj.scale() * torch.einsum("vj,bij->bvi", V, R) + t[:, None]
            cz = (traj.scale() * R @ c0 + t)[:, 2]
            sil, z = ren(verts, F)
            tg = tg_all.subset(idx)
            uv = objlib.project_points(verts[:, sub_v], ren.K) if thin else None
            l = data_loss(sil, z, tg, zc=cz, thin=thin, uv=uv, w_depth=a.w_depth).sum()
            l.backward()
            tot += float(l)
        pen, con, _ = hand_terms(traj, sdf, Hm, heldt, tol_cm, depth_only=not thin)
        R, t = traj.frames()
        # support: lowest point height (body cm) above the floor, all frames
        hv = (traj.scale() * torch.einsum("vj,tij->tvi", V, R) + t[:, None]) @ n_m - d_m
        h_cm = hv.amin(1) / r * 100.0
        rest = torch.where(has_t, (h_cm - tgt * 100.0) ** 2, torch.zeros_like(h_cm)).sum()
        below = (torch.relu(-h_cm - 1.0) ** 2).sum()
        sup = a.w_rest * rest + a.w_floor * below
        reg = lam_t * accel_penalty(t * 100.0) + a.lam_r * accel_penalty(R.reshape(T, 9))
        hl = w_pen * pen.sum() + a.w_con * con.sum()
        (reg + hl + sup).backward()
        opt.step()
        sched.step()
        if it % 25 == 0 or it == iters - 1:
            log(f"  it {it:3d} data {tot / max(len(vis), 1):.4f} pen {float(pen.mean()):.4f} "
                f"con {float(con.sum() / max(held.sum(), 1)):.4f} reg {float(reg):.3f} "
                f"rest {float(rest / max(int(has_t.sum()), 1)) ** 0.5:.2f}cm below {float(below):.2f}")


def track(ep, a, mhr, hand_ids, meta, ratio, scored, cls_scales, calib_log, device="cuda"):
    t0 = time.time()

    def log(m):
        print(f"[ep {ep}] {m}", flush=True)

    cls = meta[ep]["object"]
    thin = cls in THIN
    tol = CONTACT_TOL.get(cls, 1.0)
    init = np.load(os.path.join(a.init, f"episode_{ep:06d}_track.npz"))
    mesh_path = os.path.join(a.init, f"episode_{ep:06d}_object.ply")
    V_np, F_np = objlib.load_mesh(mesh_path, a.faces)
    cal = calibrate(ep, a.init, a.body, ratio, meta, cls_scales, V_np)
    k = cal["k"]
    log(f"{cls} k={k:.3f} ({cal['how']}); segments " + ", ".join(
        f"{s['frames'][0]}-{s['frames'][1]}:{s['height_cm']:.0f}cm{'FT'[s['kind'] - 1] if s['kind'] else ''}"
        for s in cal["segments"]))
    calib_log[str(ep)] = {kk: vv for kk, vv in cal.items() if kk not in ("kind", "target", "h_before", "h_after")}
    if a.dry:
        return
    mv, mf, sdf = cl.load_sdf(os.path.join(a.sdf, f"episode_{ep:06d}_sdf.npz"))
    E = Episode(ep)
    T = E.T
    r = ratio[str(ep)]
    # keep the v3 contact calibration of the depth targets (as v4), then our k
    k3 = 1.0
    cs = os.path.join(a.init3, "contact_scale.json")
    if os.path.exists(cs):
        k3 = float(json.load(open(cs)).get(str(ep), {}).get("k", 1.0))
    if E.depth is not None:
        E.depth = E.depth * np.float32(k3 * k)
    R_cur, t_cur, s = init["R"].astype(np.float64), init["t"].astype(np.float64) * k, float(init["scale"]) * k
    seg_id = init["seg_id"]
    Hm_np = body_hands(ep, mhr, hand_ids, a.body) * r
    Hm = torch.tensor(Hm_np, dtype=torch.float32, device=device)
    held = held_mask(ep, t_cur, seg_id, T)
    V = torch.tensor(V_np, dtype=torch.float32, device=device)
    F = torch.tensor(F_np, dtype=torch.int64, device=device)
    ren = Renderer(E.K, E.H, E.W, device)
    fr = scored.get(ep, np.arange(T))
    fr = fr[fr < T]
    iou0 = hard_ious(E, V_np, F_np, R_cur, t_cur, s)
    d0 = exact_penetration(R_cur, t_cur, s, Hm_np, mv, mf, fr)
    log(f"T={T} k3={k3:.3f} held {held.mean():.2f} floor {cal['floor_frac']:.2f} table {cal['table_frac']:.2f} "
        f"init IoU {np.median(iou0[E.visible]):.3f} pen {d0.mean() * 100 / r:.4f} cm")

    n, d = load_floor(a.body, ep)
    floor = (torch.tensor(n, dtype=torch.float32, device=device), float(r * d), cal["target"], r)
    tg_all = FrameTargets(E, np.arange(T))
    unit, nu = build_units(T, seg_id)
    Ru = np.zeros((nu, 3, 3)); tu = np.zeros((nu, 3))
    for u in range(nu):
        f_ = np.flatnonzero(unit == u)
        if len(f_) == 1:
            Ru[u], tu[u] = R_cur[f_[0]], t_cur[f_[0]]
        else:
            Ru[u], tu[u] = robust_unit_pose(R_cur, t_cur, f_, np.clip(iou0[f_], 0.05, 1) ** 2)
    traj = Trajectory(Ru, tu, unit, nu, s, device)
    optimise(traj, E, V, F, ren, tg_all, sdf, Hm, held, tol, floor, a.iters, a, a.w_pen, thin, log,
             lr_scale=a.lr_scale)
    log("final stage (smoother, stronger non-penetration)")
    optimise(traj, E, V, F, ren, tg_all, sdf, Hm, held, tol, floor, a.iters2, a, a.w_pen2, thin, log,
             lr_scale=0.5 * a.lr_scale, lam_t=a.lam_t2)
    with torch.no_grad():
        R, t = traj.frames()
        R_cur, t_cur = R.cpu().numpy().astype(np.float64), t.cpu().numpy().astype(np.float64)
        s = float(traj.scale())
        _, _, sd = hand_terms(traj, sdf, Hm, torch.tensor(held, device=device, dtype=torch.float32), tol)
        gap = torch.relu(sd).amin(1).cpu().numpy()
    iou = hard_ious(E, V_np, F_np, R_cur, t_cur, s)
    d1 = exact_penetration(R_cur, t_cur, s, Hm_np, mv, mf, fr)
    h1 = lowest_height(V_np, R_cur, t_cur, s, n, d, r)
    tb = t_cur / r
    tgt = cal["target"]
    has = np.isfinite(tgt)

    def acc(x):
        return float(np.mean(np.linalg.norm(x[fr][2:] - 2 * x[fr][1:-1] + x[fr][:-2], axis=1)) * 100)
    met = dict(ep=ep, cls=cls, k=k, iou_init=float(np.median(iou0[E.visible])), iou=float(np.median(iou[E.visible])),
               iou_mean=float(np.mean(iou[E.visible])),
               pen_init_cm=float(d0.mean() * 100 / r), pen_cm=float(d1.mean() * 100 / r),
               pen_frames_init=float(np.mean((d0 > 0).any(1))), pen_frames=float(np.mean((d1 > 0).any(1))),
               maxdepth_cm=float(d1.max() * 100 / r) if d1.size else 0.0,
               held_gap_cm=float(np.median(gap[held])) if held.any() else None,
               rest_rms_cm=float(np.sqrt(np.mean((h1[has] - tgt[has]) ** 2)) * 100) if has.any() else None,
               below_floor_cm=float(np.maximum(-h1, 0).max() * 100),
               acc_init_cm=acc(init["t"] / r), acc_cm=acc(tb), scale_body=s / r, time_s=time.time() - t0)
    log(" ".join(f"{kk}={vv:.4f}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in met.items()))
    os.makedirs(os.path.join(a.out, "vis"), exist_ok=True)
    stem = os.path.join(a.out, f"episode_{ep:06d}")
    shutil.copy(mesh_path, stem + "_object.ply")
    ref = int(np.argmax(np.where(E.visible, iou, -1)))
    np.savez(stem + "_track.npz", R=R_cur, t=t_cur, valid=np.ones(T, bool), scale=s, iou=iou, ref_frame=ref,
             seg_id=seg_id, held=held, visible=E.visible, support_kind=cal["kind"], height=h1)
    pose = dict(episode=ep, object=cls, frame=ref,
                convention="point_cam = s * R @ point_mesh + t; camera frame OpenCV; metres (raw MoGe-2 "
                           "frame); point_mesh = vertices of episode_XXXXXX_object.ply",
                R=R_cur[ref].tolist(), t=t_cur[ref].tolist(), s=s, mesh_source=mesh_path, iou=float(iou[ref]))
    json.dump(pose, open(stem + "_pose.json", "w"), indent=1)
    json.dump(met, open(stem + "_metrics.json", "w"), indent=1)
    save_vis(E, V_np, F_np, s, R_cur, t_cur, iou, seg_id, os.path.join(a.out, "vis", f"episode_{ep:06d}.jpg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--init", default=os.path.join(WORK, "objects_v4"))
    ap.add_argument("--init3", default=os.path.join(WORK, "objects_v3"), help="v3 dir (contact_scale.json)")
    ap.add_argument("--sdf", default=os.path.join(WORK, "objects_v4", "sdf"))
    ap.add_argument("--out", default=os.path.join(WORK, "objects_v5"))
    ap.add_argument("--body", default=os.path.join(WORK, "body_v3"))
    ap.add_argument("--ratio", default=os.path.join(WORK, "moge_to_body_ratio_floor_b3.json"))
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--iters2", type=int, default=50)
    ap.add_argument("--lr-scale", type=float, default=0.5)
    ap.add_argument("--faces", type=int, default=2000)
    ap.add_argument("--lam-t", type=float, default=0.03)
    ap.add_argument("--lam-t2", type=float, default=0.2)
    ap.add_argument("--lam-r", type=float, default=3.0)
    ap.add_argument("--w-pen", type=float, default=0.3)
    ap.add_argument("--w-pen2", type=float, default=1.0)
    ap.add_argument("--w-con", type=float, default=0.3)
    ap.add_argument("--w-depth", type=float, default=0.25, help="MoGe depth weight (v4: 0.5)")
    ap.add_argument("--w-rest", type=float, default=0.05, help="per resting frame, (cm)^2")
    ap.add_argument("--w-floor", type=float, default=0.05, help="per frame below the floor, (cm)^2")
    ap.add_argument("--dry", action="store_true", help="print the calibration only")
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    meta = load_meta()
    ratio = json.load(open(a.ratio))["ratio"]
    scored = cl.scored()
    cls_scales = class_scales(a.init, ratio, meta)
    mhr = hand_ids = None
    if not a.dry:
        from mhr_utils import DEFAULT_MHR, MHRForward
        mhr = MHRForward(DEFAULT_MHR, device="cuda")
        hand_ids = cl.hand_ids_from(mhr.model)
    os.makedirs(a.out, exist_ok=True)
    cpath = os.path.join(a.out, "calibration.json")
    calib = json.load(open(cpath)) if os.path.exists(cpath) and not a.dry else {}
    for ep in a.episodes:
        track(ep, a, mhr, hand_ids, meta, ratio, scored, cls_scales, calib)
        if not a.dry:
            json.dump(calib, open(cpath, "w"), indent=1, default=float)
    if a.dry:
        for ep in a.episodes:
            c = calib[str(ep)]
            print(f"ep {ep:2d} {c['cls']:18s} k {c['k']:.3f}  size {c['scale_body_before']:.3f}->{c['scale_body_after']:.3f}  {c['how']}")


if __name__ == "__main__":
    main()
