"""Object trajectory v4: render-and-compare jointly with hand-object contact and non-penetration.

Starting from the v3 trajectory (raw MoGe frame), all frames of an episode are optimised jointly
(static segments share one pose, as in track_object_v2.py):

  sum_f data_f + w_pen * pen_f + w_con * con_f  +  lambda_t |accel(t)|^2 + lambda_r |accel(R)|^2

data_f : soft-silhouette IoU, distance-to-mask and robust MoGe depth (objlib.data_loss)
pen_f  : squared penetration (cm) of the hand vertices (MHR wrist subtrees, body_v2, fixed) into
         the object, from a signed-distance grid of the submission-budgeted mesh (same inside test
         as the scorer); the body is mapped into the MoGe frame with the floor-plane ratio
con_f  : on frames where the object is held (moving, mask touching the actor), robust (Cauchy)
         gap between the closest hand vertex and the object surface beyond a per-class tolerance;
         it may only move the object along its viewing ray (the image fixes the rest), except for
         the hoop, whose tilt and depth the silhouette leaves ambiguous
The episode scale is fitted to the image and depth only (weak prior); the hands do not change it
(except for the hoop). A final stage raises w_pen and the smoothness weight. Penetration is
re-checked with the scorer's exact test on the scored frames. Hand penetrations of a few mm are
below the accuracy of the body, so they are left to fix_penetration_fingers.py.

Inputs : --init  (v3 objects: episode_XXXXXX_track.npz / _object.ply, MoGe frame)
         --sdf   (contactlib.py output: budgeted mesh + SDF grid per episode)
Outputs: --out/episode_XXXXXX_{track.npz,pose.json,object.ply,metrics.json}, --out/vis/

Usage: python track_object_v4.py --episodes 27 15 ...
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
from common import WORK, load_meta, unpack_masks  # noqa: E402
from objlib import Episode, FrameTargets, Renderer, accel_penalty, data_loss  # noqa: E402
from track_object_v2 import Trajectory, build_units, hard_ious, robust_unit_pose, save_vis  # noqa: E402

THIN = {"hula_hoop"}
# hand-surface tolerance (cm) while held; None: no contact term (handle missing from the mesh)
CONTACT_TOL = {"paint_roller": None, "black_pan": 3.0}


def held_mask(ep, t, seg_id, T, min_speed=0.004):
    """Frames where the object moves (outside static segments) and its mask touches the actor."""
    import cv2
    from scipy.ndimage import gaussian_filter1d
    md = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
    h, w = int(md["mask_h"]), int(md["mask_w"])
    om = unpack_masks(md["obj_masks"], h, w)
    pm = unpack_masks(md["person_masks"], h, w)
    speed = np.linalg.norm(np.gradient(gaussian_filter1d(t, 2, axis=0), axis=0), axis=1)
    ker = np.ones((11, 11), np.uint8)
    out = np.zeros(T, bool)
    for f in range(min(T, len(om))):
        if seg_id[f] < 0 and speed[f] > min_speed and om[f].any():
            out[f] = (cv2.dilate(pm[f].astype(np.uint8), ker) > 0)[om[f]].any()
    # close short gaps (a held object keeps being held)
    from scipy.ndimage import binary_closing
    return binary_closing(out, np.ones(9, bool)) | out


def body_hands(ep, mhr, hand_ids, body_dir):
    b = np.load(os.path.join(body_dir, f"episode_{ep:06d}.npz"))
    v, _ = mhr(b["pose"], b["scales"], b["shape"])
    return v[:, hand_ids]


def hand_terms(traj, sdf, Hm, held, tol_cm, margin_cm=0.3, con_sigma=5.0, depth_only=True):
    """Per-frame penetration [T] and contact [T] losses (cm units).

    The contact term may only move the object along its viewing ray (and change the episode
    scale) unless depth_only is False: the image already fixes the other degrees of freedom, so a
    hand that is off in the image plane (body error) cannot drag the object off its mask.
    """
    R, t = traj.frames()
    # the hands never change the episode scale of rigid objects (shrinking an object is not a
    # remedy for penetration); for the hoop, whose depth is unobservable, they may
    s = traj.scale() if not depth_only else traj.scale().detach()
    local = torch.einsum("tpi,tij->tpj", Hm - t[:, None], R) / s
    sd = sdf.query_torch(local) * s * 100.0
    pen = 0.01 * (torch.relu(margin_cm - sd) ** 2).sum(1)
    if tol_cm is None:
        con = torch.zeros_like(pen)
    else:
        if depth_only:
            u = traj.unit
            tc = ((traj.tb + torch.nn.functional.pad(traj.dt, (0, 1))).detach() * torch.exp(traj.dlz)[:, None])[u]
            lc = torch.einsum("tpi,tij->tpj", Hm - tc[:, None], R.detach()) / s
            sdc = sdf.query_torch(lc) * s * 100.0
        else:
            sdc = sd
        gap = torch.relu(sdc).amin(1)
        e = torch.relu(gap - tol_cm) / con_sigma
        con = torch.log1p(e ** 2) * held
    return pen, con, sd


def optimise(traj, E, V, F, ren, tg_all, sdf, Hm, held, tol_cm, iters, a, w_pen, fit_scale, thin, log,
             lr_scale=1.0, lam_t=None, chunk=64):
    lam_t = a.lam_t if lam_t is None else lam_t
    lr = (0.01 * lr_scale, 0.004 * lr_scale, 0.004 * lr_scale)
    params = [{"params": [traj.w], "lr": lr[0]}, {"params": [traj.dt], "lr": lr[1]},
              {"params": [traj.dlz], "lr": lr[2]}]
    if fit_scale:
        params.append({"params": [traj.ls], "lr": 0.003 * lr_scale})
    opt = torch.optim.Adam(params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr[1] * 0.05)
    T = E.T
    vis = np.flatnonzero(E.visible)
    c0 = 0.5 * (V.min(0).values + V.max(0).values)
    sub_v = torch.randperm(V.shape[0], device=V.device)[:600]
    heldt = torch.tensor(held, dtype=torch.float32, device=V.device)
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
            l = data_loss(sil, z, tg, zc=cz, thin=thin, uv=uv).sum()
            l.backward()
            tot += float(l)
        pen, con, _ = hand_terms(traj, sdf, Hm, heldt, tol_cm, depth_only=not thin)
        R, t = traj.frames()
        reg = lam_t * accel_penalty(t * 100.0) + a.lam_r * accel_penalty(R.reshape(T, 9))
        if fit_scale:
            reg = reg + a.w_scale * traj.ls ** 2
        hl = w_pen * pen.sum() + a.w_con * con.sum()
        (reg + hl).backward()
        opt.step()
        sched.step()
        if it % 25 == 0 or it == iters - 1:
            log(f"  it {it:3d} data {tot / max(len(vis), 1):.4f} pen {float(pen.mean()):.4f} "
                f"con {float(con.sum() / max(held.sum(), 1)):.4f} reg {float(reg):.3f} s {float(traj.scale()):.4f}")


def exact_penetration(R, t, s, Hm_np, mv, mf, frames):
    """Scorer penetration depth (metres, MoGe frame) of the hand vertices on the given frames."""
    local = np.einsum("tpi,tij->tpj", Hm_np[frames] - t[frames, None], R[frames]) / s
    d = cl.penetration_depth(local.reshape(-1, 3), mv, mf).reshape(local.shape[:2])
    return d * s


def track(ep, a, mhr, hand_ids, meta, ratio, scored, device="cuda"):
    t0 = time.time()

    def log(m):
        print(f"[ep {ep}] {m}", flush=True)

    cls = meta[ep]["object"]
    thin = cls in THIN
    tol = CONTACT_TOL.get(cls, 1.0)
    init = np.load(os.path.join(a.init, f"episode_{ep:06d}_track.npz"))
    mesh_path = os.path.join(a.init, f"episode_{ep:06d}_object.ply")
    mv, mf, sdf = cl.load_sdf(os.path.join(a.sdf, f"episode_{ep:06d}_sdf.npz"))
    E = Episode(ep)
    T = E.T
    # v3 rescaled some episodes about the camera (contact k); keep that calibration by rescaling the
    # MoGe depth targets with it, so the depth term does not undo it
    k3 = 1.0
    cs = os.path.join(a.init, "contact_scale.json")
    if os.path.exists(cs):
        k3 = float(json.load(open(cs)).get(str(ep), {}).get("k", 1.0))
    if E.depth is not None and k3 != 1.0:
        E.depth = E.depth * np.float32(k3)
    R_cur, t_cur, s = init["R"].astype(np.float64), init["t"].astype(np.float64), float(init["scale"])
    seg_id = init["seg_id"]
    # body hands in the MoGe frame
    Hm_np = body_hands(ep, mhr, hand_ids, a.body) * ratio[str(ep)]
    Hm = torch.tensor(Hm_np, dtype=torch.float32, device=device)
    held = held_mask(ep, t_cur, seg_id, T)
    V_np, F_np = objlib.load_mesh(mesh_path, a.faces)
    V = torch.tensor(V_np, dtype=torch.float32, device=device)
    F = torch.tensor(F_np, dtype=torch.int64, device=device)
    ren = Renderer(E.K, E.H, E.W, device)
    fr = scored.get(ep, np.arange(T))
    fr = fr[fr < T]
    iou0 = hard_ious(E, V_np, F_np, R_cur, t_cur, s)
    d0 = exact_penetration(R_cur, t_cur, s, Hm_np, mv, mf, fr)
    log(f"{cls} T={T} k3={k3:.3f} held {held.mean():.2f} init IoU {np.median(iou0[E.visible]):.3f} "
        f"pen {d0.mean() * 100 / ratio[str(ep)]:.4f} cm, frames {np.mean((d0 > 0).any(1)):.2f}")

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
    # thin objects (hoop): depth is unreliable, but the hands fix the scale while the hoop is held
    fit_scale = not a.fix_scale and (not thin or held.any())
    optimise(traj, E, V, F, ren, tg_all, sdf, Hm, held, tol, a.iters, a, a.w_pen, fit_scale, thin, log)
    log("final stage (smoother, stronger non-penetration)")
    optimise(traj, E, V, F, ren, tg_all, sdf, Hm, held, tol, a.iters2, a, a.w_pen2, fit_scale, thin, log,
             lr_scale=0.5, lam_t=a.lam_t2)
    with torch.no_grad():
        R, t = traj.frames()
        R_cur, t_cur = R.cpu().numpy().astype(np.float64), t.cpu().numpy().astype(np.float64)
        s = float(traj.scale())
        _, _, sd = hand_terms(traj, sdf, Hm, torch.tensor(held, device=device, dtype=torch.float32), tol)
        gap = torch.relu(sd).amin(1).cpu().numpy()
    iou = hard_ious(E, V_np, F_np, R_cur, t_cur, s)
    d1 = exact_penetration(R_cur, t_cur, s, Hm_np, mv, mf, fr)
    r = ratio[str(ep)]
    tb = t_cur / r

    def acc(x):
        return float(np.mean(np.linalg.norm(x[fr][2:] - 2 * x[fr][1:-1] + x[fr][:-2], axis=1)) * 100)
    met = dict(ep=ep, cls=cls, iou_init=float(np.median(iou0[E.visible])), iou=float(np.median(iou[E.visible])),
               iou_mean=float(np.mean(iou[E.visible])),
               pen_init_cm=float(d0.mean() * 100 / r), pen_cm=float(d1.mean() * 100 / r),
               pen_frames_init=float(np.mean((d0 > 0).any(1))), pen_frames=float(np.mean((d1 > 0).any(1))),
               maxdepth_cm=float(d1.max() * 100 / r) if d1.size else 0.0,
               held_gap_cm=float(np.median(gap[held])) if held.any() else None,
               acc_init_cm=acc(init["t"] / r), acc_cm=acc(tb), scale_init=float(init["scale"]), scale=s,
               time_s=time.time() - t0)
    log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in met.items()))
    os.makedirs(os.path.join(a.out, "vis"), exist_ok=True)
    stem = os.path.join(a.out, f"episode_{ep:06d}")
    shutil.copy(mesh_path, stem + "_object.ply")
    ref = int(np.argmax(np.where(E.visible, iou, -1)))
    np.savez(stem + "_track.npz", R=R_cur, t=t_cur, valid=np.ones(T, bool), scale=s, iou=iou, ref_frame=ref,
             seg_id=seg_id, held=held, visible=E.visible)
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
    ap.add_argument("--init", default=os.path.join(WORK, "objects_v3"))
    ap.add_argument("--sdf", default=os.path.join(WORK, "objects_v4", "sdf"))
    ap.add_argument("--out", default=os.path.join(WORK, "objects_v4"))
    ap.add_argument("--body", default=os.path.join(WORK, "body_v2"))
    ap.add_argument("--ratio", default=os.path.join(WORK, "moge_to_body_ratio_floor.json"))
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--iters2", type=int, default=60)
    ap.add_argument("--faces", type=int, default=2000)
    ap.add_argument("--lam-t", type=float, default=0.03)
    ap.add_argument("--lam-t2", type=float, default=0.2, help="translation accel weight, final stage")
    ap.add_argument("--lam-r", type=float, default=3.0)
    ap.add_argument("--w-pen", type=float, default=0.3)
    ap.add_argument("--w-pen2", type=float, default=1.0)
    ap.add_argument("--w-con", type=float, default=0.3)
    ap.add_argument("--w-scale", type=float, default=500.0)
    ap.add_argument("--fix-scale", action="store_true")
    a = ap.parse_args()
    from mhr_utils import DEFAULT_MHR, MHRForward
    torch.manual_seed(0)
    np.random.seed(0)
    mhr = MHRForward(DEFAULT_MHR, device="cuda")
    hand_ids = cl.hand_ids_from(mhr.model)
    meta = load_meta()
    ratio = json.load(open(a.ratio))["ratio"]
    scored = cl.scored()
    for ep in a.episodes:
        track(ep, a, mhr, hand_ids, meta, ratio, scored)


if __name__ == "__main__":
    main()
