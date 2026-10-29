"""Whole-sequence body refinement, version 5: the body_v2 setting, optionally on a fused HMR input, with two optional
terms (upright prior, MoGe relative-depth target; both off by default).

Same pipeline and default weights as refine_body_sequence.py (body_v2: per-episode median identity,
sig_acc_rel 0.008; stage B without floor, MoGe floor plane, stages C/D with floor and contacts). Added in
stages C/D, on frames detected as standing or walking:
  - spine prior: the hip-centre -> neck direction is pulled towards the floor normal (robust, sig_grav_spine);
    grav_mode "depth" (default) penalises only its component along the camera's viewing direction (projected on
    the floor), i.e. the lean towards / away from the camera that 2D keypoints barely constrain; "full" penalises
    the whole tilt (also the sideways lean, which the 2D keypoints do see: costs more held-out 2D error)
  - optional support prior (w_grav_leg, off by default): the ankle-midpoint -> hip-centre direction towards the
    floor normal. Off because on body_v2 standing frames this axis has real per-episode lateral offsets of
    4-6 deg (stance width, weight on one leg) and ~2 deg forward (hip joint ahead of the ankles).
  - hip height: the hip centre's height above the floor is pulled towards the episode's straight-leg standing
    height (median over standing frames with straight knees, sig_hip)
Standing / walking (conservative): at least one foot in floor contact, the other sole below st_swing, both knee
flexions below st_knee, trunk-thigh angle below st_hipflex, spine tilt below st_tilt, hip height above
st_hh * leg length; runs shorter than st_min_len frames are dropped and the weight ramps in / out at the run ends.
Sitting, kneeling, bending, jumping and one-leg poses get no upright prior.
The floor offset (and so floor_moge_over_body) comes from stage B exactly as in body_v2.

Optional --body-in: per-frame HMR estimates in the work/body format (e.g. a fused ensemble).
Optional --depth-target DIR (compute_moge_depth_target.py): the pelvis depth is pulled towards z_target (scale-free
MoGe relative depth of the person; correction clipped to +-mdepth_clip, robust term with sig_mdepth, saturating
beyond mdepth_c sigmas) in all stages; the floor offset then changes slightly, so build the
object scale ratio from this run's metrics (make_floor_ratio.py --body OUT).

The upright prior is OFF by default (--upright enables it): with it, CD-H got worse on Kaggle (S2, +0.24) even though
the lean towards the camera became more consistent; held-out 2D error had pointed the same way (+0.02..0.04 px).
Without --upright and without --depth-target the stages and weights are exactly body_v2's.

Output: <out>/episode_XXXXXX.npz (pose [T,136], scales [68], shape [45], valid, contact, floor_n, floor_d,
standing [T] weight, holdout, ...) and <out>/metrics/episode_XXXXXX.json (floor, floor_moge_over_body, upright
diagnostics before / after).

  python refine_body_v5.py --all --no-upright --body-in WORK/body_ens2 --out WORK/body_vA      # V-A
  python refine_body_v5.py --all --no-upright --holdout --body-in WORK/body_ens2 --out DEV/ho_vA # fit even frames
  python refine_body_v5.py --all --upright --out WORK/body_v5a                                   # upright prior
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.ndimage import binary_opening, gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refine_body_sequence as v2  # noqa: E402
from mhr_utils import FLIP  # noqa: E402
from refine_body_sequence import (BODY70, COCO_TO_70, FOOT_KP, J22, SOLE_KP, Body, acc, detect_contacts,  # noqa: E402
                                  fill_invalid, fit_floor_moge, forward_np, gmof, project_t,
                                  stationary_feet_2d)

WORK = f"{v2.ROOT}/work"
OUT_DIR = f"{WORK}/body_v5"

# MHR70 keypoints
HIP_L, HIP_R, KNEE_L, KNEE_R, ANK_L, ANK_R, NECK = 9, 10, 11, 12, 13, 14, 69

DEFAULT_CFG = dict(v2.DEFAULT_CFG)
DEFAULT_CFG.update(
    w_grav=1.0, sig_grav_spine=4.0, w_grav_leg=0.0, sig_grav_leg=6.0, grav_c=3.0,  # degrees; robust cut-off in sigmas
    grav_mode="depth",                                              # "depth" or "full" (whole tilt)
    w_mdepth=1.0, sig_mdepth=0.03, mdepth_c=4.0, mdepth_clip=0.08,  # MoGe depth target (only with --depth-target)
    w_hip=1.0, sig_hip=0.04,                                        # metres
    st_knee=30.0, st_hipflex=25.0, st_tilt=20.0, st_hh=0.85, st_swing=0.25, st_min_len=9, st_ramp=2.0,
    st_knee_straight=12.0,
)


# ------------------------------------------------------------------------------------- data
def load_episode(ep, body_dir):
    b = dict(np.load(f"{body_dir}/episode_{ep:06d}.npz"))
    T = len(b["pose"])
    valid = b["valid"].astype(bool)
    vit = torch.load(f"{v2.VIT_DIR}/episode_{ep:06d}_feat.pt", map_location="cpu", weights_only=False)
    kv = vit["kp2d"].numpy().astype(np.float64)
    vv = vit["detected"].numpy().astype(bool) if "detected" in vit else vit["valid"].numpy().astype(bool)
    if len(kv) != T:
        raise ValueError(f"ep {ep}: ViTPose length {len(kv)} != {T}")
    kv[~vv, :, 2] = 0
    return b, T, valid, kv


# ------------------------------------------------------------------------------------- upright geometry
def _ang(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def upright_geometry(kp, floor):
    """Per-frame quantities used by the standing detector (kp [T,70,3] metres, floor (n, d))."""
    n, d = floor
    hip = kp[:, [HIP_L, HIP_R]].mean(1)
    spine = kp[:, NECK] - hip
    ank = kp[:, [ANK_L, ANK_R]].mean(1)
    g = dict(
        tilt=_ang(spine, n[None]),
        leg_tilt=_ang(hip - ank, n[None]),
        knee_l=_ang(kp[:, KNEE_L] - kp[:, HIP_L], kp[:, ANK_L] - kp[:, KNEE_L]),
        knee_r=_ang(kp[:, KNEE_R] - kp[:, HIP_R], kp[:, ANK_R] - kp[:, KNEE_R]),
        hipflex=_ang(spine, hip - kp[:, [KNEE_L, KNEE_R]].mean(1)),
        hh=hip @ n - d,
        leg=0.5 * (np.linalg.norm(kp[:, KNEE_L] - kp[:, HIP_L], axis=1) + np.linalg.norm(kp[:, ANK_L] - kp[:, KNEE_L], axis=1)
                   + np.linalg.norm(kp[:, KNEE_R] - kp[:, HIP_R], axis=1) + np.linalg.norm(kp[:, ANK_R] - kp[:, KNEE_R], axis=1)),
        sole=np.stack([(kp[:, SOLE_KP[s]] @ n - d).min(1) for s in range(2)], 1),
    )
    return g


def detect_standing(kp, floor, contact, valid, cfg):
    """Standing / walking frames -> (bool mask, weight in [0, 1] with ramps, geometry)."""
    g = upright_geometry(kp, floor)
    st = (contact.any(1) & (g["sole"].max(1) < cfg["st_swing"])
          & (np.maximum(g["knee_l"], g["knee_r"]) < cfg["st_knee"]) & (g["hipflex"] < cfg["st_hipflex"])
          & (g["tilt"] < cfg["st_tilt"]) & (g["hh"] > cfg["st_hh"] * g["leg"]) & valid)
    L = int(cfg["st_min_len"])
    if L > 1:
        st = binary_opening(st, structure=np.ones(L, bool))
    w = np.minimum(st.astype(np.float64), gaussian_filter1d(st.astype(np.float64), cfg["st_ramp"], mode="constant"))
    return st, w, g


def hip_target(g, st, cfg):
    straight = st & (np.maximum(g["knee_l"], g["knee_r"]) < cfg["st_knee_straight"])
    if straight.sum() >= 20:
        return float(np.median(g["hh"][straight])), int(straight.sum())
    if st.sum() >= 20:
        return float(np.percentile(g["hh"][st], 75)), int(st.sum())
    return None, 0


def upright_stats(kp, floor, st, frames):
    """Tilt statistics on standing frames (all and scored) and on given frames."""
    g = upright_geometry(kp, floor)
    out = {}
    for name, m in (("all", st), ("scored", st & frames)):
        if m.sum() > 0:
            out[f"tilt_med_{name}"] = float(np.median(g["tilt"][m]))
            out[f"tilt_p90_{name}"] = float(np.percentile(g["tilt"][m], 90))
            out[f"legtilt_med_{name}"] = float(np.median(g["leg_tilt"][m]))
            out[f"hh_sd_{name}"] = float(np.std(g["hh"][m]))
    return out, g


# ------------------------------------------------------------------------------------- fitting
def optimise(body, init, prior, K, kv, ksam, wdata, cfg, floor=None, contact=None, grav=None, iters=600, lr=0.01,
             log=None, tag="", ztgt=None):
    """refine_body_sequence.optimise plus the upright terms (grav = (weight [T], hip target or None))."""
    dev = body.dev
    T = init.shape[0]
    tt = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    x = torch.nn.Parameter(tt(init).clone())
    pr, kvt, kst, wd, Kt = tt(prior), tt(kv), tt(ksam), tt(wdata), tt(K)
    cidx = torch.tensor(COCO_TO_70, device=dev)
    sig_th = tt(np.r_[np.full(3, 1e9), np.full(3, cfg["sig_rootrot"]), np.full(130, cfg["sig_pose"])])
    opt = torch.optim.Adam([x], lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.6), int(iters * 0.85)], 0.3)
    if floor is not None:
        fn, fd = tt(floor[0]), float(floor[1])
    if contact is not None:
        ct = tt(contact)
    use_g = grav is not None and floor is not None and float(np.sum(grav[0])) > 0
    if use_g:
        wg = tt(grav[0])
        htgt = grav[1]
        s2_sp = float(np.sin(np.radians(cfg["sig_grav_spine"])) ** 2)
        s2_lg = float(np.sin(np.radians(cfg["sig_grav_leg"])) ** 2)
        c2 = cfg["grav_c"] ** 2
        n_np = np.asarray(floor[0], np.float64)
        camf = np.array([0.0, 0.0, 1.0]) - n_np * n_np[2]   # camera viewing direction projected on the floor
        camf_t = tt(camf / np.linalg.norm(camf))
    samw = torch.ones(70, device=dev) * cfg["w_sam"]
    samw[15:21] = cfg["w_sam_feet"]
    samw[21:62] = cfg["w_sam"] * 0.3
    if ztgt is not None:
        zt = tt(np.nan_to_num(ztgt[0], nan=0.0))
        zm = tt(ztgt[1].astype(np.float64))
    for it in range(iters):
        opt.zero_grad()
        kp, j = body(x)
        uv = project_t(kp, Kt)
        r = (uv[:, cidx] - kvt[..., :2]) / cfg["sig_vit"]
        c = kvt[..., 2] * (kvt[..., 2] > cfg["vit_thr"])
        L_vit = (gmof(r.square().sum(-1), 9.0) * c).sum(1) * wd
        r = (uv - kst) / cfg["sig_sam"]
        L_sam = (gmof(r.square().sum(-1), 9.0) * samw).sum(1) * wd
        L_pr = (((x - pr) / sig_th).square().sum(1) + ((x[:, 2] - pr[:, 2]) * 0.1 / cfg["sig_depth"]).square()) * wd
        loss = (L_vit.sum() + L_sam.sum() + L_pr.sum()) / T
        jb = torch.cat([j[:, J22], kp[:, BODY70]], 1)
        root = j[:, 1]
        rel = jb - root[:, None]
        a_rel = acc(rel) / cfg["sig_acc_rel"]
        a_root = acc(root)
        L_s = a_rel.square().sum((1, 2)).sum() \
            + (a_root[:, :2] / cfg["sig_acc_root_xy"]).square().sum() \
            + (a_root[:, 2] / cfg["sig_acc_root_z"]).square().sum() \
            + (acc(x[:, 3:]) / cfg["sig_acc_param"]).square().sum() \
            + ((x[1:, 2] - x[:-1, 2]) * 0.1 / cfg["sig_vel_root_z"]).square().sum()
        loss = loss + cfg["w_smooth"] * L_s / T
        L_f = torch.zeros((), device=dev)
        if floor is not None:
            hgt = kp[:, 15:21] @ fn - fd
            L_f = (torch.relu(-hgt) / 0.01).square().sum()
            if contact is not None:
                for s in range(2):
                    sole = kp[:, SOLE_KP[s]]
                    hs = sole @ fn - fd
                    hmin = -cfg["soft_k"] * torch.logsumexp(-hs / cfg["soft_k"], 1)
                    L_f = L_f + (ct[:, s] * ((hmin - cfg["contact_h"]) / cfg["sig_contact_h"]).square()).sum()
                    fk = kp[:, FOOT_KP[s]]
                    vv = fk[1:] - fk[:-1]
                    both = ct[1:, s] * ct[:-1, s]
                    L_f = L_f + (both[:, None] * (vv / cfg["sig_slide"]).square().sum(-1)).sum()
            L_f = L_f * cfg["w_floor"] / T
        loss = loss + L_f
        L_g = torch.zeros((), device=dev)
        if use_g:
            hip = 0.5 * (kp[:, HIP_L] + kp[:, HIP_R])
            sp = kp[:, NECK] - hip
            sp = sp / sp.norm(dim=-1, keepdim=True)
            lg = hip - 0.5 * (kp[:, ANK_L] + kp[:, ANK_R])
            lg = lg / lg.norm(dim=-1, keepdim=True)
            if cfg["grav_mode"] == "depth":
                sin2_sp = (sp @ camf_t).square()                # sin^2 of the lean towards / away from the camera
            else:
                sin2_sp = (1 - (sp @ fn).square()).clamp(min=0)   # sin^2 of the angle to the floor normal
            sin2_lg = (1 - (lg @ fn).square()).clamp(min=0)
            Lg = gmof(sin2_sp / s2_sp, c2) + cfg["w_grav_leg"] * gmof(sin2_lg / s2_lg, c2)
            L_g = cfg["w_grav"] * (wg * Lg).sum() / T
            if htgt is not None and cfg["w_hip"] > 0:
                hh = hip @ fn - fd
                L_g = L_g + cfg["w_hip"] * (wg * gmof(((hh - htgt) / cfg["sig_hip"]).square(), c2)).sum() / T
            loss = loss + L_g
        L_md = torch.zeros((), device=dev)
        if ztgt is not None:
            r2 = ((j[:, 1, 2] - zt) / cfg["sig_mdepth"]).square()
            L_md = cfg["w_mdepth"] * (zm * gmof(r2, cfg["mdepth_c"] ** 2)).sum() / T  # robust: MoGe artefacts
            loss = loss + L_md
        loss.backward()
        opt.step()
        sched.step()
        if log is not None and (it % 200 == 0 or it == iters - 1):
            log(f"  [{tag}] it {it:4d} loss {loss.item():9.3f} vit {L_vit.sum().item() / T:7.3f} "
                f"sam {L_sam.sum().item() / T:7.3f} prior {L_pr.sum().item() / T:7.3f} "
                f"smooth {cfg['w_smooth'] * L_s.item() / T:7.3f} floor {L_f.item():7.3f} upright {L_g.item():7.3f}"
                f" mdepth {L_md.item():7.3f}")
    return x.detach().cpu().numpy().astype(np.float64)


# ------------------------------------------------------------------------------------- main
def run_episode(ep, args, cfg, scored, log):
    t0 = time.time()
    b, T, valid, kv = load_episode(ep, args.body_in)
    K = np.array([float(b["fx"]), float(b["fy"]), float(b["cx"]), float(b["cy"])])
    v2.METRIC_K = K
    dev = torch.device("cuda")
    shape = np.median(b["shape"][valid], 0)
    scales = np.median(b["scales"][valid], 0)
    pose0 = fill_invalid(b["pose"], valid)
    pose0[:, 3:] = np.unwrap(pose0[:, 3:], axis=0)
    ksam = fill_invalid(b["kp2d"], valid)
    wdata = valid.astype(np.float64)
    kv = kv.copy()
    kv[~valid, :, 2] = 0
    ho = np.zeros(T, bool)
    if args.holdout:
        ho[1::2] = True
        wdata[ho] = 0
    body = Body(dev, shape, scales)
    sc = scored.get(ep, [])
    scm = np.zeros(T, bool)
    scm[[f for f in sc if f < T]] = True
    f0 = sc[0] if sc else int(np.flatnonzero(valid)[0])
    st2d = stationary_feet_2d(kv, ksam, valid)
    res = {"episode": ep, "T": T, "n_valid": int(valid.sum()), "cfg": cfg, "body_in": args.body_in,
           "first_scored": f0}
    ztgt = None
    if args.depth_target:
        dt = np.load(os.path.join(args.depth_target, f"episode_{ep:06d}.npz"))
        zr = dt["z_root"][:T]
        zt = zr * np.clip(dt["z_target"][:T] / np.maximum(zr, 1e-3), 1 - cfg["mdepth_clip"], 1 + cfg["mdepth_clip"])
        ztgt = (zt, dt["valid_target"][:T].astype(bool) & valid)
        res["depth_target"] = dict(dir=args.depth_target, frames=int(ztgt[1].sum()))

    # stage A: depth for the fixed identity from the 2D skeleton scale
    kp0, j0 = forward_np(body, pose0)
    ratio = np.ones(T)
    for t in range(T):
        u = np.stack([K[0] * kp0[t, :, 0] / kp0[t, :, 2] + K[2], K[1] * kp0[t, :, 1] / kp0[t, :, 2] + K[3]], -1)
        s0 = np.linalg.norm(u[BODY70] - u[BODY70].mean(0), axis=1).mean()
        s1 = np.linalg.norm(ksam[t, BODY70] - ksam[t, BODY70].mean(0), axis=1).mean()
        ratio[t] = s0 / max(s1, 1e-6)
    init = pose0.copy()
    pel = j0[:, 1]
    init[:, :3] += (pel * ratio[:, None] - pel) * FLIP / 0.1
    res["identity_depth_ratio_median"] = float(np.median(ratio))

    # stage B: temporal fit without floor (identical to body_v2)
    stage_b = optimise(body, init, init, K, kv, ksam, wdata, cfg, iters=args.iters, log=log, tag="B", ztgt=ztgt)
    kpB, _ = forward_np(body, stage_b)

    # floor: MoGe normal, offset from the stage-B feet (identical to body_v2)
    planes = fit_floor_moge(ep, fx=K[0])
    best = None
    for (n, dm, frac) in planes:
        hs = [(kpB[:, SOLE_KP[s]] @ n).min(1) for s in range(2)]
        hlow = np.minimum(hs[0], hs[1])[valid]
        d_b = float(np.percentile(hlow, 10))
        spread = float(np.percentile(hlow, 50) - np.percentile(hlow, 5))
        cand = dict(n=n.tolist(), d_moge=dm, inlier_frac=frac, d_body=d_b, spread=spread)
        if best is None or (frac > 0.15 and spread < best["spread"]) or best["inlier_frac"] < 0.15:
            best = cand
    res["floor_candidates"] = [dict(n=list(map(float, p[0])), d=p[1], frac=p[2]) for p in planes]
    floor = None
    contact = np.zeros((T, 2), bool)
    wst = np.zeros(T)
    final = stage_b
    if best is not None:
        floor = (np.asarray(best["n"]), best["d_body"])
        res["floor"] = best
        res["floor_moge_over_body"] = best["d_moge"] / best["d_body"]
        contact, _ = detect_contacts(kpB, floor, cfg)
        st, wst, g = detect_standing(kpB, floor, contact, valid, cfg)
        htgt, nh = hip_target(g, st, cfg)
        res["standing_frac_B"] = float(st.mean())
        res["hip_target_B"] = htgt
        grav = (wst, htgt) if args.upright else None
        stage_c = optimise(body, stage_b, init, K, kv, ksam, wdata, cfg, floor=floor, contact=contact, grav=grav,
                           iters=args.iters, log=log, tag="C", ztgt=ztgt)
        kpC, _ = forward_np(body, stage_c)
        contact, _ = detect_contacts(kpC, floor, cfg)
        st, wst, g = detect_standing(kpC, floor, contact, valid, cfg)
        htgt, nh = hip_target(g, st, cfg)
        res["contact_frac"] = contact.mean(0).tolist()
        res["standing_frac"] = float(st.mean())
        res["standing_frac_scored"] = float(st[scm].mean()) if scm.any() else float("nan")
        res["first_scored_standing_weight"] = float(wst[f0])
        res["hip_target"] = htgt
        res["hip_target_frames"] = nh
        grav = (wst, htgt) if args.upright else None
        final = optimise(body, stage_c, init, K, kv, ksam, wdata, cfg, floor=floor, contact=contact, grav=grav,
                         iters=args.iters // 2, log=log, tag="D", ztgt=ztgt)
    kpF, jF = forward_np(body, final)
    res["v5"] = v2.metrics(kpF, jF, kv, valid, sc, st2d)
    if floor is not None:
        stF = wst > 0
        res["after"], gF = upright_stats(kpF, floor, stF, scm)
        res["before"], gB = upright_stats(kpB, floor, stF, scm)  # stage B (= body_v2 before the floor)
        res["first_scored_tilt_B"] = float(gB["tilt"][f0])
        res["first_scored_tilt"] = float(gF["tilt"][f0])
    if args.holdout:
        uv = np.stack([K[0] * kpF[..., 0] / kpF[..., 2] + K[2], K[1] * kpF[..., 1] / kpF[..., 2] + K[3]], -1)
        e = np.linalg.norm(uv[:, COCO_TO_70] - kv[..., :2], axis=-1)
        m = (kv[..., 2] > 0.5) & (valid & ho)[:, None]
        res["holdout_px"] = float(e[m].mean())
    res["time_s"] = time.time() - t0
    os.makedirs(os.path.join(args.out, "metrics"), exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out, f"episode_{ep:06d}.npz"),
        pose=final.astype(np.float32), scales=scales.astype(np.float32), shape=shape.astype(np.float32),
        valid=valid, contact=contact, floor_n=floor[0] if floor else np.zeros(3),
        floor_d=floor[1] if floor else 0.0, stationary_2d=st2d, standing=wst.astype(np.float32),
        pose_stage_b=stage_b.astype(np.float32), pose_init=init.astype(np.float32), holdout=ho)
    with open(os.path.join(args.out, "metrics", f"episode_{ep:06d}{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    keys = ["depth_dz_p99_cm", "depth_acc_cm", "reproj_vit_px", "skate_cm_per_frame", "acc22_cm"]
    log(f"ep {ep} v5 " + " ".join(f"{k}={res['v5'][k]:.3f}" for k in keys)
        + (f" holdout={res['holdout_px']:.3f}" if args.holdout else "")
        + (f" tilt {res['before'].get('tilt_med_all', float('nan')):.2f}->{res['after'].get('tilt_med_all', float('nan')):.2f}"
           f" first {res['first_scored_tilt_B']:.2f}->{res['first_scored_tilt']:.2f} (w {res['first_scored_standing_weight']:.2f})"
           f" standing {res['standing_frac']:.2f}" if floor is not None else "")
        + f" ({res['time_s']:.0f}s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--body-in", default=v2.BODY_DIR, help="per-frame HMR estimates (work/body format)")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--holdout", action="store_true", help="fit on even frames only (2D error on odd frames)")
    ap.add_argument("--upright", dest="upright", action="store_true",
                    help="enable the upright prior (off by default: it raised CD-H on Kaggle, submission S2)")
    ap.add_argument("--no-upright", dest="upright", action="store_false", help="body_v2 behaviour (default)")
    ap.add_argument("--depth-target", default="", help="compute_moge_depth_target.py output (pelvis depth target)")
    ap.add_argument("--cfg", default="", help="JSON overrides of DEFAULT_CFG")
    ap.add_argument("--tag", default="")
    ap.add_argument("--scored", default=f"{WORK}/body_v2/scored_frames.json")
    args = ap.parse_args()
    cfg = dict(DEFAULT_CFG)
    if args.cfg:
        cfg.update(json.loads(args.cfg))
    eps = list(range(30)) if args.all or not args.episodes else args.episodes
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        eps = eps[i::n]
    scored = v2.load_scored(args.scored)
    log = lambda s: print(s, flush=True)  # noqa: E731
    for ep in eps:
        run_episode(ep, args, cfg, scored, log)


if __name__ == "__main__":
    main()
