"""Whole-sequence body refinement, version 3 (extends refine_body_sequence.py).

Changes w.r.t. v2 (all optional through --cfg / flags, defaults = the v3 setting):
  - identity: shape / scales pooled over all episodes of the same performer (groups below, checked visually
    and by the spread of the per-episode identities) and over confident frames only (ViTPose and SAM 3D Body
    keypoints agree, whole body detected)
  - silhouette term against the person masks: projected vertices outside (person | object) are pulled in
    (distance transform), and the person-mask boundary is pulled onto the projected body (coverage)
  - optional MoGe term: pelvis depth follows the per-frame MoGe person depth normalised by the static
    background (scale-free, i.e. only the shape of the depth trajectory)
  - per-episode smoothness settings (--cfg-json mapping episode -> overrides)
Same output format as body_v2: <out>/episode_XXXXXX.npz (pose [T,136], scales [68], shape [45], valid,
contact, floor_n, floor_d, ...) and <out>/metrics/episode_XXXXXX.json (incl. floor_moge_over_body).

  python refine_body_v3.py --episodes 0 1 2 --out WORK/body_v3
  python refine_body_v3.py --all --shard 0/6 --out WORK/body_v3
  python refine_body_v3.py --episodes 2 --holdout --out WORK/body_v3/dev/ho_x --tag _x   # fit even frames only
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refine_body_sequence as v2  # noqa: E402
from common import unpack_masks  # noqa: E402
from mhr_utils import FLIP  # noqa: E402
from refine_body_sequence import (BODY70, COCO_TO_70, FOOT_KP, J22, SOLE_KP, Body, acc, detect_contacts,  # noqa: E402
                                  fill_invalid, fit_floor_moge, forward_np, gmof, load_episode, project_t,
                                  stationary_feet_2d)

ROOT = v2.ROOT
WORK = f"{ROOT}/work"
OUT_DIR = f"{WORK}/body_v3"

# Episodes showing the same performer (same clothes, same recording day; per-episode identities agree).
PERFORMERS = [[0, 1], [4, 7], [5, 8], [15, 25, 28], [14, 17, 20, 26], [16, 19, 23], [18, 21, 27], [22, 24, 29]]

DEFAULT_CFG = dict(v2.DEFAULT_CFG)
DEFAULT_CFG.update(
    identity="pooled",        # "median" (v2), "confident" (this episode, confident frames), "pooled"
    w_sil=0.0, w_sil_cov=1.0, sig_sil=6.0, sil_pts=200, sil_verts=4000, sil_batch=128, sil_dilate=2,
    w_moge=0.0, sig_moge=0.03,
    sig_acc_rel=0.016,        # v2: 0.008; the held-out-frame test favours less root-relative smoothing
)


# ------------------------------------------------------------------------------------- identity
def confident_frames(ep):
    """Frames whose SAM 3D Body keypoints agree with ViTPose and the whole body is detected."""
    b, T, valid, kv = load_episode(ep)
    k = b["kp2d"][:, COCO_TO_70]
    conf = kv[..., 2]
    e = np.linalg.norm(k - kv[..., :2], axis=-1)
    full = (conf > 0.5).all(1) & valid
    scale = np.maximum(b["bbox"][:, 3] - b["bbox"][:, 1], 1.0)
    err = np.where(conf > 0.5, e, np.nan)
    rel = np.nanmedian(err, 1) / scale
    ok = full & np.isfinite(rel)
    if ok.sum() < 30:
        return valid
    thr = np.percentile(rel[ok], 50)
    return ok & (rel <= thr)


def identity(ep, mode):
    b = np.load(f"{v2.BODY_DIR}/episode_{ep:06d}.npz")
    valid = b["valid"].astype(bool)
    if mode == "median":
        return np.median(b["shape"][valid], 0), np.median(b["scales"][valid], 0), [ep]
    group = [ep]
    if mode == "pooled":
        group = next((g for g in PERFORMERS if ep in g), [ep])
    S, C = [], []
    for e in group:
        be = np.load(f"{v2.BODY_DIR}/episode_{e:06d}.npz")
        m = confident_frames(e)
        S.append(be["shape"][m])
        C.append(be["scales"][m])
    return np.median(np.concatenate(S), 0), np.median(np.concatenate(C), 0), group


# ------------------------------------------------------------------------------------- silhouettes
def silhouette_data(ep, T, dilate, n_pts, rng):
    """Half-resolution distance transform outside (person | object) and person-boundary samples.
    Returns dt [T,h,w] float16 (pixels at half resolution), pts [T,n_pts,2] (full-res pixels), pw [T,n_pts]."""
    m = np.load(f"{WORK}/masks/episode_{ep:06d}.npz")
    h, w = int(m["mask_h"]), int(m["mask_w"])
    dt = np.zeros((T, h, w), np.float16)
    pts = np.zeros((T, n_pts, 2), np.float32)
    pw = np.zeros((T, n_pts), np.float32)
    k3 = np.ones((3, 3), np.uint8)
    for f in range(min(T, len(m["person_masks"]))):
        P = unpack_masks(m["person_masks"], h, w, f)
        if P.sum() < 100:
            continue
        O = unpack_masks(m["obj_masks"], h, w, f)
        A = binary_dilation(P | O, iterations=dilate) if dilate else (P | O)
        dt[f] = distance_transform_edt(~A).astype(np.float16)
        Pu = P.astype(np.uint8)
        edge = (Pu - cv2.erode(Pu, k3)) > 0
        # boundary pixels next to the object are occlusion edges, not body outline
        edge &= ~binary_dilation(O, iterations=2)
        ys, xs = np.nonzero(edge)
        if len(ys) < 10:
            continue
        sel = rng.choice(len(ys), n_pts, replace=len(ys) < n_pts)
        pts[f] = np.stack([xs[sel], ys[sel]], 1) * 2.0 + 0.5
        pw[f] = 1.0
    return dt, pts, pw


# ------------------------------------------------------------------------------------- MoGe depth
def moge_person_depth(ep, T):
    """Per-frame median MoGe depth on eroded person pixels divided by the frame's background scale."""
    d = np.load(f"{WORK}/depth/episode_{ep:06d}.npz")
    D = d["depth"]
    m = np.load(f"{WORK}/masks/episode_{ep:06d}.npz")
    mh, mw = int(m["mask_h"]), int(m["mask_w"])
    dh, dw = D.shape[1:]
    sy, sx = mh // dh, mw // dw
    bgs = []
    for f in np.linspace(0, len(D) - 1, 40).round().astype(int):
        pm = unpack_masks(m["person_masks"], mh, mw, f) | unpack_masks(m["obj_masks"], mh, mw, f)
        pm = binary_dilation(pm[::sy, ::sx][:dh, :dw], iterations=4)
        z = D[f].astype(np.float32)
        bgs.append(np.where(pm | ~(z > 0), np.nan, z))
    with np.errstate(all="ignore"):
        Zbg = np.nanmedian(np.stack(bgs), 0)
    out = np.full(T, np.nan)
    for f in range(min(T, len(D))):
        P = unpack_masks(m["person_masks"], mh, mw, f)
        O = unpack_masks(m["obj_masks"], mh, mw, f)
        pe = cv2.erode(P.astype(np.uint8), np.ones((7, 7), np.uint8))[::sy, ::sx][:dh, :dw] > 0
        z = D[f].astype(np.float32)
        bg = np.isfinite(Zbg) & (z > 0) & ~binary_dilation((P | O)[::sy, ::sx][:dh, :dw], iterations=4)
        if pe.sum() > 30 and bg.sum() > 500:
            out[f] = np.median(z[pe & (z > 0)]) / np.median(z[bg] / Zbg[bg])
    return out


# ------------------------------------------------------------------------------------- fitting
def optimise(body, init, prior, K, kv, ksam, wdata, cfg, floor=None, contact=None, iters=600, lr=0.01,
             log=None, tag="", sil=None, moge=None, seed=0):
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
    samw = torch.ones(70, device=dev) * cfg["w_sam"]
    samw[15:21] = cfg["w_sam_feet"]
    samw[21:62] = cfg["w_sam"] * 0.3
    gen = torch.Generator(device="cpu").manual_seed(seed)
    use_sil = sil is not None and cfg["w_sil"] > 0
    if use_sil:
        dt, spts, spw = sil
        h, w = dt.shape[1:]
        vsub = torch.randperm(body.nv, generator=gen)[: cfg["sil_verts"]].to(dev)
        sil_frames = torch.nonzero((spw[:, 0] > 0) & (wd > 0)).flatten()
    use_moge = moge is not None and cfg["w_moge"] > 0
    if use_moge:
        mz = tt(np.nan_to_num(moge, nan=1.0))
        mok = tt(np.isfinite(moge)) * wd
    for it in range(iters):
        opt.zero_grad()
        kp, j, verts = body(x, want_verts=True)
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
        L_sil = torch.zeros((), device=dev)
        if use_sil:
            nb = min(cfg["sil_batch"], len(sil_frames))
            fb = sil_frames[torch.randperm(len(sil_frames), generator=gen)[:nb].to(dev)]
            vb = verts[fb][:, vsub]
            z = vb[..., 2].clamp(min=0.2)
            u = Kt[0] * vb[..., 0] / z + Kt[2]
            v = Kt[1] * vb[..., 1] / z + Kt[3]
            # outside term: distance transform at the projected vertices (grid_sample, align_corners=False)
            g = torch.stack([(u / 2) / w * 2 - 1, (v / 2) / h * 2 - 1], -1)[:, None]
            d_out = torch.nn.functional.grid_sample(dt[fb][:, None].float(), g, mode="bilinear",
                                                    padding_mode="border", align_corners=False)[:, 0, 0] * 2
            L_out = gmof((d_out / cfg["sig_sil"]).square(), 9.0).mean(1)
            # coverage term: person boundary samples to the nearest projected vertex
            uvb = torch.stack([u, v], -1)
            dmin = torch.cdist(spts[fb], uvb).min(-1).values
            L_cov = (gmof((dmin / cfg["sig_sil"]).square(), 9.0) * spw[fb]).mean(1)
            L_sil = cfg["w_sil"] * 17.0 * ((L_out + cfg["w_sil_cov"] * L_cov) * wd[fb]).sum() / nb
            loss = loss + L_sil
        L_m = torch.zeros((), device=dev)
        if use_moge:
            lz = torch.log(root[:, 2].clamp(min=0.3)) - torch.log(mz)
            off = ((lz * mok).sum() / mok.sum().clamp(min=1)).detach()
            L_m = cfg["w_moge"] * (mok * ((lz - off) / cfg["sig_moge"]).square()).sum() / T
            loss = loss + L_m
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
        loss.backward()
        opt.step()
        sched.step()
        if log is not None and (it % 200 == 0 or it == iters - 1):
            log(f"  [{tag}] it {it:4d} loss {loss.item():9.3f} vit {L_vit.sum().item() / T:7.3f} "
                f"sam {L_sam.sum().item() / T:7.3f} prior {L_pr.sum().item() / T:7.3f} "
                f"smooth {cfg['w_smooth'] * L_s.item() / T:7.3f} sil {L_sil.item():7.3f} moge {L_m.item():7.3f} "
                f"floor {L_f.item():7.3f}")
    return x.detach().cpu().numpy().astype(np.float64)


def run_episode(ep, args, cfg, scored, log):
    t0 = time.time()
    b, T, valid, kv = load_episode(ep)
    K = np.array([float(b["fx"]), float(b["fy"]), float(b["cx"]), float(b["cy"])])
    v2.METRIC_K = K
    dev = torch.device("cuda")
    shape, scales, group = identity(ep, cfg["identity"])
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
    st2d = stationary_feet_2d(kv, ksam, valid)
    res = {"episode": ep, "T": T, "n_valid": int(valid.sum()), "cfg": cfg, "identity_group": group}

    sil = None
    if cfg["w_sil"] > 0:
        dt, spts, spw = silhouette_data(ep, T, cfg["sil_dilate"], cfg["sil_pts"], np.random.default_rng(ep))
        sil = (torch.as_tensor(dt, device=dev), torch.as_tensor(spts, device=dev), torch.as_tensor(spw, device=dev))
    moge = moge_person_depth(ep, T) if cfg["w_moge"] > 0 else None

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

    kw = dict(sil=sil, moge=moge, log=log)
    stage_b = optimise(body, init, init, K, kv, ksam, wdata, cfg, iters=args.iters, tag="B", **kw)
    kpB, jB = forward_np(body, stage_b)
    floor = None
    contact = np.zeros((T, 2), bool)
    final = stage_b
    planes = fit_floor_moge(ep)
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
    if best is not None:
        floor = (np.asarray(best["n"]), best["d_body"])
        res["floor"] = best
        res["floor_moge_over_body"] = best["d_moge"] / best["d_body"]
        contact, _ = detect_contacts(kpB, floor, cfg)
        stage_c = optimise(body, stage_b, init, K, kv, ksam, wdata, cfg, floor=floor, contact=contact,
                           iters=args.iters, tag="C", **kw)
        kpC, _ = forward_np(body, stage_c)
        contact, _ = detect_contacts(kpC, floor, cfg)
        res["contact_frac"] = contact.mean(0).tolist()
        final = optimise(body, stage_c, init, K, kv, ksam, wdata, cfg, floor=floor, contact=contact,
                         iters=args.iters // 2, tag="D", **kw)
    kpF, jF = forward_np(body, final)
    res["v3"] = v2.metrics(kpF, jF, kv, valid, sc, st2d)
    if args.holdout:
        uv = np.stack([K[0] * kpF[..., 0] / kpF[..., 2] + K[2], K[1] * kpF[..., 1] / kpF[..., 2] + K[3]], -1)
        e = np.linalg.norm(uv[:, COCO_TO_70] - kv[..., :2], axis=-1)
        m = (kv[..., 2] > 0.5) & (valid & ho)[:, None]
        res["holdout_v3"] = float(e[m].mean())
    res["time_s"] = time.time() - t0
    os.makedirs(os.path.join(args.out, "metrics"), exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out, f"episode_{ep:06d}.npz"),
        pose=final.astype(np.float32), scales=scales.astype(np.float32), shape=shape.astype(np.float32),
        valid=valid, contact=contact, floor_n=floor[0] if floor else np.zeros(3),
        floor_d=floor[1] if floor else 0.0, stationary_2d=st2d,
        pose_stage_b=stage_b.astype(np.float32), pose_init=init.astype(np.float32), holdout=ho)
    with open(os.path.join(args.out, "metrics", f"episode_{ep:06d}{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    keys = ["depth_dz_p99_cm", "depth_acc_cm", "reproj_vit_px", "skate_cm_per_frame", "acc22_cm"]
    log(f"ep {ep} v3 " + " ".join(f"{k}={res['v3'][k]:.3f}" for k in keys)
        + (f" holdout={res['holdout_v3']:.3f}" if args.holdout else "") + f" ({res['time_s']:.0f}s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--holdout", action="store_true", help="fit on even frames only (2D error on odd frames)")
    ap.add_argument("--cfg", default="", help="JSON overrides of DEFAULT_CFG")
    ap.add_argument("--cfg-json", default="", help="JSON file {episode: overrides} applied after --cfg")
    ap.add_argument("--tag", default="")
    ap.add_argument("--scored", default=f"{WORK}/body_v2/scored_frames.json")
    args = ap.parse_args()
    cfg = dict(DEFAULT_CFG)
    if args.cfg:
        cfg.update(json.loads(args.cfg))
    per_ep = json.load(open(args.cfg_json)) if args.cfg_json else {}
    eps = list(range(30)) if args.all or not args.episodes else args.episodes
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        eps = eps[i::n]
    scored = v2.load_scored(args.scored)
    log = lambda s: print(s, flush=True)  # noqa: E731
    for ep in eps:
        c = dict(cfg)
        c.update(per_ep.get(str(ep), {}))
        run_episode(ep, args, c, scored, log)


if __name__ == "__main__":
    main()
