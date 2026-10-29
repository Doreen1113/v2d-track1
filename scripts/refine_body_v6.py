"""Whole-sequence body refinement, version 6: the body_vA setting (refine_body_v5.py --no-upright) plus Sapiens2
308-keypoint 2D targets (run_sapiens_pose.py -> work/sap_kp).

The MHR model is evaluated at all 308 keypoints of SAM 3D Body's keypoint regressor (rows 0:70 are the MHR70 points
used before; rows 70:308 are the face points, in the same Sociopticon / Sapiens 308 order). Every stage (B, C, D)
of refine_body_v5 is kept with the same weights; the extra robust 2D term is
    sum_k  w_group(k) * c_tk * gmof(|proj_k - sap_tk|^2 / sig_group^2, sap_rho^2)
with c_tk the Sapiens heat-map score (zero below sap_thr). Groups (MHR70 / Sapiens indices):
    body  0:15    nose, eyes, ears, shoulders, elbows, hips, knees, ankles   (w_sap_body, sig_sap_body)
    feet  15:21   big toe, small toe, heel                                   (w_sap_feet, sig_sap_feet)
    wrist 41, 62                                                            (w_sap_wrist, sig_sap_hand)
    hand  21:41, 42:62  finger joints and tips                              (w_sap_hand, sig_sap_hand)
    extra 63:70   olecranon, cubital fossa, acromion, neck                   (w_sap_extra, sig_sap_body)
    face  70:308                                                            (w_sap_face, sig_sap_face)
With sap_hand_rel / sap_feet_rel the finger / foot points are compared relative to the wrist / ankle of the same
side (model minus its own wrist vs Sapiens minus the Sapiens wrist), so they shape the hand / foot without pulling
the arm or leg; the anchor's score gates the point.
A hand is dropped on a frame when the Sapiens wrist is more than hand_gate px from the ViTPose wrist of the same
side (ViTPose score > 0.5), or when its mean finger score is below hand_min_score.
With w_sap_* = 0 the loss is exactly refine_body_v5's without the upright / depth terms (the body_vA setting).
Optional: w_vit scales the ViTPose term (0 = Sapiens body replaces ViTPose), w_sam_feet / w_sam_hand_scale the
SAM 3D Body self-targets on the feet / hands.

Output as refine_body_v5 (pose [T,136], scales, shape, valid, contact, floor, holdout ...), plus in the metrics
json the held-out (or all-frame) 2D errors against ViTPose and against each Sapiens group.

  python refine_body_v6.py --all --body-in WORK/body_ens2 --out WORK/sap_body                     # full run
The intrinsics come from the --body-in npz (fx, fy, cx, cy; also used for the MoGe floor) and are copied to the
output npz, so the variant can be re-run unchanged on a per-frame input built at another focal; work/sap_kp is in
image pixels and does not depend on the focal.
  python refine_body_v6.py --episodes 0 3 --holdout --body-in WORK/body_ens2 --out DEV/ho_x --cfg '{"w_sap_feet":1}'
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refine_body_sequence as v2  # noqa: E402
import refine_body_v5 as v5  # noqa: E402
from mhr_utils import FLIP  # noqa: E402
from refine_body_sequence import (BODY70, COCO_TO_70, FOOT_KP, J22, SOLE_KP, Body, acc, detect_contacts,  # noqa: E402
                                  fill_invalid, fit_floor_moge, forward_np, gmof, project_t, stationary_feet_2d)

WORK = f"{v2.ROOT}/work"
SAP_DIR = f"{WORK}/sap_kp"
KP308 = f"{WORK}/sap_kp/kp308_mapping.npz"  # SAM 3D Body head_pose.keypoint_mapping, all 308 rows

GROUPS = dict(
    body=list(range(0, 15)),
    feet=list(range(15, 21)),
    wrist=[41, 62],
    hand=list(range(21, 41)) + list(range(42, 62)),
    extra=list(range(63, 70)),
    face=list(range(70, 308)),
)
HAND_R, HAND_L = list(range(21, 41)), list(range(42, 62))
VIT_WRIST = {41: 10, 62: 9}  # MHR70 wrist -> COCO-17 wrist (right 10, left 9)
# anchor of the relative terms: finger points -> wrist of the same hand, toes / heel -> ankle of the same foot
ANCHOR = {**{k: 41 for k in HAND_R}, **{k: 62 for k in HAND_L}, 15: 13, 16: 13, 17: 13, 18: 14, 19: 14, 20: 14}


def anchors(cfg):
    """Per-keypoint anchor index [308]; 308 (a zero point appended to the arrays) = absolute position."""
    a = np.full(308, 308)
    for k, v in ANCHOR.items():
        if (k in GROUPS["hand"] and cfg["sap_hand_rel"]) or (k in GROUPS["feet"] and cfg["sap_feet_rel"]):
            a[k] = v
    return a

DEFAULT_CFG = dict(v5.DEFAULT_CFG)
DEFAULT_CFG.update(
    w_vit=1.0, w_sam_hand_scale=0.3,
    sap_thr=0.3, sap_rho=3.0,
    w_sap_body=0.0, w_sap_feet=0.0, w_sap_wrist=0.0, w_sap_hand=0.0, w_sap_extra=0.0, w_sap_face=0.0,
    sig_sap_body=5.0, sig_sap_feet=5.0, sig_sap_hand=4.0, sig_sap_face=4.0,
    hand_gate=30.0, hand_min_score=0.3,
    sap_hand_rel=0, sap_feet_rel=0,  # 1: hand / foot points relative to the Sapiens wrist / ankle (shape, not position)
    sap_drop=[],  # Sapiens indices left out of the fit (e.g. [3, 4]: ears, whose annotation differs from COCO's)
)


class Body308(Body):
    """refine_body_sequence.Body with all 308 keypoint rows (rows 0:70 identical to KP_MAP)."""

    def __init__(self, device, shape, scales, kp_map=KP308):
        super().__init__(device, shape, scales)
        M = sp.load_npz(kp_map).tocsr()
        nnz = np.diff(M.indptr).max()
        n = M.shape[0]
        idx = np.zeros((n, nnz), np.int64)
        wt = np.zeros((n, nnz), np.float32)
        for r in range(n):
            a, b = M.indptr[r], M.indptr[r + 1]
            idx[r, : b - a] = M.indices[a:b]
            wt[r, : b - a] = M.data[a:b]
        self.kidx = torch.tensor(idx, device=device)
        self.kw = torch.tensor(wt, device=device)


# ------------------------------------------------------------------------------------- Sapiens targets
def load_sapiens(ep, T, sap_dir=SAP_DIR):
    d = np.load(f"{sap_dir}/episode_{ep:06d}.npz")
    kp = d["kp"].astype(np.float64)
    sc = d["score"].astype(np.float64)
    if len(kp) != T:
        raise ValueError(f"ep {ep}: Sapiens length {len(kp)} != {T}")
    bad = ~np.isfinite(kp).all(-1)
    kp[bad] = 0
    sc[bad] = 0
    return kp, sc


def sapiens_weights(sap_kp, sap_sc, kv, valid, cfg):
    """Per-frame, per-keypoint weight [T,308] (group weight x score, gated) and the per-keypoint sigma [308]."""
    T = len(sap_kp)
    w = np.zeros(308)
    sig = np.zeros(308)
    for g, ids in GROUPS.items():
        w[ids] = cfg[f"w_sap_{g}"]
    w[list(cfg["sap_drop"])] = 0
    sig[GROUPS["body"] + GROUPS["extra"]] = cfg["sig_sap_body"]
    sig[GROUPS["feet"]] = cfg["sig_sap_feet"]
    sig[GROUPS["hand"] + GROUPS["wrist"]] = cfg["sig_sap_hand"]
    sig[GROUPS["face"]] = cfg["sig_sap_face"]
    c = np.where(sap_sc > cfg["sap_thr"], sap_sc, 0.0)
    c[~valid] = 0
    # hand gating: wrist far from the ViTPose wrist of the same side, or a weak hand
    gate = np.ones((T, 2), bool)
    for s, (wr, fingers) in enumerate(((41, HAND_R), (62, HAND_L))):
        vw = kv[:, VIT_WRIST[wr]]
        d = np.linalg.norm(sap_kp[:, wr] - vw[:, :2], axis=1)
        far = (vw[:, 2] > 0.5) & (d > cfg["hand_gate"])
        weak = sap_sc[:, fingers].mean(1) < cfg["hand_min_score"]
        gate[:, s] = ~(far | weak)
        c[np.ix_(~gate[:, s], fingers + [wr])] = 0
    anc = anchors(cfg)
    rel = anc < 308
    c[:, rel] = c[:, rel] * (c[:, anc[rel]] > 0)  # relative points need a confident anchor
    return c * w[None], sig, gate


# ------------------------------------------------------------------------------------- fitting
def optimise(body, init, prior, K, kv, ksam, wdata, cfg, floor=None, contact=None, sap=None, iters=600, lr=0.01,
             log=None, tag=""):
    """refine_body_v5.optimise (no upright / depth terms) plus the Sapiens term; sap = (kp [T,308,2], w [T,308],
    sig [308]) or None."""
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
    samw[21:62] = cfg["w_sam"] * cfg["w_sam_hand_scale"]
    use_sap = sap is not None and float(np.abs(sap[1]).sum()) > 0
    if use_sap:
        spk, spw, sps = tt(sap[0]), tt(sap[1]), tt(sap[2])
        anc = torch.as_tensor(anchors(cfg), device=dev)
        spk_rel = spk - torch.cat([spk, torch.zeros_like(spk[:, :1])], 1)[:, anc]
        sps = torch.where(sps > 0, sps, torch.ones_like(sps))
        rho2 = cfg["sap_rho"] ** 2
    for it in range(iters):
        opt.zero_grad()
        kp, j = body(x)
        uv = project_t(kp, Kt)
        r = (uv[:, cidx] - kvt[..., :2]) / cfg["sig_vit"]
        c = kvt[..., 2] * (kvt[..., 2] > cfg["vit_thr"])
        L_vit = cfg["w_vit"] * (gmof(r.square().sum(-1), 9.0) * c).sum(1) * wd
        r = (uv[:, :70] - kst) / cfg["sig_sam"]
        L_sam = (gmof(r.square().sum(-1), 9.0) * samw).sum(1) * wd
        L_pr = (((x - pr) / sig_th).square().sum(1) + ((x[:, 2] - pr[:, 2]) * 0.1 / cfg["sig_depth"]).square()) * wd
        loss = (L_vit.sum() + L_sam.sum() + L_pr.sum()) / T
        L_sp = torch.zeros(T, device=dev)
        if use_sap:
            uv_rel = uv - torch.cat([uv, torch.zeros_like(uv[:, :1])], 1)[:, anc]
            r = (uv_rel - spk_rel) / sps[None, :, None]
            L_sp = (gmof(r.square().sum(-1), rho2) * spw).sum(1) * wd
            loss = loss + L_sp.sum() / T
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
        loss.backward()
        opt.step()
        sched.step()
        if log is not None and (it % 200 == 0 or it == iters - 1):
            log(f"  [{tag}] it {it:4d} loss {loss.item():9.3f} vit {L_vit.sum().item() / T:7.3f} "
                f"sam {L_sam.sum().item() / T:7.3f} sap {L_sp.sum().item() / T:7.3f} prior {L_pr.sum().item() / T:7.3f} "
                f"smooth {cfg['w_smooth'] * L_s.item() / T:7.3f} floor {L_f.item():7.3f}")
    return x.detach().cpu().numpy().astype(np.float64)


# ------------------------------------------------------------------------------------- 2D errors
def errors_2d(kp, K, kv, sap_kp, sap_sc, frames, thr=0.5):
    """Mean 2D error (px) on the given frames: ViTPose COCO-17 (score > thr) and each Sapiens group (score > thr;
    hands only where both the projected and the Sapiens wrist are inside the frame)."""
    uv = np.stack([K[0] * kp[..., 0] / kp[..., 2] + K[2], K[1] * kp[..., 1] / kp[..., 2] + K[3]], -1)
    out = {}
    e = np.linalg.norm(uv[:, COCO_TO_70] - kv[..., :2], axis=-1)
    m = (kv[..., 2] > thr) & frames[:, None]
    out["vit_px"] = float(e[m].mean()) if m.any() else float("nan")
    es = np.linalg.norm(uv - sap_kp, axis=-1)
    for g, ids in GROUPS.items():
        m = (sap_sc[:, ids] > thr) & frames[:, None]
        out[f"sap_{g}_px"] = float(es[:, ids][m].mean()) if m.any() else float("nan")
        out[f"sap_{g}_n"] = int(m.sum())
    return out


# ------------------------------------------------------------------------------------- main
def run_episode(ep, args, cfg, scored, log):
    t0 = time.time()
    b, T, valid, kv = v5.load_episode(ep, args.body_in)
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
    body = Body308(dev, shape, scales)
    sc = scored.get(ep, [])
    scm = np.zeros(T, bool)
    scm[[f for f in sc if f < T]] = True
    f0 = sc[0] if sc else int(np.flatnonzero(valid)[0])
    st2d = stationary_feet_2d(kv, ksam, valid)
    sap_kp, sap_sc = load_sapiens(ep, T, args.sap_dir)
    spw, sps, gate = sapiens_weights(sap_kp, sap_sc, kv, valid, cfg)
    sap = (sap_kp, spw, sps)
    res = {"episode": ep, "T": T, "n_valid": int(valid.sum()), "cfg": cfg, "body_in": args.body_in,
           "sap_dir": args.sap_dir, "first_scored": f0, "hand_gate_frac": gate.mean(0).tolist()}

    # stage A: depth for the fixed identity from the 2D skeleton scale (as refine_body_v5)
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

    # stage B: temporal fit without floor
    stage_b = optimise(body, init, init, K, kv, ksam, wdata, cfg, sap=sap, iters=args.iters, log=log, tag="B")
    kpB, _ = forward_np(body, stage_b)

    # floor: MoGe normal, offset from the stage-B feet (as refine_body_v5)
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
        stage_c = optimise(body, stage_b, init, K, kv, ksam, wdata, cfg, floor=floor, contact=contact, sap=sap,
                           iters=args.iters, log=log, tag="C")
        kpC, _ = forward_np(body, stage_c)
        contact, _ = detect_contacts(kpC, floor, cfg)
        st, wst, g = v5.detect_standing(kpC, floor, contact, valid, cfg)
        res["contact_frac"] = contact.mean(0).tolist()
        res["standing_frac"] = float(st.mean())
        final = optimise(body, stage_c, init, K, kv, ksam, wdata, cfg, floor=floor, contact=contact, sap=sap,
                         iters=args.iters // 2, log=log, tag="D")
    kpF, jF = forward_np(body, final)
    res["v6"] = v2.metrics(kpF[:, :70], jF, kv, valid, sc, st2d)
    ev = (valid & ho) if args.holdout else valid
    res["err_eval_frames"] = errors_2d(kpF, K, kv, sap_kp, sap_sc, ev)
    res["err_eval_frames_scored"] = errors_2d(kpF, K, kv, sap_kp, sap_sc, ev & scm)
    if args.holdout:
        res["holdout_px"] = res["err_eval_frames"]["vit_px"]
    res["time_s"] = time.time() - t0
    os.makedirs(os.path.join(args.out, "metrics"), exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out, f"episode_{ep:06d}.npz"),
        pose=final.astype(np.float32), scales=scales.astype(np.float32), shape=shape.astype(np.float32),
        valid=valid, contact=contact, floor_n=floor[0] if floor else np.zeros(3),
        floor_d=floor[1] if floor else 0.0, stationary_2d=st2d, standing=wst.astype(np.float32),
        pose_stage_b=stage_b.astype(np.float32), pose_init=init.astype(np.float32), holdout=ho,
        fx=K[0], fy=K[1], cx=K[2], cy=K[3])  # intrinsics of the body input (focal studies re-run with --body-in)
    with open(os.path.join(args.out, "metrics", f"episode_{ep:06d}{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    e = res["err_eval_frames"]
    log(f"ep {ep} v6 vit {e['vit_px']:.3f} sap body {e['sap_body_px']:.3f} feet {e['sap_feet_px']:.3f} "
        f"wrist {e['sap_wrist_px']:.3f} hand {e['sap_hand_px']:.3f} extra {e['sap_extra_px']:.3f} "
        f"face {e['sap_face_px']:.3f} ({'holdout' if args.holdout else 'all'}) ({res['time_s']:.0f}s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--body-in", default=f"{WORK}/body_ens2", help="per-frame HMR estimates (work/body format)")
    ap.add_argument("--sap-dir", default=SAP_DIR)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--holdout", action="store_true", help="fit on even frames only (2D error on odd frames)")
    ap.add_argument("--cfg", default="", help="JSON overrides of DEFAULT_CFG")
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-existing", action="store_true")
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
        if args.skip_existing and os.path.exists(os.path.join(args.out, "metrics", f"episode_{ep:06d}{args.tag}.json")):
            continue
        run_episode(ep, args, cfg, scored, log)


if __name__ == "__main__":
    main()
