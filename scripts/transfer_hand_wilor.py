"""Transfer WiLoR hands (run_wilor.py) onto a refined body run as MHR finger parameters (and optionally the wrist).

Per frame and side:
  1. Assignment. The body's MHR hand joints are projected; detections are matched to the two body hands
     (Hungarian on the distance between WiLoR's own 2D hand centre, regressed as that side, and the body's
     projected hand centre, relative to the projected hand length). The side always comes from the body.
  2. Gate (else the body's hand is kept): matched within --max-dist hand lengths (centre and wrist), detector
     score >= --min-conf, detector class agrees with the body side, enough of WiLoR's 2D hand on the person mask
     (--min-visible; hands behind the object are skipped), and the regressed finger configuration agrees with its
     temporal neighbours (median bone direction over +-4 accepted frames, --max-spike degrees).
  3. Target. WiLoR's 21 joints are rotated from the crop camera to the camera frame (rotation that takes the
     optical axis to the ray through the box centre) and made wrist-relative.
       --wrist none : the target palm (wrist + 4 MCPs) is Kabsch-aligned to the body's palm, so only the finger
                      articulation is transferred (hand orientation stays the body's)
       --wrist flex / twist : the target stays in the camera frame and the wrist flexion / deviation (and twist)
                      parameters are fitted as well, so the palm takes WiLoR's orientation; the wrist joint
                      position and the arm do not move
  4. Fit. All frames of an episode at once: sum of (1 - cos) between MHR and target bone directions (MHR70
     hand joints, the same joints as eval_hand_proxy.py), a quadratic pull to the body's parameters, a
     first-difference smoothness term on the parameter change and the MHR parameter limits. Frames without an
     accepted target only see the pull and the smoothness, so changes ramp in and out.

Output: body npz unchanged except the fitted pose columns (format of the input, for assemble_episode.py --body),
metrics/ copied, and <out>/diag/episode_XXXXXX.json with the gate statistics and fit residuals.

  python transfer_hand_wilor.py --body WORK/body_vA --wilor WORK/hand_wilor --out WORK/hand_vA [--wrist none]
  python transfer_hand_wilor.py --body WORK/body_vA_fpc --raw-dir WORK/body_fpc ...   (other intrinsics)
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import unpack_masks  # noqa: E402
from mhr_utils import DEFAULT_MHR, FLIP  # noqa: E402

WORK = "/work/doreen071/v2d/work"
# MHR joints in OpenPose hand order (wrist; thumb CMC, MCP, IP, tip; index MCP, PIP, DIP, tip; middle; ring; little)
HAND_J = {1: [42, 61, 62, 63, 64, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51, 44, 45, 46, 47]}
HAND_J[0] = [j + 36 for j in HAND_J[1]]
PARENT = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
PALM = [0, 5, 9, 13, 17]
FINGER_COLS = {1: list(range(68, 95)), 0: list(range(95, 122))}
WRIST_COLS = {"flex": {1: [38, 39], 0: [48, 49]}, "twist": {1: [37, 38, 39], 0: [47, 48, 49]}}


def bone_weights(wrist_mode):
    w = np.ones(21)
    w[0] = 0.0
    w[1] = 0.5  # wrist -> thumb CMC (driven by the thumb base joint in MHR, fixed in MANO)
    if wrist_mode == "none":
        w[[5, 9, 13, 17]] = 0.0  # palm bones: no finger parameter moves them
    return w


def kabsch(src, dst, w=None):
    """Rotation R minimising sum |R src_i - dst_i|^2 (points already centred), batched [...,N,3]."""
    H = np.einsum("...ni,...nj->...ij", src, dst)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("...ij,...jk->...ik", U, Vt)))
    D = np.zeros(H.shape)
    D[..., 0, 0] = 1
    D[..., 1, 1] = 1
    D[..., 2, 2] = d
    return np.einsum("...ji,...jk,...lk->...il", Vt, D, U)


def ray_rotation(u, v, K):
    """Rotation taking the optical axis (0,0,1) to the ray through pixel (u, v)."""
    d = np.stack([(u - K[2]) / K[0], (v - K[3]) / K[1], np.ones_like(u)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    z = np.array([0.0, 0.0, 1.0])
    ax = np.cross(z, d)
    s = np.linalg.norm(ax, axis=-1, keepdims=True)
    c = d[..., 2:3]
    ax = ax / np.maximum(s, 1e-12)
    Kx = np.zeros(d.shape[:-1] + (3, 3))
    Kx[..., 0, 1], Kx[..., 0, 2] = -ax[..., 2], ax[..., 1]
    Kx[..., 1, 0], Kx[..., 1, 2] = ax[..., 2], -ax[..., 0]
    Kx[..., 2, 0], Kx[..., 2, 1] = -ax[..., 1], ax[..., 0]
    I = np.broadcast_to(np.eye(3), Kx.shape)
    return I + s[..., None] * Kx + (1 - c[..., None]) * (Kx @ Kx)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def bones(kp):
    """Unit bone directions [..., 21, 3] (index 0 unused)."""
    b = kp - kp[..., PARENT.clip(0), :]
    return unit(b)


class Model:
    def __init__(self, device):
        self.m = torch.jit.load(DEFAULT_MHR, map_location=device).eval()
        self.dev = device
        self.flip = torch.tensor(FLIP, dtype=torch.float32, device=device)
        b = {k: v.cpu().numpy() for k, v in self.m.named_buffers() if v.numel() < 1e6}
        idx = b["character_torch.parameter_limits.minmax_parameter_index"]
        self.lo = np.full(136, -np.inf)
        self.hi = np.full(136, np.inf)
        for i, p in enumerate(idx):
            if p < 136:
                self.lo[p] = b["character_torch.parameter_limits.minmax_min"][i]
                self.hi[p] = b["character_torch.parameter_limits.minmax_max"][i]

    def joints(self, pose, scales, shape):
        """pose [T,136] tensor -> joints [T,127,3] metres, Track 1 frame (differentiable)."""
        T = len(pose)
        p = torch.cat([pose, scales.expand(T, -1)], 1)
        expr = torch.zeros(T, 72, device=self.dev)
        _, sk = self.m(shape.expand(T, -1).contiguous(), p, expr, True)
        return sk[..., :3] * self.flip / 100.0


def project(x, K):
    return np.stack([K[0] * x[..., 0] / x[..., 2] + K[2], K[1] * x[..., 1] / x[..., 2] + K[3]], -1)


def gate_and_targets(ep, wl, J0, K, a, valid):
    """Assignment + gates. Returns target joints [T,2,21,3] (camera frame, wrist-relative), weights [T,2], the
    matched detection per (frame, side) and per-match diagnostics, stats."""
    T = len(J0)
    kp_body = np.stack([J0[:, HAND_J[s]] for s in (0, 1)], 1)  # [T,2,21,3]
    uv_body = project(kp_body, K)
    ext = np.linalg.norm(uv_body - uv_body[:, :, :1], axis=-1).max(-1).clip(15, None)  # hand length px
    ctr_body = uv_body.mean(2)
    tgt = np.zeros((T, 2, 21, 3))
    tgt2d = np.zeros((T, 2, 21, 2))
    w = np.zeros((T, 2))
    det_i = -np.ones((T, 2), int)
    dg = {k: np.full((T, 2), np.nan) for k in ("conf", "cls_agree", "visible", "palm_deg", "dist")}
    st = dict(frames=int(valid.sum()), matched=[0, 0], low_conf=[0, 0], cls_disagree=[0, 0], occluded=[0, 0],
              palm_flip=[0, 0], spike=[0, 0], accepted=[0, 0])
    n = len(wl["det_frame"])
    if n == 0:
        return tgt, tgt2d, ext, w, det_i, dg, st
    fr = wl["det_frame"].astype(int)
    mk = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
    mh, mw = int(mk["mask_h"]), int(mk["mask_w"])
    sy, sx = mh / float(wl["height"]), mw / float(wl["width"])
    from scipy.ndimage import binary_dilation
    for t in np.unique(fr):
        if t >= T or not valid[t]:
            continue
        ids = np.flatnonzero(fr == t)
        C = np.zeros((2, len(ids)))
        for s in (0, 1):
            c_w = wl["kp2d"][ids, s].mean(1)
            C[s] = np.linalg.norm(c_w - ctr_body[t, s], axis=-1) / ext[t, s]
        rows, cols = linear_sum_assignment(C)
        P = None
        for s, c in zip(rows, cols):
            d = ids[c]
            wr = np.linalg.norm(wl["kp2d"][d, s, 0] - uv_body[t, s, 0]) / ext[t, s]
            if C[s, c] > a.max_dist or wr > a.max_dist:
                continue
            st["matched"][s] += 1
            if P is None:
                P = binary_dilation(unpack_masks(mk["person_masks"], mh, mw, t), iterations=3)
            q = wl["kp2d"][d, s] * np.array([sx, sy])
            qi = np.clip(q.round().astype(int), 0, [mw - 1, mh - 1])
            box = wl["det_box"][d]
            R = ray_rotation(np.array((box[0] + box[2]) / 2), np.array((box[1] + box[3]) / 2), K)
            x = wl["kp3d"][d, s].astype(np.float64) @ R.T
            x = x - x[0]
            pb = kp_body[t, s][PALM] - kp_body[t, s][PALM].mean(0)
            pw = x[PALM] - x[PALM].mean(0)
            Rp = kabsch(pw, pb)
            dg["palm_deg"][t, s] = np.degrees(np.arccos(np.clip((np.trace(Rp) - 1) / 2, -1, 1)))
            dg["conf"][t, s] = wl["det_conf"][d]
            dg["cls_agree"][t, s] = float(int(wl["det_cls"][d]) == s)
            dg["visible"][t, s] = P[qi[:, 1], qi[:, 0]].mean()
            dg["dist"][t, s] = max(C[s, c], wr)
            det_i[t, s] = d
            tgt[t, s] = x
            tgt2d[t, s] = wl["kp2d"][d, s]
            if dg["conf"][t, s] < a.min_conf:
                st["low_conf"][s] += 1
                continue
            if a.require_cls and not dg["cls_agree"][t, s]:
                st["cls_disagree"][s] += 1
                continue
            if dg["visible"][t, s] < a.min_visible:
                st["occluded"][s] += 1
                continue
            if dg["palm_deg"][t, s] > a.max_palm:
                st["palm_flip"][s] += 1
                continue
            w[t, s] = 1.0
    # temporal plausibility: finger configuration (palm-aligned bone directions) vs. its accepted neighbours
    for s in (0, 1):
        acc = np.flatnonzero(w[:, s] > 0)
        if len(acc) < 3:
            continue
        pb = tgt[acc, s][:, PALM]
        pc = pb - pb.mean(1, keepdims=True)
        R = kabsch(pc, pc[len(pc) // 2][None].repeat(len(pc), 0))  # common palm frame
        bd = bones(np.einsum("nij,nkj->nki", R, tgt[acc, s]))
        fing = [c for c in range(1, 21) if c not in (5, 9, 13, 17)]
        x = tgt[acc, s]  # palm axes in the camera frame: wrist -> middle MCP, palm normal
        ax = np.stack([unit(x[:, 9] - x[:, 0]), unit(np.cross(x[:, 5] - x[:, 0], x[:, 17] - x[:, 0]))], 1)
        bad = []
        for k, t in enumerate(acc):
            nb = [m for m in range(max(0, k - 4), min(len(acc), k + 5)) if m != k and abs(acc[m] - t) <= 8]
            if len(nb) < 2:
                continue
            med = unit(np.median(bd[nb][:, fing], 0))
            ang = np.degrees(np.arccos(np.clip((bd[k, fing] * med).sum(-1), -1, 1))).mean()
            mp = unit(np.median(ax[nb], 0))
            angp = np.degrees(np.arccos(np.clip((ax[k] * mp).sum(-1), -1, 1))).max()
            if ang > a.max_spike or angp > a.max_spike:
                bad.append(t)
        w[bad, s] = 0
        st["spike"][s] = len(bad)
    st["accepted"] = [int((w[:, s] > 0).sum()) for s in (0, 1)]
    return tgt, tgt2d, ext, w, det_i, dg, st


def fit(model, pose0, scales, shape, tgt, w, a, tgt2d=None, ext=None, K=None):
    dev = model.dev
    cols = {s: (FINGER_COLS[s] if a.fingers else []) + (WRIST_COLS[a.wrist][s] if a.wrist != "none" else [])
            for s in (0, 1)}
    allc = cols[0] + cols[1]
    P0 = torch.tensor(pose0, dtype=torch.float32, device=dev)
    sc = torch.tensor(scales, dtype=torch.float32, device=dev).reshape(-1, 68)
    sh = torch.tensor(shape, dtype=torch.float32, device=dev).reshape(-1, 45)
    T = len(pose0)
    bw = torch.tensor(bone_weights(a.wrist), dtype=torch.float32, device=dev)
    with torch.no_grad():
        J0 = model.joints(P0, sc, sh).cpu().numpy()
    tg = tgt.copy()
    if a.wrist == "none":  # keep the body's palm orientation: align the target palm to the body palm
        for s in (0, 1):
            ok = w[:, s] > 0
            if not ok.any():
                continue
            kb = J0[ok][:, HAND_J[s]]
            src = tg[ok, s][:, PALM] - tg[ok, s][:, PALM].mean(1, keepdims=True)
            dst = kb[:, PALM] - kb[:, PALM].mean(1, keepdims=True)
            R = kabsch(src, dst)
            tg[ok, s] = np.einsum("nij,nkj->nki", R, tg[ok, s])
    U = torch.tensor(bones(tg), dtype=torch.float32, device=dev)  # [T,2,21,3]
    W = torch.tensor(w, dtype=torch.float32, device=dev)
    lo = torch.tensor(np.minimum(model.lo[allc], pose0[:, allc].min(0)), dtype=torch.float32, device=dev)
    hi = torch.tensor(np.maximum(model.hi[allc], pose0[:, allc].max(0)), dtype=torch.float32, device=dev)
    prior = torch.full((len(allc),), a.prior, device=dev)
    if a.wrist != "none":
        nw = len(WRIST_COLS[a.wrist][1])
        nf = 27 if a.fingers else 0
        for s0 in (0, len(cols[0])):
            prior[s0 + nf:s0 + nf + nw] = a.wrist_prior
    delta = torch.zeros(T, len(allc), device=dev, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=a.lr)
    ci = torch.tensor(allc, device=dev)
    hj = torch.tensor([HAND_J[0], HAND_J[1]], device=dev)
    par = torch.tensor(PARENT.clip(0), device=dev)

    use2d = a.w2d > 0 and tgt2d is not None
    if use2d:
        T2 = torch.tensor(tgt2d, dtype=torch.float32, device=dev)
        E2 = torch.tensor(ext, dtype=torch.float32, device=dev)
        Kt = torch.tensor(K, dtype=torch.float32, device=dev)

    def data_term(Jt, with2d=True):
        kp = Jt[:, hj]  # [T,2,21,3]
        b = kp - kp[:, :, par]
        b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        e = ((1 - (b * U).sum(-1)) * bw).sum(-1)  # [T,2]
        if use2d and with2d:  # reprojection of the finger joints onto WiLoR's 2D, in hand lengths
            uv = torch.stack([Kt[0] * kp[..., 0] / kp[..., 2] + Kt[2], Kt[1] * kp[..., 1] / kp[..., 2] + Kt[3]], -1)
            r = ((uv[:, :, 1:] - T2[:, :, 1:]) ** 2).sum(-1).mean(-1) / E2 ** 2
            e = e + a.w2d * r
        return e

    hist = []
    for it in range(a.iters):
        opt.zero_grad()
        pose = P0.clone()
        pose[:, ci] = P0[:, ci] + delta
        e = data_term(model.joints(pose, sc, sh))
        loss = (e * W).sum() + (delta ** 2 * prior).sum() + a.smooth * ((delta[1:] - delta[:-1]) ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.copy_(torch.maximum(torch.minimum(P0[:, ci] + delta, hi), lo) - P0[:, ci])
        if it % 50 == 0 or it == a.iters - 1:
            hist.append(float(loss))
    with torch.no_grad():
        pose = P0.clone()
        pose[:, ci] = P0[:, ci] + delta
        e1 = data_term(model.joints(pose, sc, sh), False).cpu().numpy()
        e0 = data_term(torch.tensor(J0, device=dev), False).cpu().numpy()
    return pose.cpu().numpy(), allc, e0, e1, hist


def residual_deg(e, w, nb):
    """Mean bone-angle residual (degrees, approx) on accepted frames from the summed (1 - cos)."""
    m = w > 0
    return float(np.degrees(np.sqrt(2 * e[m] / nb)).mean()) if m.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default=f"{WORK}/body_vA")
    ap.add_argument("--wilor", default=f"{WORK}/hand_wilor")
    ap.add_argument("--raw-dir", default=f"{WORK}/body",
                    help="per-frame run whose fx/fy/cx/cy belong to --body (projection, ray correction, 2D term)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--wrist", choices=["none", "flex", "twist"], default="none")
    ap.add_argument("--fingers", type=int, default=1, help="0: keep the body's finger parameters (wrist only)")
    ap.add_argument("--max-dist", type=float, default=0.8, help="max centre / wrist distance, in hand lengths")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--min-visible", type=float, default=0.5)
    ap.add_argument("--require-cls", type=int, default=1, help="detector class must agree with the body side")
    ap.add_argument("--max-palm", type=float, default=180.0,
                    help="max rotation (deg) between WiLoR's palm and the body's palm (catches mirrored hands)")
    ap.add_argument("--max-spike", type=float, default=30.0)
    ap.add_argument("--w2d", type=float, default=0.0,
                    help="weight of a reprojection term on WiLoR's 2D finger joints (lets the hand orientation "
                         "absorb the body's wrist position error); 0 = 3D bone directions only")
    ap.add_argument("--freeze-first", type=int, default=0,
                    help="frames on each side of the first scored frame that keep the base hands (ramped back in "
                         "over --freeze-ramp frames); 0 = off. The scorer fits its Sim(3) on that frame's vertices")
    ap.add_argument("--freeze-ramp", type=int, default=15)
    ap.add_argument("--scored", default=f"{WORK}/body_v2/scored_frames.json")
    ap.add_argument("--prior", type=float, default=0.05)
    ap.add_argument("--wrist-prior", type=float, default=0.5)
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--iters", type=int, default=300)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(dev)
    os.makedirs(os.path.join(a.out, "diag"), exist_ok=True)
    nb = bone_weights(a.wrist).sum()
    for ep in a.episodes:
        b = dict(np.load(f"{a.body}/episode_{ep:06d}.npz"))
        raw = np.load(f"{a.raw_dir}/episode_{ep:06d}.npz")  # intrinsics of the per-frame run behind --body
        K = np.array([float(raw["fx"]), float(raw["fy"]), float(raw["cx"]), float(raw["cy"])])
        pose0 = b["pose"].astype(np.float64)
        valid = b["valid"].astype(bool)
        with torch.no_grad():
            J0 = model.joints(torch.tensor(pose0, dtype=torch.float32, device=dev),
                              torch.tensor(b["scales"], dtype=torch.float32, device=dev).reshape(-1, 68),
                              torch.tensor(b["shape"], dtype=torch.float32, device=dev).reshape(-1, 45)).cpu().numpy()
        wl = dict(np.load(f"{a.wilor}/episode_{ep:06d}.npz"))
        tgt, tgt2d, ext, w, det_i, dg, st = gate_and_targets(ep, wl, J0, K, a, valid)
        if (w > 0).sum() == 0:
            pose, e0, e1, hist, allc = pose0, np.zeros((len(pose0), 2)), np.zeros((len(pose0), 2)), [], []
        else:
            pose, allc, e0, e1, hist = fit(model, pose0, b["scales"], b["shape"], tgt, w, a, tgt2d, ext, K)
        if a.freeze_first > 0:  # keep the base hands around the first scored frame (the scorer's Sim(3) frame)
            sc0 = int(json.load(open(a.scored))[str(ep)][0])
            dist = np.abs(np.arange(len(pose)) - sc0)
            ramp = np.clip((dist - a.freeze_first) / float(a.freeze_ramp), 0, 1)
            pose = pose0 + (pose - pose0) * ramp[:, None]
        d = np.abs(pose - pose0)
        st.update(res_before=[residual_deg(e0[:, s], w[:, s], nb) for s in (0, 1)],
                  res_after=[residual_deg(e1[:, s], w[:, s], nb) for s in (0, 1)], loss=hist,
                  mean_abs_change=float(d[:, 68:122].mean()), max_abs_change=float(d.max()),
                  changed_cols=sorted(set(np.flatnonzero(d.max(0) > 1e-6).tolist())))
        b["pose"] = pose.astype(np.float32)
        np.savez_compressed(f"{a.out}/episode_{ep:06d}.npz", **b)
        np.savez_compressed(f"{a.out}/diag/episode_{ep:06d}_gate.npz", w=w, det=det_i, **dg)
        json.dump(st, open(f"{a.out}/diag/episode_{ep:06d}.json", "w"))
        print(f"ep {ep}: accepted L/R {st['accepted']} of {st['frames']} (matched {st['matched']}, low conf "
              f"{st['low_conf']}, cls {st['cls_disagree']}, occl {st['occluded']}, palm {st['palm_flip']}, "
              f"spike {st['spike']}) "
              f"residual deg L {st['res_before'][0]:.1f}->{st['res_after'][0]:.1f} "
              f"R {st['res_before'][1]:.1f}->{st['res_after'][1]:.1f}", flush=True)
    src, dst = os.path.join(a.body, "metrics"), os.path.join(a.out, "metrics")
    if os.path.isdir(src) and not os.path.exists(dst):
        shutil.copytree(src, dst)
    json.dump(vars(a), open(os.path.join(a.out, "diag", "args.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
