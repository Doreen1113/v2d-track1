"""Whole-sequence refinement of the per-frame SAM 3D Body estimates (MHR parameters).

One identity per episode (median shape / scales), then a joint optimisation over all frames of
    - 2D keypoints: ViTPose-H COCO-17 (GVHMR preprocessing) + SAM 3D Body's own 70 keypoints (weak)
    - prior to the per-frame SAM 3D Body pose parameters
    - temporal smoothness: acceleration of root-relative joints, of the root (xy / depth separately)
      and of the pose parameters
    - static floor plane (normal from MoGe-2 depth, height from the feet): no foot below the floor;
      feet in contact (low and slow) stay on the floor and do not slide
Everything lives in the Track 1 frame (OpenCV camera, metres; MHR forward as in mhr_utils.py).

Output: <out>/episode_XXXXXX.npz with pose [T,136], scales [68], shape [45], valid [T], plus
contact [T,2], floor (n, d) and diagnostics; <out>/metrics/episode_XXXXXX.json with before/after metrics.

  python refine_body_sequence.py --episodes 0 1 2
  python refine_body_sequence.py --all --shard 0/6
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import unpack_masks  # noqa: E402
from mhr_utils import DEFAULT_MHR, FLIP  # noqa: E402

ROOT = "/work/doreen071/v2d"
BODY_DIR = f"{ROOT}/work/body"
VIT_DIR = f"{ROOT}/work/body_gvhmr/preproc"
DEPTH_DIR = f"{ROOT}/work/depth"
MASK_DIR = f"{ROOT}/work/masks"
OUT_DIR = f"{ROOT}/work/body_v2"
KP_MAP = f"{ROOT}/work/body_v2/kp70_mapping.npz"  # rows 0:70 of SAM 3D Body's MHR keypoint regressor
SAMPLE = f"{ROOT}/kit/v2d_submission_kit/data/track_1_sample_submission.parquet"

# Scorer's 22 body joints (MHR skeleton indices)
J22 = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
# COCO-17 -> MHR70 keypoint index
COCO_TO_70 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
FOOT_KP = [[13, 15, 16, 17], [14, 18, 19, 20]]  # (ankle, big toe, small toe, heel) left / right
SOLE_KP = [[15, 16, 17], [18, 19, 20]]
BODY70 = list(range(0, 21)) + [41, 62] + list(range(63, 70))


# ----------------------------------------------------------------------------------------- data
def fill_invalid(x, valid):
    x = np.array(x, dtype=np.float64)
    idx = np.flatnonzero(valid)
    flat = x.reshape(len(x), -1)
    t = np.arange(len(x))
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(t, idx, flat[idx, j])
    return flat.reshape(x.shape)


def load_episode(ep):
    b = dict(np.load(f"{BODY_DIR}/episode_{ep:06d}.npz"))
    T = len(b["pose"])
    valid = b["valid"].astype(bool)
    vit = torch.load(f"{VIT_DIR}/episode_{ep:06d}_feat.pt", map_location="cpu", weights_only=False)
    kv = vit["kp2d"].numpy().astype(np.float64)
    vv = vit["detected"].numpy().astype(bool) if "detected" in vit else vit["valid"].numpy().astype(bool)
    if len(kv) != T:
        raise ValueError(f"ep {ep}: ViTPose length {len(kv)} != {T}")
    kv[~vv, :, 2] = 0
    return b, T, valid, kv


def scored_frames():
    import pandas as pd  # noqa: F401  (only available in some envs)
    df = pd.read_parquet(SAMPLE, columns=["row_id"])
    p = df.row_id.str.split("_", expand=True)
    out = {}
    for ep, g in p.groupby(1):
        out[int(ep)] = sorted({int(x) for x in g[2] if int(x) != 999999})
    return out


def load_scored(path):
    if path and os.path.exists(path):
        return {int(k): v for k, v in json.load(open(path)).items()}
    return scored_frames()


# ---------------------------------------------------------------------------------------- floor
def fit_floor_moge(ep, n_frames=40, seed=0, fx=None):
    """RANSAC planes on the static background (median MoGe depth over frames, person/object masked out).
    Returns a list of candidate planes (n, d, inlier_frac) with n pointing up (towards -y of the camera).
    fx: focal of the body run; if it differs from the MoGe focal, the background keeps its pixels and moves to
    depth z * fx / fx_moge, as the body does."""
    d = np.load(f"{DEPTH_DIR}/episode_{ep:06d}.npz")
    D = d["depth"]
    K = d["K"].astype(np.float64) / float(d["downsample"])
    K[2, 2] = 1.0
    T, h, w = D.shape
    m = np.load(f"{MASK_DIR}/episode_{ep:06d}.npz")
    mh, mw = int(m["mask_h"]), int(m["mask_w"])
    fr = np.linspace(0, T - 1, min(n_frames, T)).round().astype(int)
    stack = []
    for f in fr:
        z = D[f].astype(np.float32)
        pm = unpack_masks(m["person_masks"], mh, mw, f) | unpack_masks(m["obj_masks"], mh, mw, f)
        pm = pm[:: mh // h, :: mw // w][:h, :w]
        from scipy.ndimage import binary_dilation
        pm = binary_dilation(pm, iterations=4)
        z = np.where(pm | ~np.isfinite(z) | (z <= 0), np.nan, z)
        stack.append(z)
    Z = np.nanmedian(np.stack(stack), 0)
    v, u = np.mgrid[0:h, 0:w]
    sel = (v > 0.45 * h) & np.isfinite(Z) & (Z < 15)
    P = np.stack([(u[sel] - K[0, 2]) / K[0, 0] * Z[sel], (v[sel] - K[1, 2]) / K[1, 1] * Z[sel], Z[sel]], 1)
    if fx is not None:
        P[:, 2] *= fx / float(d["K"][0, 0])
    rng = np.random.default_rng(seed)
    planes = []
    remaining = P.copy()
    for _ in range(4):
        if len(remaining) < 500:
            break
        best = None
        for _ in range(600):
            s = remaining[rng.choice(len(remaining), 3, replace=False)]
            n = np.cross(s[1] - s[0], s[2] - s[0])
            if np.linalg.norm(n) < 1e-9:
                continue
            n /= np.linalg.norm(n)
            if n[1] > 0:
                n = -n
            if -n[1] < 0.6:  # floor-like: normal within ~53 deg of camera up
                continue
            dd = n @ s[0]
            inl = np.abs(remaining @ n - dd) < 0.03
            if best is None or inl.sum() > best[2].sum():
                best = (n, dd, inl)
        if best is None:
            break
        n, dd, inl = best
        # least-squares refit on inliers
        Q = remaining[inl]
        c = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
        n = vt[2] if vt[2][1] < 0 else -vt[2]
        dd = n @ c
        inl = np.abs(remaining @ n - dd) < 0.03
        planes.append((n, float(dd), float(inl.sum()) / len(P)))
        remaining = remaining[~inl]
    return planes


# ------------------------------------------------------------------------------------- model
class Body:
    def __init__(self, device, shape, scales):
        self.dev = device
        self.model = torch.jit.load(DEFAULT_MHR, map_location=device).float().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.flip = torch.tensor(FLIP, dtype=torch.float32, device=device)
        M = sp.load_npz(KP_MAP).tocsr()
        nnz = np.diff(M.indptr).max()
        idx = np.zeros((70, nnz), np.int64)
        wt = np.zeros((70, nnz), np.float32)
        for r in range(70):
            a, b = M.indptr[r], M.indptr[r + 1]
            idx[r, : b - a] = M.indices[a:b]
            wt[r, : b - a] = M.data[a:b]
        self.kidx = torch.tensor(idx, device=device)
        self.kw = torch.tensor(wt, device=device)
        self.shape = torch.tensor(shape, dtype=torch.float32, device=device)
        self.scales = torch.tensor(scales, dtype=torch.float32, device=device)
        self.nv = 18439

    def __call__(self, pose, want_verts=False):
        T = pose.shape[0]
        p = torch.cat([pose, self.scales[None].expand(T, -1)], 1)
        v, sk = self.model(self.shape[None].expand(T, -1).contiguous(), p,
                           torch.zeros(T, 72, device=self.dev), True)
        v = v * self.flip / 100.0
        j = sk[..., :3] * self.flip / 100.0
        vj = torch.cat([v, j], 1)
        kp = (vj[:, self.kidx] * self.kw[None, :, :, None]).sum(2)  # [T,70,3]
        return (kp, j, v) if want_verts else (kp, j)


def project_t(x, K):
    z = x[..., 2].clamp(min=0.2)
    return torch.stack([K[0] * x[..., 0] / z + K[2], K[1] * x[..., 1] / z + K[3]], -1)


def gmof(r2, s2):
    return s2 * r2 / (r2 + s2)


def acc(x):
    return x[2:] - 2 * x[1:-1] + x[:-2]


# ------------------------------------------------------------------------------------- fitting
def optimise(body, init, prior, K, kv, ksam, wdata, cfg, floor=None, contact=None, iters=600, lr=0.01,
             log=None, tag=""):
    """init/prior [T,136] (unwrapped), kv [T,17,3] ViTPose, ksam [T,70,2], wdata [T] data-term weight."""
    dev = body.dev
    T = init.shape[0]
    tt = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    x = torch.nn.Parameter(tt(init).clone())
    pr = tt(prior)
    kvt = tt(kv)
    kst = tt(ksam)
    wd = tt(wdata)
    Kt = tt(K)
    cidx = torch.tensor(COCO_TO_70, device=dev)
    sig_th = tt(np.r_[np.full(3, 1e9), np.full(3, cfg["sig_rootrot"]), np.full(130, cfg["sig_pose"])])
    opt = torch.optim.Adam([x], lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.6), int(iters * 0.85)], 0.3)
    if floor is not None:
        fn, fd = tt(floor[0]), float(floor[1])
    if contact is not None:
        ct = tt(contact)  # [T,2]
    samw = torch.ones(70, device=dev) * cfg["w_sam"]
    samw[15:21] = cfg["w_sam_feet"]
    samw[21:62] = cfg["w_sam"] * 0.3
    for it in range(iters):
        opt.zero_grad()
        kp, j = body(x)
        uv = project_t(kp, Kt)
        # ViTPose
        r = (uv[:, cidx] - kvt[..., :2]) / cfg["sig_vit"]
        c = kvt[..., 2] * (kvt[..., 2] > cfg["vit_thr"])
        L_vit = (gmof(r.square().sum(-1), 9.0) * c).sum(1) * wd
        # SAM 3D Body keypoints (its own projection; weak, mostly for feet and hands)
        r = (uv - kst) / cfg["sig_sam"]
        L_sam = (gmof(r.square().sum(-1), 9.0) * samw).sum(1) * wd
        # parameter prior
        L_pr = (((x - pr) / sig_th).square().sum(1) + ((x[:, 2] - pr[:, 2]) * 0.1 / cfg["sig_depth"]).square()) * wd
        loss = (L_vit.sum() + L_sam.sum() + L_pr.sum()) / T
        # smoothness
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
        # floor
        L_f = torch.zeros((), device=dev)
        if floor is not None:
            hgt = kp[:, 15:21] @ fn - fd  # sole keypoints
            L_f = (torch.relu(-hgt) / 0.01).square().sum()
            if contact is not None:
                for s in range(2):
                    sole = kp[:, SOLE_KP[s]]
                    hs = sole @ fn - fd
                    hmin = -cfg["soft_k"] * torch.logsumexp(-hs / cfg["soft_k"], 1)
                    L_f = L_f + (ct[:, s] * ((hmin - cfg["contact_h"]) / cfg["sig_contact_h"]).square()).sum()
                    fk = kp[:, FOOT_KP[s]]
                    v = fk[1:] - fk[:-1]
                    both = ct[1:, s] * ct[:-1, s]
                    L_f = L_f + (both[:, None] * (v / cfg["sig_slide"]).square().sum(-1)).sum()
            L_f = L_f * cfg["w_floor"] / T
        loss = loss + L_f
        loss.backward()
        opt.step()
        sched.step()
        if log is not None and (it % 200 == 0 or it == iters - 1):
            log(f"  [{tag}] it {it:4d} loss {loss.item():9.3f} vit {L_vit.sum().item() / T:7.3f} "
                f"sam {L_sam.sum().item() / T:7.3f} prior {L_pr.sum().item() / T:7.3f} "
                f"smooth {cfg['w_smooth'] * L_s.item() / T:7.3f} floor {L_f.item():7.3f}")
    return x.detach().cpu().numpy().astype(np.float64)


def forward_np(body, pose, verts=False, chunk=512):
    out = [[], [], []]
    with torch.no_grad():
        for a in range(0, len(pose), chunk):
            r = body(torch.as_tensor(pose[a:a + chunk], dtype=torch.float32, device=body.dev), want_verts=verts)
            for i, t in enumerate(r):
                out[i].append(t.cpu().numpy())
    return [np.concatenate(o) for o in out if o]


# ------------------------------------------------------------------------------------- contacts
def detect_contacts(kp, floor, cfg):
    n, d = floor
    C = np.zeros((len(kp), 2), bool)
    info = []
    for s in range(2):
        h = (kp[:, SOLE_KP[s]] @ n - d).min(1)
        c = kp[:, FOOT_KP[s]].mean(1)
        c = gaussian_filter1d(c, 1.5, axis=0, mode="nearest")
        v = np.zeros(len(c))
        v[1:-1] = np.linalg.norm((c[2:] - c[:-2]) / 2 - np.outer((c[2:] - c[:-2]) / 2 @ n, n), axis=1)
        v[0], v[-1] = v[1], v[-2]
        m = (h < cfg["contact_h_thr"]) & (v < cfg["contact_v_thr"])
        m = binary_opening(binary_closing(m, iterations=2), iterations=2)
        C[:, s] = m
        info.append(dict(h=h, v=v))
    return C, info


# ------------------------------------------------------------------------------------- metrics
def metrics(kp, j, kv, valid, scored, contact2d):
    """kp [T,70,3], j [T,127,3] in metres (camera frame)."""
    K = METRIC_K
    out = {}
    pel = j[:, 1]
    z = pel[:, 2]
    vz = valid[1:] & valid[:-1]
    dz = np.abs(np.diff(z))[vz] * 100
    out["depth_dz_p50_cm"] = float(np.percentile(dz, 50))
    out["depth_dz_p99_cm"] = float(np.percentile(dz, 99))
    vz2 = valid[2:] & valid[1:-1] & valid[:-2]
    out["depth_acc_cm"] = float(np.abs(z[2:] - 2 * z[1:-1] + z[:-2])[vz2].mean() * 100)
    uv = np.stack([K[0] * kp[..., 0] / kp[..., 2] + K[2], K[1] * kp[..., 1] / kp[..., 2] + K[3]], -1)
    e = np.linalg.norm(uv[:, COCO_TO_70] - kv[..., :2], axis=-1)
    m = (kv[..., 2] > 0.5) & valid[:, None]
    out["reproj_vit_px"] = float(e[m].mean())
    sm = np.zeros(len(kp), bool)
    sm[[f for f in scored if f < len(kp)]] = True
    out["reproj_vit_px_scored"] = float(e[m & sm[:, None]].mean()) if (m & sm[:, None]).any() else float("nan")
    # foot skating: 3D speed of feet that are stationary in the image
    sk = []
    for s in range(2):
        f3 = kp[:, FOOT_KP[s]].mean(1)
        sp3 = np.linalg.norm(np.diff(f3, axis=0), axis=1) * 100
        mm = contact2d[1:, s] & contact2d[:-1, s] & valid[1:] & valid[:-1]
        sk.append(sp3[mm])
    sk = np.concatenate(sk)
    out["skate_cm_per_frame"] = float(sk.mean()) if len(sk) else float("nan")
    out["skate_frames"] = int(len(sk))
    # 22-joint acceleration on consecutive scored frames
    js = j[:, J22]
    a = np.linalg.norm(js[2:] - 2 * js[1:-1] + js[:-2], axis=-1) * 100
    ok = sm[2:] & sm[1:-1] & sm[:-2]
    out["acc22_cm"] = float(a[ok].mean()) if ok.any() else float("nan")
    return out


METRIC_K = None


def stationary_feet_2d(kv, kpsam, valid):
    """Feet stationary in the image (ankle + SAM heel/toes speed < 1.2 px/frame, smoothed)."""
    C = np.zeros((len(kv), 2), bool)
    for s, (ia, sole) in enumerate(((15, [15, 16, 17]), (16, [18, 19, 20]))):
        a = gaussian_filter1d(kv[:, ia, :2], 1.0, axis=0, mode="nearest")
        b = gaussian_filter1d(kpsam[:, sole].mean(1), 1.0, axis=0, mode="nearest")
        va = np.r_[0, np.linalg.norm(np.diff(a, axis=0), axis=1)]
        vb = np.r_[0, np.linalg.norm(np.diff(b, axis=0), axis=1)]
        C[:, s] = (va < 1.2) & (vb < 1.2) & (kv[:, ia, 2] > 0.6) & valid
    return C


# ------------------------------------------------------------------------------------- main
DEFAULT_CFG = dict(
    sig_vit=5.0, vit_thr=0.3, sig_sam=12.0, w_sam=0.3, w_sam_feet=1.0,
    sig_pose=0.25, sig_rootrot=0.15, sig_depth=0.30,
    w_smooth=1.0, sig_acc_rel=0.008, sig_acc_root_xy=0.008, sig_acc_root_z=0.002, sig_acc_param=0.05,
    sig_vel_root_z=0.05,
    w_floor=1.0, contact_h=0.0, sig_contact_h=0.015, sig_slide=0.004, soft_k=0.01,
    contact_h_thr=0.08, contact_v_thr=0.012,
)


def run_episode(ep, args, cfg, scored, log):
    global METRIC_K
    t0 = time.time()
    b, T, valid, kv = load_episode(ep)
    K = np.array([float(b["fx"]), float(b["fy"]), float(b["cx"]), float(b["cy"])])
    METRIC_K = K
    dev = torch.device("cuda")
    shape = np.median(b["shape"][valid], 0)
    scales = np.median(b["scales"][valid], 0)
    pose0 = fill_invalid(b["pose"], valid)
    pose0[:, 3:] = np.unwrap(pose0[:, 3:], axis=0)
    ksam = fill_invalid(b["kp2d"], valid)
    # data weight: detected person and the ViTPose track agrees with the SAM 3D Body box
    wdata = valid.astype(np.float64)
    kv = kv.copy()
    kv[~valid, :, 2] = 0
    if args.holdout:
        ho = np.zeros(T, bool)
        ho[1::2] = True
        wdata[ho] = 0
    body = Body(dev, shape, scales)
    sc = scored.get(ep, [])
    st2d = stationary_feet_2d(kv, ksam, valid)

    res = {"episode": ep, "T": T, "n_valid": int(valid.sum()), "cfg": cfg}
    # --- before: raw SAM 3D Body per frame (per-frame identity) and the v1 assembly
    raw = Body(dev, b["shape"][valid][0], b["scales"][valid][0])
    kp_raw, j_raw = [], []
    with torch.no_grad():
        for a in range(0, T, 256):
            p = torch.as_tensor(pose0[a:a + 256], dtype=torch.float32, device=dev)
            n_ = len(p)
            pp = torch.cat([p, torch.as_tensor(fill_invalid(b["scales"], valid)[a:a + 256], dtype=torch.float32,
                                               device=dev)], 1)
            shp = torch.as_tensor(fill_invalid(b["shape"], valid)[a:a + 256], dtype=torch.float32, device=dev)
            v, sk = raw.model(shp, pp, torch.zeros(n_, 72, device=dev), True)
            v = v * raw.flip / 100
            jj = sk[..., :3] * raw.flip / 100
            vj = torch.cat([v, jj], 1)
            kp_raw.append((vj[:, raw.kidx] * raw.kw[None, :, :, None]).sum(2).cpu().numpy())
            j_raw.append(jj.cpu().numpy())
    kp_raw, j_raw = np.concatenate(kp_raw), np.concatenate(j_raw)
    res["raw"] = metrics(kp_raw, j_raw, kv, valid, sc, st2d)
    pose_v1 = gaussian_filter1d(pose0, 2.0, axis=0, mode="nearest")
    kp_v1, j_v1 = forward_np(body, pose_v1)
    res["v1"] = metrics(kp_v1, j_v1, kv, valid, sc, st2d)

    # --- stage A: re-solve depth for the fixed identity (per frame, scale of the 2D skeleton)
    kp0, j0 = forward_np(body, pose0)
    ratio = np.ones(T)
    for t in range(T):
        u = np.stack([K[0] * kp0[t, :, 0] / kp0[t, :, 2] + K[2], K[1] * kp0[t, :, 1] / kp0[t, :, 2] + K[3]], -1)
        s0 = np.linalg.norm(u[BODY70] - u[BODY70].mean(0), axis=1).mean()
        s1 = np.linalg.norm(ksam[t, BODY70] - ksam[t, BODY70].mean(0), axis=1).mean()
        ratio[t] = s0 / max(s1, 1e-6)
    # moving the whole body along the pelvis ray by factor r scales its image by ~1/r
    init = pose0.copy()
    pel = j0[:, 1]
    new_pel = pel * ratio[:, None]
    init[:, :3] += (new_pel - pel) * FLIP / 0.1
    res["identity_depth_ratio_median"] = float(np.median(ratio))

    # --- stage B: temporal fit without contacts
    stage_b = optimise(body, init, init, K, kv, ksam, wdata, cfg, iters=args.iters, log=log, tag="B")
    kpB, jB = forward_np(body, stage_b)

    # --- floor: MoGe normal, offset from the feet in the body frame
    floor = None
    contact = np.zeros((T, 2), bool)
    if not args.no_floor:
        planes = fit_floor_moge(ep)
        best = None
        for (n, dm, frac) in planes:
            hs = []
            for s in range(2):
                hs.append((kpB[:, SOLE_KP[s]] @ n).min(1))
            hlow = np.minimum(hs[0], hs[1])[valid]
            # offset: robust low percentile of the per-frame lowest sole point
            d_b = float(np.percentile(hlow, 10))
            spread = float(np.percentile(hlow, 50) - np.percentile(hlow, 5))
            cand = dict(n=n.tolist(), d_moge=dm, inlier_frac=frac, d_body=d_b, spread=spread)
            if best is None or (frac > 0.15 and spread < best["spread"]) or best["inlier_frac"] < 0.15:
                best = cand
        res["floor_candidates"] = [dict(n=list(map(float, p[0])), d=p[1], frac=p[2]) for p in planes]
        if best is not None:
            n = np.asarray(best["n"])
            floor = (n, best["d_body"])
            res["floor"] = best
            res["floor_moge_over_body"] = best["d_moge"] / best["d_body"]
            contact, _ = detect_contacts(kpB, floor, cfg)
            contact &= valid[:, None] | True
            res["contact_frac"] = contact.mean(0).tolist()
            # --- stage C: with floor and contacts
            stage_c = optimise(body, stage_b, init, K, kv, ksam, wdata, cfg, floor=floor, contact=contact,
                               iters=args.iters, log=log, tag="C")
            # refine contact labels once
            kpC, _ = forward_np(body, stage_c)
            contact, _ = detect_contacts(kpC, floor, cfg)
            res["contact_frac"] = contact.mean(0).tolist()
            final = optimise(body, stage_c, init, K, kv, ksam, wdata, cfg, floor=floor, contact=contact,
                             iters=args.iters // 2, log=log, tag="D")
        else:
            final = stage_b
    else:
        final = stage_b
    kpF, jF = forward_np(body, final)
    res["v2_nofloor"] = metrics(kpB, jB, kv, valid, sc, st2d)
    res["v2"] = metrics(kpF, jF, kv, valid, sc, st2d)
    if args.holdout:
        ho = np.zeros(T, bool)
        ho[1::2] = True
        for name, kpx, jx in (("v1", kp_v1, j_v1), ("v2", kpF, jF)):
            uv = np.stack([K[0] * kpx[..., 0] / kpx[..., 2] + K[2], K[1] * kpx[..., 1] / kpx[..., 2] + K[3]], -1)
            e = np.linalg.norm(uv[:, COCO_TO_70] - kv[..., :2], axis=-1)
            m = (kv[..., 2] > 0.5) & (valid & ho)[:, None]
            res[f"holdout_{name}"] = float(e[m].mean())
    res["time_s"] = time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "metrics"), exist_ok=True)
    if not args.holdout:
        np.savez_compressed(
            os.path.join(args.out, f"episode_{ep:06d}.npz"),
            pose=final.astype(np.float32), scales=scales.astype(np.float32), shape=shape.astype(np.float32),
            valid=valid, contact=contact, floor_n=floor[0] if floor else np.zeros(3),
            floor_d=floor[1] if floor else 0.0, stationary_2d=st2d,
            pose_stage_b=stage_b.astype(np.float32), pose_init=init.astype(np.float32))
    with open(os.path.join(args.out, "metrics", f"episode_{ep:06d}{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    keys = ["depth_dz_p99_cm", "depth_acc_cm", "reproj_vit_px", "skate_cm_per_frame", "acc22_cm"]
    for name in ("raw", "v1", "v2_nofloor", "v2"):
        log(f"ep {ep} {name:10s} " + " ".join(f"{k}={res[name][k]:.3f}" for k in keys))
    if args.holdout:
        log(f"ep {ep} holdout reproj v1 {res['holdout_v1']:.3f} v2 {res['holdout_v2']:.3f}")
    log(f"ep {ep} done in {res['time_s']:.0f}s floor={res.get('floor')} contact={res.get('contact_frac')}")
    return res


def export_kp_map(path=KP_MAP):
    """Rows 0:70 of SAM 3D Body's keypoint regressor over [lod1 vertices | 127 joints] -> sparse npz."""
    from huggingface_hub import snapshot_download
    ck = torch.load(os.path.join(snapshot_download("facebook/sam-3d-body-dinov3"), "model.ckpt"),
                    map_location="cpu", weights_only=False)
    M = ck.get("state_dict", ck)["head_pose.keypoint_mapping"].float().numpy()[:70]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sp.save_npz(path, sp.csr_matrix(np.where(np.abs(M) > 1e-8, M, 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-kp-map", action="store_true", help="write the 70-keypoint regressor and exit")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--holdout", action="store_true", help="fit on even frames, report 2D error on odd frames")
    ap.add_argument("--no-floor", action="store_true")
    ap.add_argument("--cfg", default="", help="JSON overrides of DEFAULT_CFG")
    ap.add_argument("--tag", default="")
    ap.add_argument("--scored", default=f"{ROOT}/work/body_v2/scored_frames.json")
    args = ap.parse_args()
    if args.export_kp_map:
        export_kp_map()
        return
    cfg = dict(DEFAULT_CFG)
    if args.cfg:
        cfg.update(json.loads(args.cfg))
    eps = list(range(30)) if args.all or not args.episodes else args.episodes
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        eps = eps[i::n]
    scored = load_scored(args.scored)
    log = lambda s: print(s, flush=True)  # noqa: E731
    for ep in eps:
        run_episode(ep, args, cfg, scored, log)


if __name__ == "__main__":
    main()
