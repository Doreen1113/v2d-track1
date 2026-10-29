"""Physical-consistency audit of object trajectories (no ground truth needed).

For every episode and every object directory given (same layout as work/objects_v4: raw MoGe frame,
episode_XXXXXX_{object.ply,track.npz}), the posed object is mapped into the body frame (t, s / ratio)
and compared with the body, the floor, the MoGe depth and the masks:

  size      longest oriented-box extent (cm, body frame) and its ratio to the class median of the
            same directory (the same physical object appears in the three episodes of a class)
  iou       hard-silhouette IoU against the object masks (actor pixels ignored), all visible frames
            and visible scored frames
  kz        rendered object depth / MoGe depth on the object pixels (median), static and held frames:
            1 = consistent with MoGe, k > 1 = object placed farther than MoGe sees it
  support   per static segment: height of the lowest point above the body floor (cm); for segments
            above 30 cm the MoGe table top next to the object (mode of the scene heights within 60 cm)
  gap       hand-object gap (cm) on held frames (object moving, mask touching the actor), median
  k*        per held frame, the scaling about the camera that closes the hand-object gap (median over
            frames where a contact is reachable within 0.7-1.4)
  tilt      furniture only: angle between the class up axis and the floor normal in static segments
  first     the same quantities on the first scored frame (the frame of the body alignment)
  stakes    symmetric chamfer (cm, scorer convention: sum of both directions) between the directories'
            posed objects on the scored frames (how much the choice between them can matter)

Outputs (--out): audit.json, audit.md (tables), overlays/episode_XXXXXX.png (mask contour red,
directories green / blue / magenta / cyan, hands orange; image crop, top view and side view of the
body, hands, objects and the MoGe scene around the object).

    python audit_objects.py --objects work/objects_v4 work/objects_v5 [work/objects_v6] --out work/audit
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import VIDEO_TMPL, WORK, load_meta, read_frame, unpack_masks  # noqa: E402

K_FULL = np.array([[807.66, 0, 768.0], [0, 807.66, 576.0], [0, 0, 1.0]])
DS = 4  # depth / quarter-resolution masks
FURNITURE_UP = {}  # class -> mesh up axis (unit vector, mesh frame), filled by up_axis()
UPRIGHT_REF = {"white_desk": (10, 0), "short_wood_stool": (24, 0), "white_laptop_cart": (28, 0)}
# plausible real-world size of the longest extent (cm); generous ranges, only gross outliers are flagged
SIZE_PRIOR = {"hula_hoop": (60, 110), "big_red_bowl": (22, 40), "iron": (22, 40), "white_desk": (90, 140),
              "black_pan": (30, 55), "foam_grass_block": (15, 40), "paint_roller": (20, 50),
              "pink_foam_roll": (30, 95), "short_wood_stool": (30, 60), "white_laptop_cart": (60, 110)}


# ---------------------------------------------------------------------------------------------- data
def scored_frames():
    return {int(k): np.asarray(v) for k, v in json.load(open(os.path.join(WORK, "scored_frames.json"))).items()}


def quarter(m):
    T, h, w = m.shape
    return m[:, :h // 2 * 2, :w // 2 * 2].reshape(T, h // 2, 2, w // 2, 2).any((2, 4))


def load_masks(ep):
    md = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
    h, w = int(md["mask_h"]), int(md["mask_w"])
    om, pm = unpack_masks(md["obj_masks"], h, w), unpack_masks(md["person_masks"], h, w)
    return om, pm, md["obj_score"]


def stable_depth(ep, omq, pmq):
    """MoGe depth divided by its per-frame scale w.r.t. the static background (as objlib.load_stable_depth)."""
    from scipy.ndimage import gaussian_filter1d
    D = np.load(os.path.join(WORK, "depth", f"episode_{ep:06d}.npz"))["depth"]  # float16
    T = len(D)
    fg = omq | pmq
    ker = np.ones((9, 9), np.uint8)
    sub = np.linspace(0, T - 1, min(T, 120)).astype(int)
    ref = np.median(D[sub].astype(np.float32), 0)
    r = np.ones(T)
    for t in range(T):
        Dt = D[t].astype(np.float32)
        bg = ~(cv2.dilate(fg[t].astype(np.uint8), ker) > 0) & (ref > 0) & (Dt > 0)
        if bg.sum() > 500:
            r[t] = np.median(Dt[bg] / ref[bg])
    return D, gaussian_filter1d(r, 1.0, mode="nearest")


def budget_mesh(ep, sdf_dir):
    d = np.load(os.path.join(sdf_dir, f"episode_{ep:06d}_sdf.npz"))
    return d["mv"].astype(np.float64), d["mf"].astype(np.int64)


def sample_surface(mv, mf, n, seed=0):
    a, b, c = mv[mf[:, 0]], mv[mf[:, 1]], mv[mf[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    ch = rng.choice(len(mf), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)[:, None]
    return (1 - s) * a[ch] + s * (1 - r2[:, None]) * b[ch] + s * r2[:, None] * c[ch]


def obb_extent(P):
    c = P - P.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    q = c @ vt.T
    return np.sort(q.max(0) - q.min(0))[::-1]


class BodyCache:
    """Hand vertices, a vertex subset and the floor of an episode's body (body frame, metres)."""

    def __init__(self, body_dir):
        from mhr_utils import DEFAULT_MHR, MHRForward
        from fix_penetration import hand_vertex_ids
        self.dir = body_dir
        self.mhr = MHRForward(DEFAULT_MHR, device="cuda")
        self.hand_ids = hand_vertex_ids(self.mhr.model)

    def load(self, ep, sub=7):
        b = np.load(os.path.join(self.dir, f"episode_{ep:06d}.npz"))
        hands, verts = [], []
        for a in range(0, len(b["pose"]), 128):
            v, _ = self.mhr(b["pose"][a:a + 128], b["scales"], b["shape"])
            hands.append(v[:, self.hand_ids].astype(np.float32))
            verts.append(v[:, ::sub].astype(np.float32))
        return dict(hands=np.concatenate(hands), verts=np.concatenate(verts),
                    n=np.asarray(b["floor_n"], float), d=float(b["floor_d"]))


# ---------------------------------------------------------------------------------------------- geometry
def project(Pc, K):
    z = np.clip(Pc[..., 2], 1e-4, None)
    return np.stack([K[0, 0] * Pc[..., 0] / z + K[0, 2], K[1, 1] * Pc[..., 1] / z + K[1, 2]], -1)


def silhouette(Vc, F, K, H, W):
    ok = (Vc[F, 2] > 1e-3).all(1)
    uv = project(Vc, K)
    sil = np.zeros((H, W), np.uint8)
    cv2.fillPoly(sil, list(np.round(uv[F[ok]] * 4).astype(np.int32)), 1, lineType=cv2.LINE_8, shift=2)
    return sil > 0


def splat_depth(Pc, K, H, W):
    z = Pc[:, 2]
    ok = z > 0.05
    uv = np.round(project(Pc[ok], K)).astype(int)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    buf = np.full(H * W, np.inf, np.float32)
    np.minimum.at(buf, uv[inb, 1] * W + uv[inb, 0], z[ok][inb])
    return buf.reshape(H, W)


def floor_frame(n):
    up = n / np.linalg.norm(n)
    x = np.array([1.0, 0, 0]) - up * up[0]
    x /= np.linalg.norm(x)
    z = np.cross(x, up)
    if z[2] < 0:
        z, x = -z, -x
    return np.stack([x, up, z])


def static_segments(seg_id):
    out = []
    for sg in range(int(seg_id.max()) + 1 if len(seg_id) else 0):
        f = np.flatnonzero(seg_id == sg)
        if len(f):
            out.append((sg, f))
    return out


def up_axis(cls, objects_dir, floor_normal):
    """Mesh up axis of a furniture class: the floor normal in mesh coordinates in a reference episode
    where it stands upright (UPRIGHT_REF)."""
    if cls in FURNITURE_UP or cls not in UPRIGHT_REF:
        return FURNITURE_UP.get(cls)
    ep, f = UPRIGHT_REF[cls]
    tr = np.load(os.path.join(objects_dir, f"episode_{ep:06d}_track.npz"))
    u = tr["R"][f].T @ floor_normal(ep)
    FURNITURE_UP[cls] = u / np.linalg.norm(u)
    return FURNITURE_UP[cls]


# ---------------------------------------------------------------------------------------------- audit
def audit_episode(ep, dirs, meta, scored, ratio, bc, sdf_dir, out_dir, n_pts=3000):
    import torch
    t0 = time.time()
    cls = meta[ep]["object"]
    r = float(ratio[str(ep)])
    om, pm, score = load_masks(ep)
    omq, pmq = quarter(om), quarter(pm)
    D, dscale = stable_depth(ep, omq, pmq)
    T = min(len(D), len(om))
    H4, W4 = omq.shape[1:]
    K4 = K_FULL.copy(); K4[:2] /= DS
    visible = (score[:T] > 0) & (omq[:T].sum((1, 2)) > 8)
    occ = pmq & ~omq
    B = bc.load(ep)
    n, d = B["n"], B["d"]
    Tb = min(T, len(B["hands"]))
    fr_sc = scored.get(ep, np.arange(T))
    fr_sc = fr_sc[fr_sc < Tb]
    f0 = int(fr_sc[0])
    mv, mf = budget_mesh(ep, sdf_dir)
    P = sample_surface(mv, mf, n_pts, seed=ep)
    Pd = sample_surface(mv, mf, 20000, seed=ep + 100)
    Hh = torch.tensor(B["hands"][:Tb, ::3], device="cuda")
    ks = np.exp(np.linspace(np.log(0.7), np.log(1.4), 57))
    kt = torch.tensor(ks, device="cuda", dtype=torch.float32)
    res = dict(ep=ep, cls=cls, ratio=r, T=T, n_scored=int(len(fr_sc)), f0=f0, src={})
    posed_sc = {}
    held_ref = None
    for name, dpath in dirs.items():
        tr = np.load(os.path.join(dpath, f"episode_{ep:06d}_track.npz"))
        R, t, s = tr["R"].astype(np.float64), tr["t"].astype(np.float64), float(tr["scale"])
        seg = tr["seg_id"] if "seg_id" in tr.files else -np.ones(len(t), int)
        held = tr["held"].astype(bool) if "held" in tr.files else np.zeros(len(t), bool)
        if held_ref is None:
            held_ref = held.copy()
        held = held_ref  # one definition of "held" for all directories (the first one's)
        Tn = min(Tb, len(t))
        sb = s / r
        ext = obb_extent(P * sb) * 100
        # silhouette IoU and MoGe depth ratio
        iou = np.full(Tn, np.nan)
        kz = np.full(Tn, np.nan)
        ker = np.ones((3, 3), np.uint8)
        for f in range(Tn):
            if not visible[f]:
                continue
            Vc = s * mv @ R[f].T + t[f]
            sil = silhouette(Vc, mf, K4, H4, W4)
            keep = ~occ[f]
            iou[f] = (sil & omq[f] & keep).sum() / max(((sil | omq[f]) & keep).sum(), 1)
            mm = (cv2.erode(omq[f].astype(np.uint8), ker) > 0) & ~pmq[f]
            if mm.sum() >= 15:
                zb = splat_depth(s * Pd @ R[f].T + t[f], K4, H4, W4)
                Df = D[f].astype(np.float32) / np.float32(dscale[f])
                ok = mm & np.isfinite(zb) & (Df > 0)
                if ok.sum() >= 15:
                    kz[f] = np.median(zb[ok]) / np.median(Df[ok])
        # body-frame points
        Pb = s * np.einsum("tij,nj->tni", R[:Tn], P) / r + (t[:Tn] / r)[:, None]
        hmin = (Pb @ n - d).min(1)
        gap = np.full(Tn, np.nan)
        kstar = np.full(Tn, np.nan)
        for f in range(Tn):
            O = torch.tensor(Pb[f, ::3], device="cuda", dtype=torch.float32)
            dm = torch.cdist((kt[:, None, None] * O[None]).reshape(-1, 3), Hh[f]).min(1).values
            g = dm.reshape(len(ks), -1).min(1).values.cpu().numpy()
            gap[f] = float(np.interp(0.0, np.log(ks), g))
            ok = np.flatnonzero(g <= min(g.min() + 0.01, 0.03))
            if len(ok):  # the scaling closest to 1 that (nearly) closes the gap
                kstar[f] = ks[ok[np.argmin(np.abs(np.log(ks[ok])))]]
        st = np.zeros(Tn, bool)
        segs = []
        for sg, fr in static_segments(seg[:Tn]):
            st[fr] = True
            fm = int(fr[len(fr) // 2])
            hs = float(np.median(hmin[fr])) * 100
            row = dict(seg=sg, a=int(fr[0]), b=int(fr[-1]), h=hs, scored=int(np.isin(fr, fr_sc).sum()))
            if hs > 30:
                row["table"] = table_height(D, dscale, omq, pmq, fr, K4, r, n, d, Pb[fm])
            segs.append(row)
        hm = held[:Tn]
        vis = visible[:Tn]
        isc = np.isin(np.arange(Tn), fr_sc)

        def med(x, m):
            x = x[m & np.isfinite(x)]
            return float(np.median(x)) if len(x) else None
        tilt = None
        ax = up_axis(cls, dpath, lambda e: np.asarray(np.load(os.path.join(bc.dir, f"episode_{e:06d}.npz"))["floor_n"], float))
        if ax is not None:
            ang = np.degrees(np.arccos(np.clip(np.einsum("tij,j->ti", R[:Tn], ax) @ n, -1, 1)))
            tilt = dict(first=float(ang[f0]), static=[float(np.median(ang[fr])) for _, fr in static_segments(seg[:Tn])])
        tl = [x["table"]["mode"] - x["h"] for x in segs if "table" in x and x["table"]["mode"] is not None]
        o = dict(scale=s, scale_body=sb, extent_cm=[float(v) for v in ext],
                 iou_med=med(iou, vis), iou_mean=float(np.nanmean(iou[vis])) if vis.any() else None,
                 iou_scored=float(np.nanmean(iou[vis & isc])) if (vis & isc).any() else None,
                 iou_f0=None if np.isnan(iou[f0]) else float(iou[f0]),
                 kz_static=med(kz, st), kz_held=med(kz, hm), kz_scored=med(kz, isc), kz_f0=med(kz, np.abs(np.arange(Tn) - f0) <= 5),
                 gap_held_cm=None if med(gap, hm) is None else 100 * med(gap, hm),
                 kstar_held=med(kstar, hm), kstar_f0=med(kstar, np.abs(np.arange(Tn) - f0) <= 5),
                 h_f0_cm=float(hmin[f0] * 100), gap_f0_cm=float(gap[f0] * 100),
                 below_floor_cm=float(max(0.0, -np.nanmin(hmin[isc])) * 100),
                 held_frac=float(hm.mean()), held_scored=int((hm & isc).sum()), static_scored=int((st & isc).sum()),
                 segments=segs, table_gap_cm=float(np.median(tl)) if tl else None, tilt=tilt,
                 depth_f0_m=float(np.median(Pb[f0, :, 2])))
        res["src"][name] = o
        posed_sc[name] = Pb[fr_sc[fr_sc < Tn]]
        np.savez_compressed(os.path.join(out_dir, "series", f"episode_{ep:06d}_{name}.npz"), iou=iou, kz=kz,
                            gap=gap, kstar=kstar, hmin=hmin, held=hm, static=st, visible=vis)
    names = list(dirs)
    res["stakes_cm"] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = posed_sc[names[i]], posed_sc[names[j]]
            m = min(len(a), len(b))
            res["stakes_cm"][f"{names[i]}-{names[j]}"] = 100 * float(np.mean(chamfer(a[:m], b[:m])))
    overlay(ep, dirs, meta, B, om, pm, D, dscale, r, mv, mf, fr_sc, out_dir)
    res["time_s"] = time.time() - t0
    return res


def table_height(D, dscale, omq, pmq, frames, K4, r, n, d, Pobj):
    """Mode of the MoGe scene heights (body frame) within 15-60 cm (horizontally) of the object."""
    sub = frames[np.linspace(0, len(frames) - 1, min(5, len(frames))).astype(int)]
    Z = np.median(np.stack([D[g].astype(np.float32) / np.float32(dscale[g]) for g in sub]), 0)
    msk = ~(np.any(omq[sub], 0) | np.any(pmq[sub], 0))
    msk = cv2.erode(msk.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    H, W = Z.shape
    v, u = np.mgrid[0:H, 0:W]
    S = np.stack([(u - K4[0, 2]) / K4[0, 0] * Z, (v - K4[1, 2]) / K4[1, 1] * Z, Z], -1)[msk] / r
    hS = S @ n - d
    c = Pobj.mean(0)
    hor = (S - c) - np.outer((S - c) @ n, n)
    dist = np.linalg.norm(hor, axis=1)
    near = (dist < 0.6) & (dist > 0.15) & (hS > 0.3) & (hS < 1.3)
    if near.sum() < 30:
        return dict(mode=None, n=int(near.sum()))
    hist, edges = np.histogram(hS[near], bins=np.arange(0.3, 1.3, 0.01))
    return dict(mode=float(edges[np.argmax(hist)] + 0.005) * 100, n=int(near.sum()))


def chamfer(A, B):
    import torch
    out = []
    for f in range(len(A)):
        a = torch.tensor(A[f], device="cuda", dtype=torch.float32)
        b = torch.tensor(B[f], device="cuda", dtype=torch.float32)
        dm = torch.cdist(a, b)
        out.append(float(dm.min(1).values.mean() + dm.min(0).values.mean()))
    return np.array(out)


# ---------------------------------------------------------------------------------------------- overlays
def overlay(ep, dirs, meta, B, om, pm, D, dscale, r, mv, mf, fr_sc, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames = sorted({int(fr_sc[0]), int(fr_sc[len(fr_sc) // 3]), int(fr_sc[2 * len(fr_sc) // 3]), int(fr_sc[-1])})
    K2 = K_FULL.copy(); K2[:2] /= 2
    K4 = K_FULL.copy(); K4[:2] /= 4
    n, d = B["n"], B["d"]
    Fm = floor_frame(n)
    tracks = {k: np.load(os.path.join(v, f"episode_{ep:06d}_track.npz")) for k, v in dirs.items()}
    cols = [(0, 255, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255)]
    mcols = ["green", "blue", "magenta", "cyan"]
    fig, axes = plt.subplots(len(frames), 3, figsize=(15, 4.3 * len(frames)))
    axes = np.atleast_2d(axes)
    for i, f in enumerate(frames):
        img = cv2.resize(read_frame(VIDEO_TMPL.format(ep), f), (768, 576))
        cs, _ = cv2.findContours(om[f].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, (255, 0, 0), 1)
        pts = [np.argwhere(om[f])[:, ::-1].astype(float)]
        obj3 = []
        for j, (name, tr) in enumerate(tracks.items()):
            if f >= len(tr["t"]):
                continue
            Vc = float(tr["scale"]) * mv @ tr["R"][f].T + tr["t"][f]
            sil = silhouette(Vc, mf, K2, 576, 768)
            c2, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, c2, -1, cols[j], 1)
            pts.append(project(Vc, K2))
            obj3.append((name, Vc / r))
        if f < len(B["hands"]):
            for p in project(B["hands"][f][::12], K2):
                if 0 <= p[0] < 768 and 0 <= p[1] < 576:
                    cv2.circle(img, (int(p[0]), int(p[1])), 1, (255, 165, 0), -1)
        allp = np.concatenate([p for p in pts if len(p)])
        x0, y0 = np.maximum(allp.min(0) - 50, 0).astype(int)
        x1, y1 = np.minimum(allp.max(0) + 50, [767, 575]).astype(int)
        if x1 - x0 < 40 or y1 - y0 < 40:
            x0, y0, x1, y1 = 0, 0, 767, 575
        axes[i, 0].imshow(img[y0:y1, x0:x1])
        axes[i, 0].set_title(f"ep{ep} f{f} {meta[ep]['object']}{' (first scored)' if f == fr_sc[0] else ''}", fontsize=8)
        center = obj3[0][1].mean(0) if obj3 else B["verts"][f].mean(0)

        def tf(Q):
            X = (Q - center) @ Fm.T
            X[:, 1] = Q @ n - d
            return X
        Z = D[f].astype(np.float32) / np.float32(dscale[f])
        H, W = Z.shape
        v, u = np.mgrid[0:H, 0:W]
        S = np.stack([(u - K4[0, 2]) / K4[0, 0] * Z, (v - K4[1, 2]) / K4[1, 1] * Z, Z], -1)
        msk = ~(quarter(om[f:f + 1])[0] | quarter(pm[f:f + 1])[0])
        S = tf(S[msk] / r)
        S = S[(np.abs(S[:, 0]) < 1.3) & (np.abs(S[:, 2]) < 1.3)]
        at, asd = axes[i, 1], axes[i, 2]
        at.scatter(S[:, 0], S[:, 2], c=S[:, 1], s=1, cmap="viridis", vmin=0, vmax=1.2, alpha=0.4)
        asd.scatter(S[:, 2], S[:, 1], c="0.75", s=1, alpha=0.3)
        if f < len(B["verts"]):
            Bv, Bh = tf(B["verts"][f]), tf(B["hands"][f])
            at.scatter(Bv[:, 0], Bv[:, 2], c="0.35", s=1)
            asd.scatter(Bv[:, 2], Bv[:, 1], c="0.35", s=1)
            at.scatter(Bh[:, 0], Bh[:, 2], c="orange", s=1)
            asd.scatter(Bh[:, 2], Bh[:, 1], c="orange", s=1)
        for j, (name, O) in enumerate(obj3):
            Q = tf(O[::4])
            at.scatter(Q[:, 0], Q[:, 2], c=mcols[j], s=1, label=f"{name} z={np.median(O[:, 2]):.2f}")
            asd.scatter(Q[:, 2], Q[:, 1], c=mcols[j], s=1)
        for a_ in (at, asd):
            a_.set_aspect("equal")
            a_.grid(alpha=0.3)
        at.set_xlim(-1.3, 1.3); at.set_ylim(-1.3, 1.3)
        at.set_xlabel("x along floor (m)"); at.set_ylabel("depth along floor (m)")
        asd.set_xlim(-1.3, 1.3); asd.set_ylim(-0.1, 2.0)
        asd.set_xlabel("depth along floor (m)"); asd.set_ylabel("height above body floor (m)")
        at.legend(fontsize=6, markerscale=6, loc="upper right")
        at.set_title("top view (scene coloured by height 0-1.2 m)", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overlays", f"episode_{ep:06d}.png"), dpi=70)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- report
def flags(o, cls, med_size):
    out = []
    lo, hi = SIZE_PRIOR.get(cls, (0, 1e9))
    L = o["extent_cm"][0]
    if not lo <= L <= hi:
        out.append(f"size {L:.0f} cm outside {lo}-{hi}")
    if med_size and abs(np.log(L / med_size)) > np.log(1.08):
        out.append(f"size x{L / med_size:.2f} of class median")
    if o["kz_static"] is not None and abs(np.log(o["kz_static"])) > np.log(1.06) and cls != "hula_hoop":
        out.append(f"static depth x{o['kz_static']:.2f} of MoGe")
    for sg in o["segments"]:
        if -30 < sg["h"] < 30 and abs(sg["h"]) > 3 and sg["scored"]:
            out.append(f"floor seg {sg['a']}-{sg['b']} at {sg['h']:.0f} cm")
        if "table" in sg and sg["table"]["mode"] is not None and abs(sg["table"]["mode"] - sg["h"]) > 4 and sg["scored"]:
            out.append(f"table seg {sg['a']}-{sg['b']}: object {sg['h']:.0f} cm vs MoGe table {sg['table']['mode']:.0f} cm")
    if o["gap_held_cm"] is not None and o["gap_held_cm"] > 5 and o["held_scored"]:
        out.append(f"held gap {o['gap_held_cm']:.0f} cm")
    if o["iou_scored"] is not None and o["iou_scored"] < 0.6:
        out.append(f"scored IoU {o['iou_scored']:.2f}")
    if o["below_floor_cm"] > 3:
        out.append(f"{o['below_floor_cm']:.0f} cm below floor")
    if o["tilt"] is not None and o["tilt"]["first"] > 20:
        out.append(f"tilt {o['tilt']['first']:.0f} deg at first scored frame")
    return out


def write_md(rows, names, path):
    meta_cls = {}
    for r in rows:
        meta_cls.setdefault(r["cls"], []).append(r)
    med = {(c, nm): float(np.median([x["src"][nm]["extent_cm"][0] for x in rs])) for c, rs in meta_cls.items() for nm in names}
    L = ["# Object audit", "", "Body frame (body_v2 + floor ratio). `kz` = rendered depth / MoGe depth on object pixels; "
         "`h` = lowest point above the body floor; `gap` = hand-object gap on held frames; `k*` = scaling that closes it; "
         "`stakes` = chamfer between the directories on the scored frames (cm).", ""]
    hdr = "| ep | class | dir | size cm (x class med) | IoU all/scored/f0 | kz static/held/f0 | static segments h cm (table) | gap held cm / k* | f0: h, depth m | flags |"
    L += [hdr, "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for nm in names:
            o = r["src"][nm]
            f = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"
            segs = "; ".join(f"{s['a']}-{s['b']}{'*' if s['scored'] else ''}: {s['h']:.0f}"
                             + (f" ({s['table']['mode']:.0f})" if "table" in s and s["table"]["mode"] is not None else "")
                             for s in o["segments"][:6])
            fl = flags(o, r["cls"], med[(r["cls"], nm)])
            L.append(f"| {r['ep']} | {r['cls']} | {nm} | {o['extent_cm'][0]:.0f} (x{o['extent_cm'][0] / med[(r['cls'], nm)]:.2f}) | "
                     f"{f(o['iou_mean'], 2)}/{f(o['iou_scored'], 2)}/{f(o['iou_f0'], 2)} | {f(o['kz_static'])}/{f(o['kz_held'])}/{f(o['kz_f0'])} | "
                     f"{segs} | {f(o['gap_held_cm'], 1)} / {f(o['kstar_held'])} | {o['h_f0_cm']:.0f}, {o['depth_f0_m']:.2f} | {'; '.join(fl)} |")
        st = ", ".join(f"{k} {v:.1f}" for k, v in r.get("stakes_cm", {}).items())
        if st:
            L.append(f"| {r['ep']} | | stakes | {st} | | | | | | |")
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", required=True, help="object directories (raw MoGe frame)")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--body", default=os.path.join(WORK, "body_v2"))
    ap.add_argument("--ratio", default=os.path.join(WORK, "moge_to_body_ratio_floor.json"))
    ap.add_argument("--sdf", default=os.path.join(WORK, "objects_v4", "sdf"), help="budgeted meshes")
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = a.names or [Path(p).name.replace("objects_", "") for p in a.objects]
    dirs = dict(zip(names, a.objects))
    out = Path(a.out)
    for sub in ("overlays", "series"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    scored = scored_frames()
    ratio = json.load(open(a.ratio))["ratio"]
    bc = BodyCache(a.body)
    jp = out / "audit.json"
    rows = json.load(open(jp)) if jp.exists() else {}
    for ep in a.episodes:
        rows[str(ep)] = audit_episode(ep, dirs, meta, scored, ratio, bc, a.sdf, str(out))
        o = rows[str(ep)]
        print(f"ep {ep:2d} {o['cls']:18s} " + " | ".join(
            f"{k}: iou {v['iou_scored'] or 0:.2f} kz {v['kz_static'] or 0:.3f}/{v['kz_held'] or 0:.3f} gap {v['gap_held_cm'] or 0:.1f} "
            f"k* {v['kstar_held'] or 0:.3f} h0 {v['h_f0_cm']:.0f}" for k, v in o["src"].items())
            + f" | stakes {o['stakes_cm']} ({o['time_s']:.0f}s)", flush=True)
        json.dump(rows, open(jp, "w"), indent=1)
    allrows = [rows[k] for k in sorted(rows, key=int)]
    write_md(allrows, names, out / "audit.md")
    print("wrote", out / "audit.md")


if __name__ == "__main__":
    main()
