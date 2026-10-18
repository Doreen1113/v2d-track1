"""Hand-object consistency diagnostics of assembled Track 1 episodes (body frame).

Per episode, on the scored frames:
  * scorer penetration of the hand vertices into the budgeted object mesh (cm, body frame),
  * a scan over a scale k about the camera (object t and s multiplied by k; silhouettes are
    unchanged): mean penetration and median hand-surface gap on held frames,
  * hand depth vs MoGe depth at the hand pixels (body z / (MoGe z / ratio)),
and an overlay sheet (image with body / hands / object, top and side views) of the frames with
the deepest penetration.

    python diagnose_object_contact.py --episodes-dir EPS --objects OBJ --ratio R.json --out DIR [--episodes ...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contactlib as cl  # noqa: E402
from common import VIDEO_TMPL, WORK, load_intrinsics, read_frame, unpack_masks  # noqa: E402
from mhr_utils import DEFAULT_MHR, MHRForward  # noqa: E402


def held_frames(ep, obj_dir, T, min_speed=0.004):
    """Frames where the object moves (not in a static segment) and its mask touches the actor."""
    tr = np.load(obj_dir / f"episode_{ep:06d}_track.npz")
    md = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
    h, w = int(md["mask_h"]), int(md["mask_w"])
    om = unpack_masks(md["obj_masks"], h, w)
    pm = unpack_masks(md["person_masks"], h, w)
    from scipy.ndimage import gaussian_filter1d
    t = gaussian_filter1d(tr["t"], 2, axis=0)
    speed = np.linalg.norm(np.gradient(t, axis=0), axis=1)
    ker = np.ones((11, 11), np.uint8)
    out = np.zeros(T, bool)
    for f in range(min(T, len(om))):
        if tr["seg_id"][f] < 0 and speed[f] > min_speed:
            out[f] = (cv2.dilate(pm[f].astype(np.uint8), ker) > 0)[om[f]].any()
    return out, om, pm


def project(P, K):
    z = np.clip(P[..., 2], 1e-3, None)
    return np.stack([K[0, 0] * P[..., 0] / z + K[0, 2], K[1, 1] * P[..., 1] / z + K[1, 2]], -1)


def hand_moge_ratio(ep, verts, hand_ids, frames, ratio, om, pm):
    """median(body hand z / (MoGe z / ratio)) at visible hand pixels (quarter-res depth)."""
    D = np.load(os.path.join(WORK, "depth", f"episode_{ep:06d}.npz"))["depth"]
    K, _ = load_intrinsics(ep)
    Kq = K.copy(); Kq[:2] /= 4
    H, W = D.shape[1:]
    out = []
    for i, f in enumerate(frames):
        hv = verts[i, hand_ids]
        uv = np.round(project(hv, Kq)).astype(int)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if ok.sum() < 20:
            continue
        uv, hz = uv[ok], hv[ok, 2]
        pmq = cv2.resize(pm[f].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        omq = cv2.resize(om[f].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        sel = pmq[uv[:, 1], uv[:, 0]] & ~omq[uv[:, 1], uv[:, 0]]
        if sel.sum() < 20:
            continue
        # front-most hand points per pixel approximate the visible surface
        dz = D[f][uv[sel, 1], uv[sel, 0]].astype(float) / ratio
        hzs = hz[sel]
        front = hzs <= np.percentile(hzs, 30)
        out.append(np.median(hzs[front] / np.clip(dz[front], 1e-3, None)))
    return float(np.median(out)) if out else float("nan"), len(out)


def overlay(ep, f, verts, hand_ids, Vc, F, K, om, title):
    import objlib
    im = read_frame(VIDEO_TMPL.format(ep), int(f))
    im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2))
    h, w = im.shape[:2]
    Kh = K.copy(); Kh[:2] /= 2
    sil = objlib.hard_silhouette(Vc, F, Kh, h, w)
    im[sil] = (0.45 * im[sil] + [0, 140, 0]).astype(np.uint8)
    uv = project(verts[::7], Kh).astype(int)
    for x, y in uv:
        if 0 <= x < w and 0 <= y < h:
            im[y, x] = (255, 255, 255)
    uvh = project(verts[hand_ids], Kh).astype(int)
    for x, y in uvh:
        if 0 <= x < w and 0 <= y < h:
            im[y, x] = (255, 0, 0)
    cnt, _ = cv2.findContours(om[f].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, cnt, -1, (0, 120, 255), 1)
    ys, xs = np.nonzero(sil | om[f])
    allx = np.concatenate([xs, uv[:, 0]]); ally = np.concatenate([ys, uv[:, 1]])
    x0, x1 = np.clip([allx.min() - 20, allx.max() + 20], 0, w); y0, y1 = np.clip([ally.min() - 20, ally.max() + 20], 0, h)
    crop = im[y0:y1, x0:x1]
    sc = 360 / max(crop.shape[0], 1)
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), 360))
    # top (x, z) and side (z, -y) views
    views = []
    for a, b in ((0, 2), (2, 1)):
        canvas = np.full((360, 360, 3), 255, np.uint8)
        pts = [verts[::5], verts[hand_ids], Vc]
        allp = np.concatenate(pts)
        ca, cb = allp[:, a], allp[:, b]
        lo = np.array([ca.min(), cb.min()]); span = max(np.ptp(ca), np.ptp(cb)) * 1.1 + 1e-6
        for P, col in zip(pts, ((150, 150, 150), (255, 0, 0), (0, 160, 0))):
            q = ((np.stack([P[:, a], P[:, b]], 1) - lo) / span * 340 + 10).astype(int)
            for x, y in q[:: max(1, len(q) // 3000)]:
                canvas[np.clip(y, 0, 359), np.clip(x, 0, 359)] = col
        cv2.putText(canvas, "top x-z" if a == 0 else "side z-y", (5, 15), 0, 0.45, (0, 0, 0), 1)
        views.append(canvas)
    row = np.hstack([crop] + views)
    cv2.putText(row, title, (5, 352), 0, 0.5, (255, 255, 0), 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-dir", type=Path, required=True)
    ap.add_argument("--objects", type=Path, required=True, help="object dir with _track.npz (MoGe frame)")
    ap.add_argument("--ratio", type=Path, default=Path(WORK) / "moge_to_body_ratio_floor.json")
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-vis", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "vis").mkdir(exist_ok=True)
    ratio = json.loads(a.ratio.read_text())["ratio"]
    mhr = MHRForward(DEFAULT_MHR, device="cuda" if torch.cuda.is_available() else "cpu")
    hand_ids = cl.hand_ids_from(mhr.model)
    scored = cl.scored()
    ks = np.exp(np.linspace(np.log(0.8), np.log(1.3), 27))
    for ep in a.episodes:
        p = a.episodes_dir / f"episode_{ep:06d}.npz"
        d = dict(np.load(p))
        mesh = next(q for s in (".ply", ".obj", ".glb") if (q := p.with_name(p.stem + "_object" + s)).exists())
        mv, mf = cl.budget(mesh)
        sdf = cl.SDFGrid(mv, mf)
        fr = scored[ep]
        T = len(d["pose"])
        verts, _ = mhr(d["pose"][fr], d["scales"], d["shape"])
        hands = verts[:, hand_ids]
        R, t, s = d["object_rotation"][fr], d["object_translation"][fr], float(d["object_scale"])
        local = np.einsum("tpi,tij->tpj", hands - t[:, None], R) / s
        dep = cl.penetration_depth(local.reshape(-1, 3), mv, mf).reshape(local.shape[:2]) * s
        pen_frame = dep.mean(1) * 100
        held, om, pm = held_frames(ep, a.objects, T)
        hf = held[fr]
        # k scan on the SDF grid
        scan = []
        for k in ks:
            lk = np.einsum("tpi,tij->tpj", hands - k * t[:, None], R) / (k * s)
            sd = sdf.query(lk) * k * s
            pen = np.clip(-sd, 0, None).mean(1).mean() * 100
            maxpen = np.clip(-sd, 0, None).max(1)
            gap = np.clip(sd, 0, None).min(1)
            g = np.median((gap + maxpen)[hf]) * 100 if hf.any() else np.nan
            scan.append((float(k), float(pen), float(g), float(np.median(gap[hf]) * 100) if hf.any() else np.nan,
                         float(np.median(maxpen[hf]) * 100) if hf.any() else np.nan))
        scan = np.array(scan)
        if hf.any():
            kb = float(scan[np.nanargmin(scan[:, 2]), 0])
        else:
            kb = float("nan")
        sub = np.linspace(0, len(fr) - 1, min(40, len(fr))).astype(int)
        hr, nh = hand_moge_ratio(ep, verts[sub], hand_ids, fr[sub], ratio[str(ep)], om, pm)
        sdf_now = sdf.query(local) * s
        res = dict(ep=ep, pen_cm=float(pen_frame.mean()), pen_frames=float((dep > 0).any(1).mean()),
                   maxdepth_p90_cm=float(np.percentile(dep.max(1), 90) * 100),
                   held_frac=float(hf.mean()),
                   held_gap_cm=float(np.median(np.clip(sdf_now, 0, None).min(1)[hf]) * 100) if hf.any() else None,
                   held_maxpen_cm=float(np.median(np.clip(-sdf_now, 0, None).max(1)[hf]) * 100) if hf.any() else None,
                   k_best=kb, hand_body_over_moge=hr, hand_ratio_frames=nh,
                   scan=scan.tolist(), s_body=s,
                   extent_body=(s * (mv.max(0) - mv.min(0))).tolist())
        print(f"[ep {ep}] pen {res['pen_cm']:.2f} cm frames {res['pen_frames']:.2f} p90max {res['maxdepth_p90_cm']:.1f} "
              f"held {res['held_frac']:.2f} gap {res['held_gap_cm']} maxpen {res['held_maxpen_cm']} k* {kb:.3f} "
              f"hand/moge {hr:.3f} ext {np.round(res['extent_body'], 2)}", flush=True)
        json.dump(res, open(a.out / f"episode_{ep:06d}.json", "w"), indent=1)
        if not a.no_vis:
            K, _ = load_intrinsics(ep)
            order = np.argsort(-dep.max(1))
            pick = list(order[:3])
            if hf.any():
                pick.append(int(np.flatnonzero(hf)[len(np.flatnonzero(hf)) // 2]))
            rows = []
            for i in pick:
                Vc = s * mv @ R[i].T + t[i]
                rows.append(overlay(ep, fr[i], verts[i], hand_ids, Vc, mf, K, om,
                                    f"ep{ep} f{fr[i]} maxpen {dep[i].max() * 100:.1f}cm held {int(hf[i])}"))
            wmax = max(r.shape[1] for r in rows)
            rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
            cv2.imwrite(str(a.out / "vis" / f"episode_{ep:06d}.jpg"), np.vstack(rows)[:, :, ::-1])


if __name__ == "__main__":
    main()
