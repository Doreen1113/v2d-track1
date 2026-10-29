"""Diagnostics for the WiLoR hand transfer (transfer_hand_wilor.py).

stats  : on the scored frames that have confident RTMPose hand keypoints (work/hand_kp2d), the 2D finger error of
         (a) WiLoR's own projected keypoints on the frames its gate accepted, (b) the base body run and (c) the
         transferred run, on exactly those frames/hands; wrist-aligned variants (prediction shifted so its wrist
         sits on the RTMPose wrist) isolate the articulation / orientation part from the wrist position.
cdh    : how far the transferred run is from the base run under the CD-H metric (Sim(3) on the first scored
         frame, mean symmetric Chamfer over up to 120 scored frames, all vertices, cm) and the mean displacement of
         the vertices that move (> 1 mm): the size of the change, not its sign (no ground truth)
render : hand close-ups (crop x4) with the base run (red), transferred run (green), WiLoR 2D (blue) and RTMPose
         (yellow) keypoints, for the given episode / frames, into one PNG per episode.

  python eval_wilor_hands.py stats --base WORK/body_vA --new WORK/hand_vA --sigma-h 3
  python eval_wilor_hands.py render --base WORK/body_vA --new WORK/hand_vA --episodes 4 11 --n 8 --out DIR
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_body_proxies import assembled_pose  # noqa: E402
from mhr_utils import MHRForward  # noqa: E402
from transfer_hand_wilor import HAND_J, PARENT  # noqa: E402

ROOT = "/work/doreen071/v2d"
WORK = f"{ROOT}/work"
VIDEO = f"{ROOT}/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{{:06d}}.mp4"


def hand_uv(mhr, d, ep, frames, sigma_h, K):
    b = np.load(os.path.join(d, f"episode_{ep:06d}.npz"))
    pose = assembled_pose(b, sigma_h)
    _, J = mhr(pose[frames], b["scales"], b["shape"])
    kp = np.stack([J[:, HAND_J[s]] for s in (0, 1)], 1)  # [N,2(L,R),21,3]
    return np.stack([K[0] * kp[..., 0] / kp[..., 2] + K[2], K[1] * kp[..., 1] / kp[..., 2] + K[3]], -1)


def wilor_uv(new, ep, T):
    """WiLoR 2D keypoints [T,2,21,2] on the accepted (frame, side) pairs, NaN elsewhere."""
    g = np.load(os.path.join(new, "diag", f"episode_{ep:06d}_gate.npz"))
    args = json.load(open(os.path.join(new, "diag", "args.json")))
    wl = np.load(os.path.join(args["wilor"], f"episode_{ep:06d}.npz"))
    out = np.full((T, 2, 21, 2), np.nan)
    for s in (0, 1):
        ok = (g["w"][:, s] > 0) & (g["det"][:, s] >= 0)
        out[ok, s] = wl["kp2d"][g["det"][ok, s], s]
    return out, g["w"]


def stats(a):
    mhr = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu")
    rows = {}
    for ep in a.episodes:
        p = os.path.join(WORK, "hand_kp2d", f"episode_{ep:06d}.npz")
        if not os.path.exists(p):
            continue
        h = np.load(p)
        fr = h["frames"]
        if len(fr) == 0:
            continue
        raw = np.load(os.path.join(WORK, "body", f"episode_{ep:06d}.npz"))
        K = np.array([float(raw["fx"]), float(raw["fy"]), float(raw["cx"]), float(raw["cy"])])
        det = np.stack([h["left"], h["right"]], 1)  # [N,2,21,3] (L,R) like HAND_J
        conf = det[..., 2] > a.thr
        conf[..., 0] &= True
        T = len(np.load(os.path.join(a.base, f"episode_{ep:06d}.npz"))["pose"])
        wu, w = wilor_uv(a.new, ep, T)
        wu = wu[fr]
        acc = w[fr] > 0  # [N,2]
        preds = dict(base=hand_uv(mhr, a.base, ep, fr, a.sigma_h, K), new=hand_uv(mhr, a.new, ep, fr, a.sigma_h, K),
                     wilor=wu)
        fing = conf.copy()
        fing[..., 0] = False
        r = {}
        for name, uv in preds.items():
            e = np.linalg.norm(uv - det[..., :2], axis=-1)
            uva = uv - uv[:, :, :1] + det[:, :, :1, :2]  # wrist-aligned
            ea = np.linalg.norm(uva - det[..., :2], axis=-1)
            m_all = fing
            m_acc = fing & acc[..., None] & conf[..., :1]
            r[name] = dict(all=float(np.nanmean(e[m_all])) if name != "wilor" else float("nan"),
                           acc=float(np.nanmean(e[m_acc])) if m_acc.any() else float("nan"),
                           acc_wa=float(np.nanmean(ea[m_acc])) if m_acc.any() else float("nan"))
        r["n_acc_hands"] = int(acc.sum())
        r["n_hands"] = int(acc.size)
        rows[ep] = r
        print(f"ep {ep:2d} acc {r['n_acc_hands']:4d}/{r['n_hands']:4d} | all base {r['base']['all']:.2f} new "
              f"{r['new']['all']:.2f} | accepted base {r['base']['acc']:.2f} new {r['new']['acc']:.2f} wilor "
              f"{r['wilor']['acc']:.2f} | wrist-aligned base {r['base']['acc_wa']:.2f} new {r['new']['acc_wa']:.2f} "
              f"wilor {r['wilor']['acc_wa']:.2f}", flush=True)
    keys = [("base", "all"), ("new", "all"), ("base", "acc"), ("new", "acc"), ("wilor", "acc"), ("base", "acc_wa"),
            ("new", "acc_wa"), ("wilor", "acc_wa")]
    print("mean over episodes: " + "  ".join(f"{k}/{m} {np.nanmean([rows[e][k][m] for e in rows]):.3f}"
                                             for k, m in keys))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)


def cdh(a):
    from scipy.spatial import cKDTree
    from sim_cdh_sensitivity import sim3_apply
    mhr = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu")
    scored = json.load(open(os.path.join(WORK, "body_v2/scored_frames.json")))
    rows = {}
    for ep in a.episodes:
        sc = np.array(scored[str(ep)])
        sel = np.unique(np.r_[sc[0], sc[np.linspace(0, len(sc) - 1, min(120, len(sc))).round().astype(int)]])
        V = []
        for d in (a.base, a.new):
            b = np.load(os.path.join(d, f"episode_{ep:06d}.npz"))
            V.append(mhr(assembled_pose(b, a.sigma_h)[sel], b["scales"], b["shape"])[0])
        gt, pr = V
        p = sim3_apply(pr[0], gt[0], pr.reshape(-1, 3)).reshape(pr.shape)
        ch = []
        for t in range(len(sel)):
            ch.append(cKDTree(gt[t]).query(p[t])[0].mean() + cKDTree(p[t]).query(gt[t])[0].mean())
        disp = np.linalg.norm(pr - gt, axis=-1)
        mv = disp > 1e-3
        rows[ep] = dict(cdh_cm=float(np.mean(ch) * 100), moved_frac=float(mv.mean()),
                        moved_disp_cm=float(disp[mv].mean() * 100) if mv.any() else 0.0)
        print(f"ep {ep:2d} cdh(base,new) {rows[ep]['cdh_cm']:.3f} cm  moved {rows[ep]['moved_frac'] * 100:.1f}% of "
              f"verts by {rows[ep]['moved_disp_cm']:.2f} cm", flush=True)
    print("mean: " + "  ".join(f"{k} {np.mean([r[k] for r in rows.values()]):.3f}" for k in next(iter(rows.values()))))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


def draw(img, uv, color, r=2):
    for c in range(1, 21):
        p, q = uv[PARENT[c]], uv[c]
        if np.isfinite(p).all() and np.isfinite(q).all():
            cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), color, 1, cv2.LINE_AA)
    for q in uv:
        if np.isfinite(q).all():
            cv2.circle(img, tuple(int(v) for v in q), r, color, -1, cv2.LINE_AA)


def render(a):
    mhr = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    Z = 4
    for ep in a.episodes:
        raw = np.load(os.path.join(WORK, "body", f"episode_{ep:06d}.npz"))
        K = np.array([float(raw["fx"]), float(raw["fy"]), float(raw["cx"]), float(raw["cy"])])
        T = len(raw["valid"])
        wu, w = wilor_uv(a.new, ep, T)
        if a.frames:
            frames = np.array(a.frames)
        else:  # frames with both a WiLoR hand and the largest change between the runs
            b0 = np.load(os.path.join(a.base, f"episode_{ep:06d}.npz"))["pose"]
            b1 = np.load(os.path.join(a.new, f"episode_{ep:06d}.npz"))["pose"]
            ch = np.abs(b1[:, 68:122] - b0[:, 68:122]).sum(1) * (w.max(1) > 0)
            sc = json.load(open(os.path.join(WORK, "body_v2/scored_frames.json")))[str(ep)]
            m = np.zeros(T, bool)
            m[[f for f in sc if f < T]] = True
            if a.contact:  # hand-object contact: base hand points on the (dilated) object mask
                from scipy.ndimage import binary_dilation
                from common import unpack_masks
                mk = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
                mh, mw = int(mk["mask_h"]), int(mk["mask_w"])
                cand = np.flatnonzero(m & (ch > 0))
                uvc = hand_uv(mhr, a.base, ep, cand, a.sigma_h, K) * (mh / float(raw["height"]))
                for k, f in enumerate(cand):
                    O = binary_dilation(unpack_masks(mk["obj_masks"], mh, mw, f), iterations=4)
                    q = np.clip(uvc[k].round().astype(int), 0, [mw - 1, mh - 1])
                    if O[q[..., 1], q[..., 0]].mean(-1).max() < 0.3:
                        m[f] = False
            ch = np.where(m, ch, -1)
            frames = np.sort(np.argsort(-ch)[:a.n])
        ub = hand_uv(mhr, a.base, ep, frames, a.sigma_h, K)
        un = hand_uv(mhr, a.new, ep, frames, a.sigma_h, K)
        rtm = {}
        p = os.path.join(WORK, "hand_kp2d", f"episode_{ep:06d}.npz")
        if os.path.exists(p):
            h = np.load(p)
            for k, f in enumerate(h["frames"]):
                rtm[int(f)] = np.stack([h["left"][k], h["right"][k]])
        cap = cv2.VideoCapture(VIDEO.format(ep))
        tiles = []
        for k, f in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if not ok:
                continue
            row = []
            for s in (1, 0):
                c = np.nanmean(np.concatenate([ub[k, s], un[k, s]]), 0)
                half = 48
                x0, y0 = int(c[0]) - half, int(c[1]) - half
                crop = cv2.copyMakeBorder(img, half * 2, half * 2, half * 2, half * 2, cv2.BORDER_CONSTANT)
                crop = crop[y0 + 2 * half:y0 + 4 * half, x0 + 2 * half:x0 + 4 * half]
                crop = cv2.resize(crop, None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC)
                off = np.array([x0, y0])
                draw(crop, (ub[k, s] - off) * Z, (0, 0, 255))
                draw(crop, (un[k, s] - off) * Z, (0, 255, 0))
                draw(crop, (wu[f, s] - off) * Z, (255, 128, 0))
                if int(f) in rtm:
                    q = rtm[int(f)][s]
                    for j in range(21):
                        if q[j, 2] > 0.25:
                            cv2.circle(crop, tuple(int(v) for v in (q[j, :2] - off) * Z), 3, (0, 255, 255), 1)
                cv2.putText(crop, f"ep{ep} f{f} {'R' if s else 'L'} w={w[f, s]:.0f}", (4, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                row.append(crop)
            tiles.append(np.concatenate(row, 1))
        cap.release()
        if tiles:
            grid = [np.concatenate(tiles[i:i + 2], 1) if i + 1 < len(tiles) else
                    np.concatenate([tiles[i], np.zeros_like(tiles[i])], 1) for i in range(0, len(tiles), 2)]
            cv2.imwrite(os.path.join(a.out, f"episode_{ep:06d}.jpg"), np.concatenate(grid, 0),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"ep {ep}: frames {frames.tolist()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stats", "render", "cdh"])
    ap.add_argument("--base", default=f"{WORK}/body_vA")
    ap.add_argument("--new", required=True)
    ap.add_argument("--sigma-h", type=float, default=3.0)
    ap.add_argument("--thr", type=float, default=0.25)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--contact", action="store_true", help="render: only frames with a hand on the object mask")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dict(stats=stats, render=render, cdh=cdh)[a.mode](a)


if __name__ == "__main__":
    main()
