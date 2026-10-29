"""objects_v6: per-episode composition of the v4 / v5 object trajectories with targeted corrections.

The audit (audit_objects.py) shows where each trajectory is physically inconsistent. v6 keeps the
mesh and, per episode, follows a recipe (RECIPES below, every change is logged with its reason):

  base       directory giving the default trajectory and the episode scale s
  scale      optional multiplier on s (the object's size in the body frame)
  static     directory used on static (resting) frames, e.g. v5 where the v4 support was wrong
  held       directory used on held frames (object moving with the hands), e.g. v4 where its hand
             contact is right and v5 moved the object off the hands
  anchor     'hands': on held frames move the object along its viewing ray (fixed size) to the
             depth that closes the hand-object gap (median-filtered, clipped)
  flip       optional per-frame-range rotation fix (rotation applied in mesh coordinates)

Frames are labelled static (seg_id >= 0 of the base track), held (v4 'held' mask) or free (the
rest; they follow the nearest labelled neighbour through the smooth weights). Positions are taken
from the chosen directory as they are (absolute, raw MoGe frame); switching directories is blended
with Gaussian weights (rotations by normalised quaternion blending, translations linearly) so
ACC-O does not see a jump. Output layout = objects_v4 (episode_XXXXXX_{object.ply,pose.json,
track.npz}), plus changes.json.

    python compose_objects_v6.py --out work/objects_v6        (recipes: recipes_v6.json)
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, load_meta  # noqa: E402

SRC = {"v4": os.path.join(WORK, "objects_v4"), "v5": os.path.join(WORK, "objects_v5")}

# Per-episode recipes with their reasons (audit_objects.py evidence); episodes not listed keep v4.
RECIPES_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes_v6.json")


def load(src, ep):
    d = dict(np.load(os.path.join(SRC[src], f"episode_{ep:06d}_track.npz")))
    d["scale"] = float(d["scale"])
    return d


def blend_rotations(Ra, Rb, w):
    """Per-frame weighted rotation: w=0 -> Ra, w=1 -> Rb (quaternion blend, sign-aligned)."""
    qa, qb = Rotation.from_matrix(Ra).as_quat(), Rotation.from_matrix(Rb).as_quat()
    sgn = np.sign((qa * qb).sum(1, keepdims=True))
    sgn[sgn == 0] = 1
    q = (1 - w)[:, None] * qa + w[:, None] * qb * sgn
    return Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()


def smooth_weight(mask, sigma):
    w = gaussian_filter1d(mask.astype(float), sigma, mode="nearest") if sigma > 0 else mask.astype(float)
    return np.clip(w, 0, 1)


def hand_depth(ep, R, t, s, ratio, body_dir, held, kmin=0.85, kmax=1.25, win=9):
    """Per-frame scaling of t (about the camera, size fixed) that closes the hand-object gap."""
    import torch
    from fix_penetration import hand_vertex_ids
    from mhr_utils import DEFAULT_MHR, MHRForward
    from audit_objects import budget_mesh, sample_surface
    mhr = MHRForward(DEFAULT_MHR, device="cuda")
    hid = hand_vertex_ids(mhr.model)
    b = np.load(os.path.join(body_dir, f"episode_{ep:06d}.npz"))
    T = len(t)
    mv, mf = budget_mesh(ep, os.path.join(WORK, "objects_v4", "sdf"))
    P = sample_surface(mv, mf, 800, seed=ep)
    ks = np.exp(np.linspace(np.log(kmin), np.log(kmax), 41))
    kstar = np.full(T, np.nan)
    fr = np.flatnonzero(held[:T])
    for a in range(0, len(fr), 128):
        idx = fr[a:a + 128]
        v, _ = mhr(b["pose"][idx], b["scales"], b["shape"])
        Hh = torch.tensor(v[:, hid][:, ::3] * ratio, device="cuda", dtype=torch.float32)  # MoGe frame
        for i, f in enumerate(idx):
            O = s * P @ R[f].T  # object points relative to t
            # moving t by k: points = O + k t
            Q = torch.tensor(O[None] + ks[:, None, None] * t[f][None, None], device="cuda", dtype=torch.float32)
            g = torch.cdist(Q.reshape(-1, 3), Hh[i]).min(1).values.reshape(len(ks), -1).min(1).values.cpu().numpy()
            ok = np.flatnonzero(g <= min(g.min() + 0.01, 0.03))
            if len(ok) and ok.min() > 0 and ok.max() < len(ks) - 1:  # closest to 1 that closes the gap
                kstar[f] = ks[ok[np.argmin(np.abs(np.log(ks[ok])))]]
    ok = np.isfinite(kstar)
    k = np.ones(T)
    if ok.sum() >= 5:
        k_f = np.interp(np.arange(T), np.flatnonzero(ok), kstar[ok])
        k = median_filter(k_f, size=win, mode="nearest")
    return k, kstar


def reorient_segment(ep, R, t, s, frames, n, step=10.0, min_gain=0.05, up_mesh=None, floor_d=None):
    """Rotation about the floor normal (+ a shift along the floor) of a resting object, searched by
    silhouette IoU over the segment. With up_mesh (furniture), the object is first stood upright
    (its mesh up axis turned onto the floor normal) and every hypothesis is put back on the floor
    (lowest point on the plane n.x = floor_d, MoGe frame). Returns the world rotation Rw (about the
    pivot), the shift, the IoU before / after and whether the gain reaches min_gain.""" 
    import cv2
    from audit_objects import K_FULL, budget_mesh, load_masks, quarter, sample_surface
    mv, mf = budget_mesh(ep, os.path.join(WORK, "objects_v4", "sdf"))
    om, pm, score = load_masks(ep)
    omq, pmq = quarter(om), quarter(pm)
    K4 = K_FULL.copy(); K4[:2] /= 4
    H, W = omq.shape[1:]
    fr = [f for f in frames[np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)] if omq[f].sum() > 8]
    if not fr:
        return None
    f0 = fr[len(fr) // 2]
    pivot = (s * mv @ R[f0].T + t[f0]).mean(0)  # noqa: (budget mesh vertices)
    up = n / np.linalg.norm(n)
    e1 = np.cross(up, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    P = sample_surface(mv, mf, 6000, seed=ep)
    Rpre = np.eye(3)
    if up_mesh is not None:
        a_ = R[f0] @ up_mesh
        ax = np.cross(a_, up)
        ang = np.arctan2(np.linalg.norm(ax), a_ @ up)
        if np.linalg.norm(ax) > 1e-9:
            Rpre = Rotation.from_rotvec(ax / np.linalg.norm(ax) * ang).as_matrix()
    P0 = s * P @ R[f0].T + t[f0] - pivot
    ker = np.ones((3, 3), np.uint8)
    keeps = {f: ~(pmq[f] & ~omq[f]) for f in fr}

    def sil_pts(Vc):
        """Fast silhouette: splat dense surface points, close 1-px gaps."""
        uv = np.round(K4[:2, :2] @ (Vc[:, :2] / Vc[:, 2:3]).T + K4[:2, 2:3]).astype(int)
        ok = (uv[0] >= 0) & (uv[0] < W) & (uv[1] >= 0) & (uv[1] < H) & (Vc[:, 2] > 0.05)
        m = np.zeros((H, W), np.uint8)
        m[uv[1, ok], uv[0, ok]] = 1
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker) > 0

    def world(th, dx, dz):
        Rw = Rotation.from_rotvec(np.radians(th) * up).as_matrix() @ Rpre
        sh = dx * e1 + dz * e2
        if floor_d is not None and up_mesh is not None:
            low = ((P0 @ Rw.T + pivot + sh) @ up).min()
            sh = sh + (floor_d - low) * up
        return Rw, sh

    def score_h(th, dx, dz):
        Rz, sh = world(th, dx, dz)
        tot = 0.0
        for f in fr:
            Vc = (s * P @ R[f].T + t[f] - pivot) @ Rz.T + pivot + sh
            sil = sil_pts(Vc)
            keep = keeps[f]
            tot += (sil & omq[f] & keep).sum() / max(((sil | omq[f]) & keep).sum(), 1)
        return tot / len(fr)
    base = score_h(0, 0, 0) if up_mesh is None else None
    if base is None:  # upright mode: compare with the pose as it is
        Rpre_, Rpre = Rpre, np.eye(3)
        fd_, floor_d = floor_d, None
        base = score_h(0, 0, 0)
        Rpre, floor_d = Rpre_, fd_
    best = (-1.0, 0.0, 0.0, 0.0)
    for th in np.arange(0, 360, step):
        for dx in np.arange(-0.12, 0.121, 0.04):
            for dz in np.arange(-0.12, 0.121, 0.04):
                sc = score_h(th, dx, dz)
                if sc > best[0]:
                    best = (sc, th, dx, dz)
    sc, th, dx, dz = best
    for _ in range(2):  # local refinement
        for dth in (-step / 2, step / 2):
            for ddx in (-0.02, 0, 0.02):
                for ddz in (-0.02, 0, 0.02):
                    c = score_h(th + dth, dx + ddx, dz + ddz)
                    if c > sc:
                        sc, th, dx, dz = c, th + dth, dx + ddx, dz + ddz
        step /= 2
    Rw, sh = world(th, dx, dz)
    return dict(Rz=Rw, pivot=pivot, shift=sh, theta=float(th), iou_before=float(base),
                iou_after=float(sc), accept=bool(sc - base >= min_gain))


def compose(ep, rec, meta, ratio, body_dir, log):
    base = load(rec.get("base", "v4"), ep)
    T = len(base["t"])
    s = base["scale"] * rec.get("scale", 1.0)
    R, t = base["R"].copy(), base["t"].copy()
    seg = base["seg_id"]
    held = load("v4", ep)["held"].astype(bool)[:T]
    static = seg >= 0
    sig = rec.get("sigma", 6.0)
    notes = []
    for label, mask in (("static", static), ("held", held)):
        src = rec.get(label)
        if src is None or src == rec.get("base", "v4"):
            continue
        o = load(src, ep)
        w = smooth_weight(mask, sig)
        R = blend_rotations(R, o["R"][:T], w)
        t = (1 - w)[:, None] * t + w[:, None] * o["t"][:T]
        notes.append(f"{label} frames ({int(mask.sum())}) from {src}")
    if rec.get("anchor") == "hands":
        k, kstar = hand_depth(ep, R, t, s, ratio, body_dir, held)
        w = smooth_weight(held, sig)
        kk = 1 + w * (k - 1)
        t = t * kk[:, None]
        notes.append(f"hand-anchored depth on held frames: k median {np.median(k[held]) if held.any() else 1:.3f} "
                     f"(range {kk.min():.3f}-{kk.max():.3f}, {int(np.isfinite(kstar).sum())} contact frames)")
    if rec.get("reorient"):
        n_cam = np.asarray(np.load(os.path.join(body_dir, f"episode_{ep:06d}.npz"))["floor_n"], float)
        up_mesh = None
        if rec.get("upright"):
            from audit_objects import UPRIGHT_REF
            e_ref, f_ref = UPRIGHT_REF[meta[ep]["object"]]
            n_ref = np.asarray(np.load(os.path.join(body_dir, f"episode_{e_ref:06d}.npz"))["floor_n"], float)
            up_mesh = load("v4", e_ref)["R"][f_ref].T @ n_ref
            up_mesh /= np.linalg.norm(up_mesh)
        d_body = float(np.load(os.path.join(body_dir, f"episode_{ep:06d}.npz"))["floor_d"])
        for (a, b) in rec["reorient"]:
            fr = np.arange(a, min(b + 1, T))
            r_ = reorient_segment(ep, R, t, s, fr, n_cam, up_mesh=up_mesh, floor_d=d_body * ratio,
                                  min_gain=rec.get("min_gain", 0.05))
            if r_ is None:
                continue
            msg = (f"reorient {a}-{b}: yaw {r_['theta']:.0f} deg about the floor normal, shift "
                   f"{np.linalg.norm(r_['shift']) * 100:.0f} cm, IoU {r_['iou_before']:.3f} -> {r_['iou_after']:.3f}")
            if r_["accept"]:
                m = np.zeros(T, bool); m[fr] = True
                w = smooth_weight(m, 2.0)
                w[fr] = 1.0
                R2 = np.einsum("ij,tjk->tik", r_["Rz"], R)
                t2 = (t - r_["pivot"]) @ r_["Rz"].T + r_["pivot"] + r_["shift"]
                R = blend_rotations(R, R2, w)
                t = (1 - w)[:, None] * t + w[:, None] * t2
                notes.append(msg + " (applied)")
            else:
                notes.append(msg + " (rejected: gain < 0.05)")
    for (a, b, rotvec) in rec.get("flip", []):
        Rm = Rotation.from_rotvec(rotvec).as_matrix()
        R[a:b] = R[a:b] @ Rm
        notes.append(f"rotation {rotvec} (mesh frame) on frames {a}-{b}")
    return dict(R=R, t=t, s=s, seg=seg, held=held, notes=notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WORK, "objects_v6"))
    ap.add_argument("--recipes", default=RECIPES_JSON, help="JSON file with per-episode recipes")
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--body", default=os.path.join(WORK, "body_v2"))
    ap.add_argument("--ratio", default=os.path.join(WORK, "moge_to_body_ratio_floor.json"))
    a = ap.parse_args()
    recipes = {int(k): v for k, v in json.load(open(a.recipes)).items()}
    meta = load_meta()
    ratio = json.load(open(a.ratio))["ratio"]
    os.makedirs(a.out, exist_ok=True)
    cpath = os.path.join(a.out, "changes.json")
    changes = json.load(open(cpath)) if os.path.exists(cpath) else {}
    for ep in a.episodes:
        rec = recipes.get(ep, {"base": "v4"})
        out = compose(ep, rec, meta, ratio[str(ep)], a.body, print)
        stem = os.path.join(a.out, f"episode_{ep:06d}")
        shutil.copy(os.path.join(SRC["v4"], f"episode_{ep:06d}_object.ply"), stem + "_object.ply")
        T = len(out["t"])
        np.savez(stem + "_track.npz", R=out["R"], t=out["t"], valid=np.ones(T, bool), scale=out["s"],
                 seg_id=out["seg"], held=out["held"])
        ref = int(T // 2)
        json.dump(dict(episode=ep, object=meta[ep]["object"], frame=ref,
                       convention="point_cam = s * R @ point_mesh + t; camera frame OpenCV; metres (raw MoGe-2 "
                                  "frame); point_mesh = vertices of episode_XXXXXX_object.ply",
                       R=out["R"][ref].tolist(), t=out["t"][ref].tolist(), s=out["s"], recipe=rec),
                  open(stem + "_pose.json", "w"), indent=1)
        changes[str(ep)] = dict(cls=meta[ep]["object"], recipe=rec, notes=out["notes"],
                                reason=rec.get("reason", "unchanged (v4)"), scale=out["s"],
                                scale_body=out["s"] / ratio[str(ep)])
        print(f"ep {ep:2d} {meta[ep]['object']:18s} {rec.get('reason', 'v4')[:60]} | " + "; ".join(out["notes"]), flush=True)
    json.dump(changes, open(cpath, "w"), indent=1)


if __name__ == "__main__":
    main()
