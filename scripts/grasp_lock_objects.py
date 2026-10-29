"""objects_v7: grasp-locked object trajectories.

While an object is held it moves rigidly with the hand: T_obj(t) = T_hand(t) . G, with one constant
grasp transform G per grasp segment. The MHR body (body_v2, pose smoothed exactly as
assemble_episode.py does) is temporally smooth and metrically anchored to the floor, whereas the
per-frame object fit is noisy along the camera ray, flips on symmetric objects and drifts while the
body occludes it. Here G is estimated from all frames of a grasp segment and the held frames follow
the hand.

Per episode (objects_v6 track as input, raw MoGe frame; the body frame is MoGe / ratio):
  hands     MHR wrist frames r_wrist (joint 42) / l_wrist (joint 78) and the hand vertices (subtrees
            of r_wrist_twist 41 / l_wrist_twist 77), mapped to the MoGe frame (x ratio)
  contact   per frame and hand: projected hand vertices touch the object mask (10th percentile of
            the distance to the mask <= --touch-px, quarter resolution) and the hand is within
            --touch-cm of the object surface for some depth scaling k in [0.8, 1.25] about the camera
  segments  inside moving intervals (seg_id < 0), maximal runs where at least one hand stays in
            contact; a handover is cut in the middle of its two-hand period; with --split-pen > 0 a
            segment is also cut where the object centre jumps in the holding hand's frame (regrasp:
            piecewise-constant fit by dynamic programming, pieces >= --split-min frames); a hand of a
            two-hand segment is dropped when its median gap is > 10 cm and the other's <= 5 cm
  G         candidates per segment and hand frame (right, left, and for two-handed holds 'mid' =
            midpoint / mean wrist rotation, 'bar' = mid frame turned so a fixed axis follows the
            wrist-to-wrist vector): robust average of T_hand(t)^-1 T_obj(t) over frames with good IoU
            and depth agreement (rotation medoid, inliers within 30 deg, quaternion mean, median
            translation), and the static poses right before / after the segment (pick-up / place);
            the best by silhouette IoU is refined (soft silhouette + mask chamfer + MoGe depth, weak
            hand contact / penetration terms, prior to the start) over up to --refine-frames frames,
            rendered in a crop around the object.
            Grasp-point model ('<hand>_pt', unless --no-point): the wrist stays at a fixed point of
            the object frame, the rotation is the v6 one (t = p_wrist - s R q); a weaker rigid model
            that does not use the wrist rotation. A segment takes an accepted strict lock if any,
            else an accepted grasp-point lock, else keeps v6
  accept    on the trajectory as written (lock ramped in from the segment edges, see compose):
            mean IoU (visible frames) >= v6 - --tol-iou and median |kz - 1| <= v6 + --tol-kz
            (kz = rendered depth / MoGe depth on object pixels), and the lock keeps the object on the
            hand (median hand-surface gap <= --max-gap cm); otherwise the segment keeps v6
  compose   locked weight ramps from 0 at the segment edge (1 - exp(-d^2 / 2 sigma^2), d = frames
            to the nearest unlocked frame), adjacent locked segments cross-fade with Gaussian
            weights; static and unlocked frames are exactly v6
Episodes in EXCLUDE (contact is not a rigid grasp) are copied unchanged.

Outputs (--out, layout of objects_v6): episode_XXXXXX_{object.ply,pose.json,track.npz}, plus
logs/episode_XXXXXX.json (segments, hands, IoU / depth / hand gap before and after for every candidate,
G, acceleration on scored frames before and after) and summary.json / summary.md. The track npz gains
'locked' (bool per frame) and 'lock_id' (accepted segment index, -1 elsewhere).

    python grasp_lock_objects.py --src work/objects_v6 --out work/objects_v7 [--episodes ...]
    python grasp_lock_objects.py --recompose work/objects_v7 --out DIR --tol-iou X   (no refitting)
"""
import argparse
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter1d
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assemble_episode import fill_invalid, smooth, unwrap_euler  # noqa: E402
from common import WORK, load_meta  # noqa: E402
from contactlib import load_sdf  # noqa: E402
from mhr_utils import DEFAULT_MHR, MHRForward  # noqa: E402
from objlib import Episode, FrameTargets, Renderer, hard_silhouette, project_points, so3_exp  # noqa: E402

HAND_JOINT = {"right": 42, "left": 78}  # r_wrist / l_wrist
HAND_ROOT = {"right": 41, "left": 77}  # r_wrist_twist / l_wrist_twist (hand vertex subtrees)
EXCLUDE = {
    0: "hoop lifted over the body; the held interval 358-662 has no scored frame",
    1: "hoop swung / passed through (sliding grip)",
    2: "hoop swung as a jump rope (sliding grip)",
    4: "bowl pushed across the table",
    9: "desk tipped over on its edge by pushing with both hands",
    10: "desk pushed with the foot",
    15: "block turned over and over between the two hands (contact slides, rotation residual 55-110 deg)",
    16: "block turned over and over between the two hands (contact slides, rotation residual 70-90 deg)",
    22: "foam roll rolled on the table under the palms",
    24: "stool sat on, no grasp",
    26: "stool sat on, no grasp",
    29: "cart pushed / pulled",
}
SDF_DIR = os.path.join(WORK, "objects_v4", "sdf")


# ------------------------------------------------------------------------------------------ utils
def runs(mask):
    m = np.r_[0, np.asarray(mask, int), 0]
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def quat_mean(R, w=None):
    """Weighted chordal mean of rotations [N,3,3] (Markley)."""
    q = Rotation.from_matrix(R).as_quat()
    w = np.ones(len(q)) if w is None else np.asarray(w, float)
    M = (q * w[:, None]).T @ q
    _, vec = np.linalg.eigh(M)
    return Rotation.from_quat(vec[:, -1]).as_matrix()


def geodesic(Ra, Rb):
    """Angle (rad) between rotations, broadcasting over leading dims."""
    tr = np.einsum("...ij,...ij->...", Ra, Rb)
    return np.arccos(np.clip((tr - 1) / 2, -1, 1))


def blend_quats(Rs, W):
    """Rs [K,T,3,3], W [K,T] (sum 1) -> [T,3,3]; signs aligned to the first source."""
    q = np.stack([Rotation.from_matrix(R).as_quat() for R in Rs])  # K,T,4
    sgn = np.sign((q * q[:1]).sum(-1, keepdims=True))
    sgn[sgn == 0] = 1
    qm = (q * sgn * W[..., None]).sum(0)
    return Rotation.from_quat(qm / np.linalg.norm(qm, axis=1, keepdims=True)).as_matrix()


def accel_scored(t_body, scored):
    """Mean |second difference| (cm / frame^2) of a translation [T,3] (body frame, m) over scored stretches."""
    sc = np.asarray(scored)
    sc = sc[sc < len(t_body)]
    br = np.flatnonzero(np.diff(sc) != 1) + 1
    vals = []
    for part in np.split(sc, br):
        if len(part) >= 3:
            x = t_body[part]
            vals.append(np.linalg.norm(x[:-2] - 2 * x[1:-1] + x[2:], axis=1))
    return float(np.mean(np.concatenate(vals)) * 100) if vals else None


def hand_vertex_split(model):
    b = {k: v.cpu().numpy() for k, v in model.named_buffers()}
    c = "character_torch."
    parents = b[c + "skeleton.joint_parents"]
    nv = b[c + "mesh.rest_vertices"].shape[0]
    w = np.zeros((nv, len(parents)))
    np.add.at(w, (b[c + "linear_blend_skinning.vert_indices_flattened"],
                  b[c + "linear_blend_skinning.skin_indices_flattened"]),
              b[c + "linear_blend_skinning.skin_weights_flattened"])
    out = {}
    for name, root in HAND_ROOT.items():
        sub = {root}
        for j in range(len(parents)):
            p = parents[j]
            while p >= 0:
                if p == root:
                    sub.add(j)
                    break
                p = parents[p]
        out[name] = np.flatnonzero(w[:, sorted(sub)].sum(1) > 0.5)
    return out


def load_hands(ep, body_dir, sigma_h, mhr, hids, sub=2):
    """Wrist frames and hand vertices (body frame, metres) of the body as assemble_episode.py builds it."""
    b = dict(np.load(os.path.join(body_dir, f"episode_{ep:06d}.npz")))
    valid = b.get("valid", np.ones(len(b["pose"]), bool)).astype(bool)
    pose = smooth(unwrap_euler(fill_invalid(b["pose"], valid), list(range(3, 136))), sigma_h)
    scales = np.median(np.asarray(b["scales"])[valid], axis=0) if np.ndim(b["scales"]) == 2 else b["scales"]
    shape = np.median(np.asarray(b["shape"])[valid], axis=0) if np.ndim(b["shape"]) == 2 else b["shape"]
    ids = {h: hids[h][::sub] for h in ("right", "left")}
    allid = np.concatenate([ids["right"], ids["left"]])
    pos, rot, verts = mhr.skeleton(pose, scales, shape, [HAND_JOINT["right"], HAND_JOINT["left"]], vert_ids=allid)
    nr = len(ids["right"])
    return dict(p={"right": pos[:, 0], "left": pos[:, 1]}, R={"right": rot[:, 0], "left": rot[:, 1]},
                V={"right": verts[:, :nr].astype(np.float32), "left": verts[:, nr:].astype(np.float32)},
                floor_n=np.asarray(b.get("floor_n", [0, -1, 0]), float), floor_d=float(b.get("floor_d", 0.0)))


def hand_frame(H, name, frames, r, axis=None, R_obj=None):
    """Hand frame (R [N,3,3], p [N,3]) in the MoGe frame for the given frames.

    '<hand>_pt' (grasp-point lock): position of the hand frame, rotation of the object itself (R_obj, the
    v6 rotation), so that with G = (I, tG) the wrist stays at a fixed point of the object's frame."""
    if name.endswith("_pt"):
        _, p = hand_frame(H, name[:-3], frames, r, axis)
        return R_obj[frames], p
    if name in ("right", "left"):
        return H["R"][name][frames], H["p"][name][frames] * r
    RR, RL = H["R"]["right"][frames], H["R"]["left"][frames]
    p = 0.5 * (H["p"]["right"][frames] + H["p"]["left"][frames]) * r
    W = np.full((2, len(frames)), 0.5)
    Rm = blend_quats(np.stack([RR, RL]), W)
    if name == "mid":
        return Rm, p
    # 'bar': the mid frame turned (minimal rotation) so that its axis `axis` (mid-frame coordinates,
    # the segment's median wrist-to-wrist direction) follows the actual wrist-to-wrist vector
    x = H["p"]["left"][frames] - H["p"]["right"][frames]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    a = np.einsum("nij,j->ni", Rm, axis)
    v = np.cross(a, x)
    sn = np.linalg.norm(v, axis=1)
    ang = np.arctan2(sn, (a * x).sum(1))
    rv = v / np.maximum(sn, 1e-12)[:, None] * ang[:, None]
    return Rotation.from_rotvec(rv).as_matrix() @ Rm, p


def bar_axis(H, frames):
    RR, RL = H["R"]["right"][frames], H["R"]["left"][frames]
    Rm = blend_quats(np.stack([RR, RL]), np.full((2, len(frames)), 0.5))
    x = H["p"]["left"][frames] - H["p"]["right"][frames]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    u = np.median(np.einsum("nji,nj->ni", Rm, x), axis=0)
    return u / np.linalg.norm(u)


# ------------------------------------------------------------------------------------------ episode data
class StableDepth:
    """MoGe depth (float16) with its per-frame background scale removed on access (float32 frame)."""

    def __init__(self, D, r):
        self.D, self.r = D, r

    def __getitem__(self, f):
        return self.D[f].astype(np.float32) / np.float32(self.r[f])

    def __len__(self):
        return len(self.D)


class LightEpisode(Episode):
    """objlib.Episode built frame by frame (quarter-resolution masks only, float16 depth): same attributes
    as used here and by FrameTargets / data_loss, about a third of the memory."""

    def __init__(self, ep):
        from common import unpack_masks
        self.ep = ep
        md = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
        h, w = int(md["mask_h"]), int(md["mask_w"])
        self.T = T = int(md["num_frames"])
        h2, w2 = h // 2, w // 2
        self.obj = np.zeros((T, h2, w2), bool)
        self.per = np.zeros((T, h2, w2), bool)
        om_p, pm_p = md["obj_masks"], md["person_masks"]
        for f in range(T):
            for src, dst in ((om_p, self.obj), (pm_p, self.per)):
                m = unpack_masks(src, h, w, f)
                dst[f] = m[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).any((1, 3))
        self.obj_half = None
        self.H, self.W = h2, w2
        self.score = md["obj_score"]
        self.area = self.obj.sum((1, 2))
        self.visible = (self.score > 0) & (self.area > 8)
        self.occ = self.per & ~self.obj
        self.K_full = np.array([[807.66, 0, 768.0], [0, 807.66, 576.0], [0, 0, 1]])
        kp = os.path.join(WORK, "intrinsics", f"episode_{ep:06d}.json")
        if os.path.exists(kp):
            d = json.load(open(kp))
            self.K_full = np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]], np.float64)
        self.K = self.K_full.copy()
        self.K[:2] /= 4
        # as objlib.load_stable_depth, without float32 copies of the whole sequence
        D = np.load(os.path.join(WORK, "depth", f"episode_{ep:06d}.npz"))["depth"]
        fg = self.obj | self.per
        ker = np.ones((9, 9), np.uint8)
        sub = np.linspace(0, T - 1, min(T, 120)).astype(int)
        ref = np.median(D[sub].astype(np.float32), 0)
        r = np.ones(T)
        for t in range(T):
            Dt = D[t].astype(np.float32)
            bg = ~(cv2.dilate(fg[t].astype(np.uint8), ker) > 0) & (ref > 0) & (Dt > 0)
            if bg.sum() > 500:
                r[t] = np.median(Dt[bg] / ref[bg])
        self.depth_scale = gaussian_filter1d(r, 1.0, mode="nearest")
        self.depth = StableDepth(D, self.depth_scale)
        self._dt = {}


# ------------------------------------------------------------------------------------------ scoring
class Scorer:
    """Hard-silhouette IoU, MoGe depth ratio and hand gap of posed objects (raw MoGe frame)."""

    def __init__(self, E, mv, mf, s, sdf, r, P_dense):
        self.E, self.mv, self.mf, self.s, self.sdf, self.r = E, mv, mf, s, sdf, r
        self.P = P_dense
        ker = np.ones((3, 3), np.uint8)
        self.inner = {}
        self._ker = ker

    def iou(self, f, R, t):
        E = self.E
        if not E.visible[f]:
            return np.nan
        Vc = self.s * self.mv @ R.T + t
        sil = hard_silhouette(Vc, self.mf, E.K, E.H, E.W)
        keep = ~E.occ[f]
        return float((sil & E.obj[f] & keep).sum() / max(((sil | E.obj[f]) & keep).sum(), 1))

    def kz(self, f, R, t):
        E = self.E
        if not E.visible[f] or E.depth is None:
            return np.nan
        if f not in self.inner:
            self.inner[f] = (cv2.erode(E.obj[f].astype(np.uint8), self._ker) > 0) & ~E.per[f]
        mm = self.inner[f]
        if mm.sum() < 15:
            return np.nan
        Pc = self.s * self.P @ R.T + t
        z = Pc[:, 2]
        ok = z > 0.05
        K = E.K
        u = np.round(K[0, 0] * Pc[ok, 0] / z[ok] + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * Pc[ok, 1] / z[ok] + K[1, 2]).astype(int)
        inb = (u >= 0) & (u < E.W) & (v >= 0) & (v < E.H)
        buf = np.full(E.H * E.W, np.inf, np.float32)
        np.minimum.at(buf, v[inb] * E.W + u[inb], z[ok][inb].astype(np.float32))
        zb = buf.reshape(E.H, E.W)
        Df = E.depth[f]
        sel = mm & np.isfinite(zb) & (Df > 0)
        if sel.sum() < 15:
            return np.nan
        return float(np.median(zb[sel]) / np.median(Df[sel]))

    @torch.no_grad()
    def hand_sd(self, R, t, Hv):
        """Signed distance (body cm) of hand vertices Hv [N,M,3] (MoGe frame) to posed objects R [N], t [N]:
        returns min over vertices [N] and the penetration sum of squares [N]."""
        dev = "cuda"
        Rt = torch.tensor(R, dtype=torch.float32, device=dev)
        tt = torch.tensor(t, dtype=torch.float32, device=dev)
        Ht = torch.tensor(Hv, dtype=torch.float32, device=dev)
        loc = torch.einsum("npi,nij->npj", Ht - tt[:, None], Rt) / self.s
        sd = self.sdf.query_torch(loc) * self.s * 100.0 / self.r
        return sd.amin(1).cpu().numpy(), (torch.relu(-sd) ** 2).sum(1).cpu().numpy()

    def series(self, frames, R, t, do_kz=True):
        iou = np.array([self.iou(f, R[i], t[i]) for i, f in enumerate(frames)])
        kz = np.array([self.kz(f, R[i], t[i]) for i, f in enumerate(frames)]) if do_kz else None
        return iou, kz


def summarize(iou, kz, gap):
    def nm(x):
        x = x[np.isfinite(x)] if x is not None else np.array([])
        return x
    i, k, g = nm(iou), nm(kz), nm(gap)
    return dict(iou=float(i.mean()) if len(i) else None, n_iou=int(len(i)),
                kz_med=float(np.median(k)) if len(k) else None,
                dz=float(np.median(np.abs(k - 1))) if len(k) else None, n_kz=int(len(k)),
                gap_cm=float(np.median(np.maximum(g, 0))) if len(g) else None,
                pen_cm=float(np.median(np.maximum(-g, 0))) if len(g) else None)


# ------------------------------------------------------------------------------------------ contact
@torch.no_grad()
def contact_features(E, H, R6, t6, s, sdf, r, frames, ks=np.exp(np.linspace(np.log(0.8), np.log(1.25), 19)),
                     chunk=48):
    """Per frame and hand: 2-D distance (quarter px, 10th percentile) of the projected hand vertices to
    the object mask, and the 3-D gap (body cm) minimised over a depth scaling k of the object."""
    out = {h: dict(d2=np.full(E.T, np.nan), g3=np.full(E.T, np.nan)) for h in ("right", "left")}
    K = E.K
    frames = np.asarray(frames, int)
    for f in frames:
        if E.area[f] <= 8:
            continue
        dt = cv2.distanceTransform((~E.obj[f]).astype(np.uint8), cv2.DIST_L2, 3)
        for h in ("right", "left"):
            Vb = H["V"][h][f]  # body frame (projection is scale-free)
            z = np.clip(Vb[:, 2], 1e-3, None)
            u = np.clip(np.round(K[0, 0] * Vb[:, 0] / z + K[0, 2]).astype(int), 0, E.W - 1)
            v = np.clip(np.round(K[1, 1] * Vb[:, 1] / z + K[1, 2]).astype(int), 0, E.H - 1)
            out[h]["d2"][f] = float(np.percentile(dt[v, u], 10))
    kt = torch.tensor(ks, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        for h in ("right", "left"):
            g = []
            for a in range(0, len(frames), chunk):
                fr = frames[a:a + chunk]
                Hm = torch.tensor(H["V"][h][fr] * r, dtype=torch.float32, device="cuda")  # MoGe frame
                Rt = torch.tensor(R6[fr], dtype=torch.float32, device="cuda")
                tt = torch.tensor(t6[fr], dtype=torch.float32, device="cuda")
                # object scaled by k about the camera == hand scaled by 1/k, distances x k
                Q = Hm[:, None] / kt[None, :, None, None]
                loc = torch.einsum("fkpi,fij->fkpj", Q - tt[:, None, None], Rt) / s
                sd = sdf.query_torch(loc) * s * kt[None, :, None] * 100.0 / r
                g.append(torch.relu(sd).amin(2).amin(1))
            if g:
                out[h]["g3"][frames] = torch.cat(g).cpu().numpy()
    return out


def contact_masks(feat, moving, touch_px, touch_cm):
    c = {}
    for h in ("right", "left"):
        d2, g3 = feat[h]["d2"], feat[h]["g3"]
        raw = (d2 <= touch_px) & (g3 <= touch_cm)
        known = np.isfinite(d2)
        x = raw.astype(float)
        x[~known] = np.nan
        idx = np.flatnonzero(known)
        if len(idx):
            x = np.interp(np.arange(len(x)), idx, x[idx]) >= 0.5  # occluded mask: nearest evidence
        else:
            x = np.zeros(len(x), bool)
        x = binary_closing(x, np.ones(7, bool)) | x
        x = binary_opening(x, np.ones(5, bool))
        c[h] = x & moving
    return c


def segment(moving, c, min_len):
    """Runs inside moving intervals where at least one hand stays in contact (anchor set)."""
    segs = []
    for a, b in runs(moving):
        f = a
        while f < b:
            cur = {h for h in ("right", "left") if c[h][f]}
            if not cur:
                f += 1
                continue
            s0, anchor, g = f, cur, f + 1
            while g < b:
                nxt = {h for h in anchor if c[h][g]}
                if not nxt:
                    break
                anchor, g = nxt, g + 1
            segs.append([s0, g, sorted(anchor)])
            f = g
    # handover: move the cut between touching segments to the middle of the two-hand period
    for i in range(len(segs) - 1):
        a0, b0, h0 = segs[i]
        a1, b1, h1 = segs[i + 1]
        if b0 != a1:
            continue
        x = b0
        while x - 1 > a0 and c["right"][x - 1] and c["left"][x - 1]:
            x -= 1
        if x < b0:
            cut = (x + b0) // 2
            segs[i][1], segs[i + 1][0] = cut, cut
    return [sg for sg in segs if sg[1] - sg[0] >= min_len]


# ------------------------------------------------------------------------------------------ G
def regrasp_cuts(X, idx, n, min_len, pen):
    """Change points of a piecewise-constant signal (object centre in the hand frame, cm) by dynamic
    programming on the squared error, X [m,3] observed at segment-relative frames idx (sorted).
    The penalty per extra piece is pen * sigma^2 * log(m), sigma the robust frame-to-frame noise.
    Returns cut frames (segment-relative) such that every piece spans >= min_len frames."""
    m = len(X)
    if m < 2 * 5 or n < 2 * min_len:
        return []
    dX = np.diff(X, axis=0)
    sig2 = float(np.mean((1.4826 * np.median(np.abs(dX - np.median(dX, 0)), 0) / np.sqrt(2)) ** 2)) + 1e-6
    lam = pen * sig2 * np.log(m) * X.shape[1]
    c1 = np.vstack([np.zeros((1, 3)), np.cumsum(X, 0)])
    c2 = np.r_[0.0, np.cumsum((X ** 2).sum(1))]

    def cost(i, j):  # observations i..j-1
        k = j - i
        mu = (c1[j] - c1[i]) / k
        return c2[j] - c2[i] - k * (mu ** 2).sum()
    # piece boundaries are allowed only where both sides span >= min_len frames
    best = np.full(m + 1, np.inf)
    prev = np.zeros(m + 1, int)
    best[0] = 0.0
    for j in range(1, m + 1):
        end = n if j == m else idx[j]  # frame where the piece ending before observation j stops
        for i in range(0, j):
            start = 0 if i == 0 else idx[i]
            if end - start < min_len or not np.isfinite(best[i]):
                continue
            v = best[i] + cost(i, j) + (lam if i > 0 else 0.0)
            if v < best[j]:
                best[j], prev[j] = v, i
    cuts, j = [], m
    while j > 0:
        i = prev[j]
        if i > 0:
            cuts.append(int(idx[i]))
        j = i
    return sorted(cuts)


def robust_G(RG, tG, w):
    n = len(RG)
    if n == 1:
        return RG[0], tG[0], 1
    ang = geodesic(RG[:, None], RG[None])
    med = int(np.argmin((ang * w[None]).sum(1)))
    inl = ang[med] < np.radians(30)
    Rm = quat_mean(RG[inl], w[inl])
    tm = np.median(tG[inl], axis=0)
    return Rm, tm, int(inl.sum())


class CropDepth:
    def __init__(self, D, box):
        self.D, self.box = D, box

    def __getitem__(self, f):
        y0, y1, x0, x1 = self.box
        return self.D[f][y0:y1, x0:x1]


class CropView:
    """The episode restricted to an image box (array views) for FrameTargets: rendering only the
    object's neighbourhood instead of the whole quarter-resolution frame."""
    dt_out = Episode.dt_out

    def __init__(self, E, box):
        y0, y1, x0, x1 = box
        self.obj = E.obj[:, y0:y1, x0:x1]
        self.occ = E.occ[:, y0:y1, x0:x1]
        self.per = E.per[:, y0:y1, x0:x1]
        self.visible = E.visible
        self.depth = None if E.depth is None else CropDepth(E.depth, box)
        self.K = E.K.copy()
        self.K[0, 2] -= x0
        self.K[1, 2] -= y0
        self.H, self.W = y1 - y0, x1 - x0
        self._dt = {}


def crop_box(E, frames, margin=0.35, min_margin=16):
    m = E.obj[frames].any(0)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return 0, E.H, 0, E.W
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    g = int(max(min_margin, margin * max(h, w)))
    return (int(max(0, ys.min() - g)), int(min(E.H, ys.max() + 1 + g)),
            int(max(0, xs.min() - g)), int(min(E.W, xs.max() + 1 + g)))


def data_loss_vec(sil, z, tg, zc, thin=False, uv=None, w_dt=0.03, w_depth=0.5, w_ch=0.05):
    """objlib.data_loss with the per-frame depth medians computed in one batched call (no host syncs)."""
    from objlib import chamfer2d
    m, keep = tg.m, tg.keep
    inter = (sil * m * keep).sum((1, 2))
    union = ((sil + m - sil * m) * keep).sum((1, 2)) + 1e-6
    loss = 1 - inter / union
    area = sil.sum((1, 2)) + 1.0
    loss = loss + w_dt * (sil * tg.dt).sum((1, 2)) / area
    if uv is not None and w_ch > 0:
        loss = loss + w_ch * chamfer2d(uv, tg)
    if zc is not None and tg.d is not None and w_depth > 0 and not thin:
        with torch.no_grad():
            B = z.shape[0]
            vis = (z > 0) & (tg.d > 0)
            nan = torch.full_like(z, float("nan"))
            gap = (torch.nanmedian(torch.where(vis, z, nan).reshape(B, -1), 1).values
                   - torch.nanmedian(torch.where(vis, tg.d, nan).reshape(B, -1), 1).values)
            ok = vis.reshape(B, -1).sum(1) > 15
            gap = torch.where(ok, gap, torch.zeros_like(gap))
        e = (zc - zc.detach() + gap).abs()
        hub = torch.where(e < 0.05, 0.5 * e ** 2 / 0.05, e - 0.025)
        loss = loss + w_depth * torch.where(ok, hub, torch.zeros_like(hub))
    return loss * tg.has


def refine_G(RG0, tG0, Rh, ph, frames, E, mv_t, mf_t, P_t, c_mesh, s, sdf, Hv, r, iters, thin,
             w_gap, w_pen, w_prior, rot=True):
    dev = "cuda"
    C = CropView(E, crop_box(E, frames))
    ren = Renderer(C.K, C.H, C.W)
    tg = FrameTargets(C, frames, device=dev, use_depth=not thin)
    Rh_t = torch.tensor(Rh, dtype=torch.float32, device=dev)
    ph_t = torch.tensor(ph, dtype=torch.float32, device=dev)
    RG0_t = torch.tensor(RG0, dtype=torch.float32, device=dev)
    tG0_t = torch.tensor(tG0, dtype=torch.float32, device=dev)
    K_t = torch.tensor(C.K, dtype=torch.float32, device=dev)
    c_t = torch.tensor(c_mesh, dtype=torch.float32, device=dev)
    H_t = torch.tensor(Hv, dtype=torch.float32, device=dev) if Hv is not None else None
    w = torch.zeros(3, device=dev, requires_grad=True)
    d = torch.zeros(3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [w], "lr": 0.01 if rot else 0.0}, {"params": [d], "lr": 0.004}])
    hist = []
    for it in range(iters):
        RG = RG0_t @ so3_exp(w)
        tG = tG0_t + d
        Ro = Rh_t @ RG
        to = ph_t + torch.einsum("nij,j->ni", Rh_t, tG)
        V = s * torch.einsum("nij,vj->nvi", Ro, mv_t) + to[:, None]
        sil, z = ren(V, mf_t)
        uv = project_points(s * torch.einsum("nij,vj->nvi", Ro, P_t) + to[:, None], K_t)
        zc = (s * Ro @ c_t + to)[:, 2]
        loss = data_loss_vec(sil, z, tg, zc, thin=thin, uv=uv).mean()
        if H_t is not None and (w_gap > 0 or w_pen > 0):
            loc = torch.einsum("npi,nij->npj", H_t - to[:, None], Ro) / s
            sd = sdf.query_torch(loc) * s * 100.0 / r  # body cm
            loss = loss + w_gap * torch.relu(sd.amin(1) - 0.5).mean() + w_pen * (torch.relu(-sd) ** 2).sum(1).mean()
        loss = loss + w_prior * ((w ** 2).sum() / 0.3 ** 2 + (d ** 2).sum() / 0.1 ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 20 == 0 or it == iters - 1:
            hist.append(loss.detach())
    hist = [round(float(x), 4) for x in hist]
    with torch.no_grad():
        RG = (RG0_t @ so3_exp(w)).double().cpu().numpy()
        tG = (tG0_t + d).double().cpu().numpy()
    U, _, Vt = np.linalg.svd(RG)
    return U @ Vt, tG, hist


def split_regrasps(segs, E, S, H, R6, t6, s, c_mesh, r, a):
    """Split grasp segments where the object centre jumps in the frame of the holding hand (regrasp)."""
    out = []
    for sa, sb, anchor in segs:
        fr = np.arange(sa, sb)
        iou6 = np.array([S.iou(f, R6[f], t6[f]) for f in fr])
        ok = E.visible[fr] & np.isfinite(iou6)
        if ok.sum() >= 10:
            ok &= iou6 >= max(0.5, float(np.quantile(iou6[ok], 0.3)))
        hname = anchor[0] if len(anchor) == 1 else "mid"
        Rh, ph = hand_frame(H, hname, fr, r)
        ctr = s * np.einsum("nij,j->ni", R6[fr], c_mesh) + t6[fr]
        X = np.einsum("nji,nj->ni", Rh, ctr - ph) * 100 / r  # cm, body units
        idx = np.flatnonzero(ok)
        cuts = regrasp_cuts(X[idx], idx, len(fr), a.split_min, a.split_pen) if len(idx) >= 10 else []
        edges = [0] + cuts + [len(fr)]
        for x0, x1 in zip(edges[:-1], edges[1:]):
            out.append([sa + x0, sa + x1, anchor])
    return out


def select(rec):
    """Accepted strict lock first, then accepted grasp-point lock; within a model the best-fitting hand frame."""
    C = rec["cands"]
    h = min(C, key=lambda k: (not C[k]["accepted"], C[k]["model"] == "point",
                              -(C[k]["lock"]["iou"] if C[k]["lock"]["iou"] is not None else -1)))
    rec["hand"], rec["model"] = h, C[h]["model"]
    for k in ("before", "lock", "after", "accepted", "reason", "tol", "G"):
        rec[k] = C[h][k]


def decide(rec, a, thin):
    """Acceptance of a locked segment on the trajectory as written (rec['after']) against v6 (rec['before'])."""
    b4, af = rec["before"], rec["after"]
    ok_iou = b4["iou"] is not None and af["iou"] is not None and af["iou"] >= b4["iou"] - a.tol_iou
    if thin or b4["dz"] is None or af["dz"] is None or min(b4["n_kz"], af["n_kz"]) < 5:
        ok_kz, kz_note = True, "depth not compared"
    else:
        ok_kz, kz_note = af["dz"] <= b4["dz"] + a.tol_kz, ""
    lg = rec.get("lock", {}).get("gap_cm")
    ok_gap = lg is not None and lg <= a.max_gap  # a grasp lock must keep the object on the hand
    rec["accepted"] = bool(ok_iou and ok_kz and ok_gap) or a.force
    rec["reason"] = ("ok" if rec["accepted"] else
                     ("IoU" if not ok_iou else "") + (" depth" if not ok_kz else "") +
                     (" hand-gap" if not ok_gap else "")).strip() + (f" ({kz_note})" if kz_note else "")
    rec["tol"] = dict(iou=a.tol_iou, kz=a.tol_kz, gap_cm=a.max_gap)


def decimate(mv, mf, n):
    if len(mf) <= n:
        return mv, mf
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mv), o3d.utility.Vector3iVector(mf))
    m = m.simplify_quadric_decimation(n)
    return np.asarray(m.vertices), np.asarray(m.triangles).astype(np.int64)


def apply_G(Rh, ph, RG, tG):
    return Rh @ RG, ph + np.einsum("nij,j->ni", Rh, tG)


# ------------------------------------------------------------------------------------------ episode
def process(ep, a, mhr, hids, scored, ratio, meta):
    t0 = time.time()
    src, out = a.src, a.out
    tr = dict(np.load(os.path.join(src, f"episode_{ep:06d}_track.npz")))
    R6, t6, s = tr["R"].astype(np.float64), tr["t"].astype(np.float64), float(tr["scale"])
    seg_id = tr["seg_id"]
    r = float(ratio[str(ep)])
    log = dict(ep=ep, cls=meta[ep]["object"], ratio=r, scale=s, segments=[], excluded=None)
    sc = np.asarray(scored[str(ep)])
    T6 = len(t6)

    def finish(Rn, tn, locked, lock_id):
        np.savez(os.path.join(out, f"episode_{ep:06d}_track.npz"), **{**tr, "R": Rn, "t": tn,
                 "locked": locked, "lock_id": lock_id})
        pj = json.load(open(os.path.join(src, f"episode_{ep:06d}_pose.json")))
        fr = int(pj.get("frame", 0))
        pj["R"], pj["t"] = Rn[fr].tolist(), tn[fr].tolist()
        pj["grasp_lock"] = [dict(a=x["a"], b=x["b"], hand=x.get("hand"), accepted=x.get("accepted"))
                            for x in log["segments"]]
        json.dump(pj, open(os.path.join(out, f"episode_{ep:06d}_pose.json"), "w"), indent=1)
        shutil.copy(os.path.join(src, f"episode_{ep:06d}_object.ply"), os.path.join(out, f"episode_{ep:06d}_object.ply"))
        tb6, tbn = t6 / r, tn / r
        log["acc_scored"] = dict(before=accel_scored(tb6, sc), after=accel_scored(tbn, sc),
                                 before_s12=accel_scored(gaussian_filter1d(tb6, 12, axis=0, mode="nearest"), sc),
                                 after_s12=accel_scored(gaussian_filter1d(tbn, 12, axis=0, mode="nearest"), sc))
        scv = sc[sc < T6]
        log["n_locked"] = int(locked.sum())
        log["n_locked_scored"] = int(locked[scv].sum())
        log["shift_scored_cm"] = float(np.linalg.norm(tbn[scv] - tb6[scv], axis=1).mean() * 100)
        log["time_s"] = round(time.time() - t0, 1)
        json.dump(log, open(os.path.join(out, "logs", f"episode_{ep:06d}.json"), "w"), indent=1)
        return log

    if ep in EXCLUDE:
        log["excluded"] = EXCLUDE[ep]
        return finish(R6, t6, np.zeros(T6, bool), -np.ones(T6, int))

    E = LightEpisode(ep)
    H = load_hands(ep, a.body, a.sigma_h, mhr, hids)
    T = min(T6, E.T, len(H["p"]["right"]))
    mv, mf, sdf = load_sdf(os.path.join(SDF_DIR, f"episode_{ep:06d}_sdf.npz"))
    from audit_objects import sample_surface
    P_dense = sample_surface(mv, mf, 20000, seed=ep + 100)
    P_ch = sample_surface(mv, mf, 600, seed=ep)
    c_mesh = mv.mean(0)
    S = Scorer(E, mv, mf, s, sdf, r, P_dense)
    thin = log["cls"] == "hula_hoop"

    if a.recompose:  # segments and G from an earlier run; only the acceptance is re-decided
        old = json.load(open(os.path.join(a.recompose, "logs", f"episode_{ep:06d}.json")))
        log["contact_frames"] = old.get("contact_frames")
        segs = []
        old_segs = old["segments"]
    else:
        moving = np.zeros(T6, bool)
        moving[:T] = seg_id[:T] < 0
        feat = contact_features(E, H, R6, t6, s, sdf, r, np.flatnonzero(moving[:T]))
        c = contact_masks(feat, moving[:T], a.touch_px, a.touch_cm)
        segs = segment(moving[:T], c, a.min_len)
        log["contact_frames"] = {h: int(c[h].sum()) for h in c}
        if a.split_pen > 0:
            segs = split_regrasps(segs, E, S, H, R6, t6, s, c_mesh, r, a)
            log["split"] = [[int(x[0]), int(x[1])] for x in segs]
        old_segs = []

    dev = "cuda"
    mv_r, mf_r = decimate(mv, mf, a.render_faces)
    mv_t = torch.tensor(mv_r, dtype=torch.float32, device=dev)
    mf_t = torch.tensor(mf_r, dtype=torch.int64, device=dev)
    P_t = torch.tensor(P_ch, dtype=torch.float32, device=dev)
    static = seg_id[:T6] >= 0
    accepted = []
    dbg = {}
    for (sa, sb, anchor) in segs:
        fr = np.arange(sa, sb)
        rec = dict(a=int(sa), b=int(sb), n=int(sb - sa), anchor=anchor, n_scored=int(np.isin(fr, sc).sum()))
        vis = E.visible[fr]
        if len(anchor) == 2:  # a 'both hands' contact can come from the depth-scaled test only: keep the near hand
            gm = {h: float(np.median(np.maximum(S.hand_sd(R6[fr], t6[fr], H["V"][h][fr] * r)[0], 0))) for h in anchor}
            rec["gap_v6_cm"] = gm
            near = [h for h in anchor if gm[h] <= 5.0]
            if len(near) == 1 and max(gm.values()) > 10.0:
                anchor = near
                rec["anchor_used"] = anchor
        iou6, kz6 = S.series(fr, R6[fr], t6[fr])
        cands = list(anchor) + (["mid", "bar"] if len(anchor) == 2 else [])
        if a.point:  # grasp-point lock (wrist fixed in object coordinates, rotation from v6)
            cands += [h + "_pt" for h in anchor] + (["mid_pt"] if len(anchor) == 2 else [])
        q = vis & np.isfinite(iou6)
        if q.sum() >= 5:
            thr = max(0.5, float(np.quantile(iou6[q], 0.3)))
            good = q & (iou6 >= thr) & (~np.isfinite(kz6) | (np.abs(kz6 - 1) <= 0.08))
            if good.sum() < 5:
                good = q
        else:
            good = q
        rec["n_good"] = int(good.sum())
        hv_all = {h: H["V"][h][fr] * r for h in ("right", "left")}
        # pick-up / place boundary poses (static frames adjacent to the segment)
        pre = sa - 1 if sa - 1 >= 0 and static[sa - 1] else None
        post = sb if sb < T6 and static[sb] else None
        rec["cands"] = {}
        keep = {}
        for hname in cands:
            point = hname.endswith("_pt")
            base = hname[:-3] if point else hname
            axis = bar_axis(H, fr) if base == "bar" else None
            Rh, ph = hand_frame(H, hname, fr, r, axis, R_obj=R6)
            Hv = np.concatenate([hv_all[h] for h in anchor], 1) if base in ("mid", "bar") else hv_all[base]
            RG_t = np.einsum("nji,njk->nik", Rh, R6[fr])
            tG_t = np.einsum("nji,nj->ni", Rh, t6[fr] - ph)
            Gs = {}
            res = None
            if good.sum() >= 3:
                Rm, tm, ninl = robust_G(RG_t[good], tG_t[good], iou6[good])
                Gs["robust"] = (Rm, tm)
                # rigid residual of the per-frame relative transforms (inliers of the robust mean)
                ang = np.degrees(geodesic(RG_t[good], Rm[None]))
                ctr = np.einsum("nij,j->ni", RG_t[good], s * c_mesh) + tG_t[good]
                ctr0 = Rm @ (s * c_mesh) + tm
                res = dict(rot_deg=float(np.median(ang)), ctr_cm=float(np.median(np.linalg.norm(ctr - ctr0, axis=1)) * 100 / r),
                           inliers=ninl)
            if pre is not None:
                Gs["pickup"] = (Rh[0].T @ R6[pre], Rh[0].T @ (t6[pre] - ph[0]))
            if post is not None:
                Gs["place"] = (Rh[-1].T @ R6[post], Rh[-1].T @ (t6[post] - ph[-1]))
            if point:
                Gs = {k: (np.eye(3), v[1]) for k, v in Gs.items()}
            if not Gs:
                continue
            ev = np.flatnonzero(vis)
            ev = ev[np.linspace(0, len(ev) - 1, min(len(ev), 60)).astype(int)] if len(ev) else ev
            sco = {}
            for gname, (RG, tG) in Gs.items():
                Ro, to = apply_G(Rh[ev], ph[ev], RG, tG)
                sco[gname] = float(np.nanmean([S.iou(fr[i], Ro[j], to[j]) for j, i in enumerate(ev)])) if len(ev) else 0.0
            gbest = max(sco, key=lambda k: sco[k])
            RG0, tG0 = Gs[gbest]
            # refinement on up to --refine-frames visible frames
            rf = np.flatnonzero(vis)
            if len(rf) >= 3 and a.iters > 0:
                rf = rf[np.linspace(0, len(rf) - 1, min(len(rf), a.refine_frames)).astype(int)]
                RG1, tG1, hist = refine_G(RG0, tG0, Rh[rf], ph[rf], fr[rf], E, mv_t, mf_t, P_t, c_mesh, s, sdf,
                                          Hv[rf], r, a.iters, thin, a.w_gap, a.w_pen, a.w_prior, rot=not point)
            else:
                RG1, tG1, hist = RG0, tG0, []
            if point:
                RG1 = np.eye(3)
            E._dt.clear()
            Ro, to = apply_G(Rh, ph, RG1, tG1)
            iou7, kz7 = S.series(fr, Ro, to)
            gap6, _ = S.hand_sd(R6[fr], t6[fr], Hv)
            gap7, pen7 = S.hand_sd(Ro, to, Hv)
            st7 = summarize(iou7, kz7, gap7)
            # the trajectory that is written: lock weight ramped in from the segment edges
            dd = np.minimum(np.arange(1, len(fr) + 1), np.arange(len(fr), 0, -1)).astype(float)
            u = 1 - np.exp(-dd ** 2 / (2 * a.sigma_blend ** 2))
            Rb = blend_quats(np.stack([R6[fr], Ro]), np.stack([1 - u, u]))
            tb = (1 - u)[:, None] * t6[fr] + u[:, None] * to
            ioub, kzb = S.series(fr, Rb, tb)
            gapb, _ = S.hand_sd(Rb, tb, Hv)
            crec = dict(model="point" if point else "rigid", init=gbest, init_iou=sco, residual=res, refine_loss=hist,
                        before=summarize(iou6, kz6, gap6), lock=st7, after=summarize(ioub, kzb, gapb),
                        dG_deg=float(np.degrees(geodesic(RG0, RG1))), dG_cm=float(np.linalg.norm(tG1 - tG0) * 100 / r))
            crec["G"] = dict(R=RG1.tolist(), t_body=(tG1 / r).tolist())
            decide(crec, a, thin)
            rec["cands"][hname] = crec
            keep[hname] = (RG1, tG1)
            if a.debug:
                dbg[f"{sa}_{hname}"] = np.stack([fr, iou6, kz6, iou7, kz7, gap6, gap7, ioub, kzb, gapb]).astype(np.float32)
        if not rec["cands"]:
            rec["before"] = summarize(iou6, kz6, None)
            rec["accepted"], rec["reason"] = False, "no candidate"
            log["segments"].append(rec)
            continue
        select(rec)
        RG1, tG1 = keep[rec["hand"]]
        hname = rec["hand"]
        log["segments"].append(rec)
        if rec["accepted"]:
            accepted.append((sa, sb, hname, RG1, tG1, anchor))
    for rec in old_segs:
        if rec.get("cands") and all("G" in c for c in rec["cands"].values()):
            for c in rec["cands"].values():
                decide(c, a, thin)
            select(rec)
        elif "G" in rec:
            decide(rec, a, thin)
        log["segments"].append(rec)
        if rec.get("accepted"):
            accepted.append((rec["a"], rec["b"], rec["hand"], np.asarray(rec["G"]["R"]),
                             np.asarray(rec["G"]["t_body"]) * r, rec.get("anchor_used", rec["anchor"])))

    if a.debug and dbg:
        np.savez(os.path.join(out, "logs", f"episode_{ep:06d}_series.npz"), **dbg)
    # -------------------------------------------------------------------------- compose
    lab = np.zeros(T6, int)
    for k, (sa, sb, *_rest) in enumerate(accepted, 1):
        lab[sa:sb] = k
    locked = lab > 0
    if not accepted:
        return finish(R6, t6, locked, lab - 1)
    allf = np.arange(T)
    srcR, srcT = [R6], [t6]
    for (sa, sb, hname, RG1, tG1, anchor) in accepted:
        axis = bar_axis(H, np.arange(sa, sb)) if hname.split("_")[0] == "bar" else None
        Rh, ph = hand_frame(H, hname, allf, r, axis, R_obj=R6)
        Ro, to = apply_G(Rh, ph, RG1, tG1)
        Rf, tf = R6.copy(), t6.copy()
        Rf[:T], tf[:T] = Ro, to
        srcR.append(Rf)
        srcT.append(tf)
    # distance (frames) to the nearest unlocked frame
    idx0 = np.flatnonzero(lab == 0)
    d0 = np.abs(np.arange(T6)[:, None] - idx0[None]).min(1) if len(idx0) else np.full(T6, 1e9)
    u = 1 - np.exp(-d0.astype(float) ** 2 / (2 * a.sigma_blend ** 2))
    u[lab == 0] = 0
    V = np.stack([gaussian_filter1d((lab == k).astype(float), a.sigma_blend, mode="nearest")
                  for k in range(1, len(accepted) + 1)])
    V[:, lab > 0] += 1e-6 * (lab[lab > 0][None] == np.arange(1, len(accepted) + 1)[:, None])
    V = V / np.maximum(V.sum(0, keepdims=True), 1e-12)
    W = np.concatenate([(1 - u)[None], u[None] * V])
    W[:, lab == 0] = 0
    W[0, lab == 0] = 1
    tn = np.einsum("kt,kti->ti", W, np.stack(srcT))
    Rn = blend_quats(np.stack(srcR), W)
    Rn[lab == 0], tn[lab == 0] = R6[lab == 0], t6[lab == 0]
    # whole-episode before / after on the locked frames
    lf = np.flatnonzero(locked[:T])
    iou6, kz6 = S.series(lf, R6[lf], t6[lf])
    iou7, kz7 = S.series(lf, Rn[lf], tn[lf])
    gb, ga = [], []
    for (sa, sb, hname, RG1, tG1, anchor) in accepted:
        fr = np.arange(sa, sb)
        Hv = np.concatenate([H["V"][h][fr] * r for h in anchor], 1)
        gb.append(S.hand_sd(R6[fr], t6[fr], Hv)[0])
        ga.append(S.hand_sd(Rn[fr], tn[fr], Hv)[0])
    log["locked_before"] = summarize(iou6, kz6, np.concatenate(gb))
    log["locked_after"] = summarize(iou7, kz7, np.concatenate(ga))
    lid = lab - 1
    return finish(Rn, tn, locked, lid)


def write_summary(out, logs):
    json.dump(logs, open(os.path.join(out, "summary.json"), "w"), indent=1)
    L = ["# objects_v7 grasp lock", "",
         "Locked frames: accepted segments. IoU / dz (median |kz-1|) / gap (median hand-surface gap, body cm) "
         "on the locked frames; acc = mean |2nd difference| of t on scored frames (body cm/frame^2), raw and "
         "after the sigma-12 smoothing of assemble_episode.py; shift = mean |t7 - t6| on scored frames (cm).", "",
         "| ep | class | segs acc/all | locked (scored) | IoU v6 -> v7 | dz v6 -> v7 | gap v6 -> v7 | acc raw v6 -> v7 | acc s12 v6 -> v7 | shift | note |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(x, n=3):
        return "-" if x is None else f"{x:.{n}f}"
    for lg in logs:
        if lg.get("excluded"):
            L.append(f"| {lg['ep']} | {lg['cls']} | - | - | | | | | | | excluded: {lg['excluded']} |")
            continue
        segs = lg["segments"]
        na = sum(1 for x in segs if x.get("accepted"))
        b, af = lg.get("locked_before", {}), lg.get("locked_after", {})
        acc = lg["acc_scored"]
        hands = ",".join(f"{x['a']}-{x['b']}:{x.get('hand')}" for x in segs if x.get("accepted"))
        rej = ",".join(f"{x['a']}-{x['b']}:{x.get('reason')}" for x in segs if not x.get("accepted"))
        L.append(f"| {lg['ep']} | {lg['cls']} | {na}/{len(segs)} | {lg['n_locked']} ({lg['n_locked_scored']}) | "
                 f"{f(b.get('iou'))} -> {f(af.get('iou'))} | {f(b.get('dz'))} -> {f(af.get('dz'))} | "
                 f"{f(b.get('gap_cm'), 1)} -> {f(af.get('gap_cm'), 1)} | {f(acc['before'], 4)} -> {f(acc['after'], 4)} | "
                 f"{f(acc['before_s12'], 4)} -> {f(acc['after_s12'], 4)} | {f(lg['shift_scored_cm'], 1)} | "
                 f"{('locked ' + hands) if hands else ''}{('; kept ' + rej) if rej else ''} |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(WORK, "objects_v6"))
    ap.add_argument("--out", default=os.path.join(WORK, "objects_v7"))
    ap.add_argument("--body", default=os.path.join(WORK, "body_v2"))
    ap.add_argument("--ratio", default=os.path.join(WORK, "moge_to_body_ratio_floor.json"))
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--sigma-h", type=float, default=3.0, help="body pose smoothing (as assemble_episode.py)")
    ap.add_argument("--touch-px", type=float, default=4.0)
    ap.add_argument("--touch-cm", type=float, default=4.0)
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--refine-frames", type=int, default=32)
    ap.add_argument("--render-faces", type=int, default=1500, help="decimated mesh for the soft rasteriser")
    ap.add_argument("--w-gap", type=float, default=0.005)
    ap.add_argument("--w-pen", type=float, default=0.0005)
    ap.add_argument("--w-prior", type=float, default=0.02)
    ap.add_argument("--tol-iou", type=float, default=0.02)
    ap.add_argument("--tol-kz", type=float, default=0.005)
    ap.add_argument("--max-gap", type=float, default=3.0,
                    help="accept only locks whose median hand-surface gap (body cm) is at most this")
    ap.add_argument("--sigma-blend", type=float, default=4.0)
    ap.add_argument("--debug", action="store_true", help="save per-frame series of every candidate")
    ap.add_argument("--force", action="store_true", help="diagnostics only: accept every segment")
    ap.add_argument("--split-pen", type=float, default=3.0,
                    help="> 0: split grasp segments at regrasps (change points of the object centre in the hand frame); 0: off")
    ap.add_argument("--split-min", type=int, default=30, help="minimum length (frames) of a split piece")
    ap.add_argument("--no-point", dest="point", action="store_false",
                    help="strict lock only (no grasp-point candidates)")
    ap.add_argument("--recompose", default=None,
                    help="output dir of an earlier run: reuse its segments and G, re-decide acceptance, compose")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    ratio = json.load(open(a.ratio))["ratio"]
    scored = json.load(open(os.path.join(WORK, "scored_frames.json")))
    meta = load_meta()
    mhr = MHRForward(DEFAULT_MHR, device="cuda")
    hids = hand_vertex_split(mhr.model)
    logs = []
    for ep in a.episodes:
        lg = process(ep, a, mhr, hids, scored, ratio, meta)
        logs.append(lg)
        na = sum(1 for x in lg["segments"] if x.get("accepted"))
        print(f"ep {ep:2d} {lg['cls']:18s} segs {na}/{len(lg['segments'])} locked {lg['n_locked']} "
              f"({lg['n_locked_scored']} scored) acc {lg['acc_scored']} shift {lg['shift_scored_cm']:.1f} cm "
              f"{lg['time_s']} s {('EXCLUDED ' + lg['excluded']) if lg['excluded'] else ''}", flush=True)
        torch.cuda.empty_cache()
    # merge with logs of episodes processed earlier (partial runs)
    allp = sorted(os.listdir(os.path.join(a.out, "logs")))
    write_summary(a.out, [json.load(open(os.path.join(a.out, "logs", p))) for p in allp if p.endswith(".json")])


if __name__ == "__main__":
    main()
