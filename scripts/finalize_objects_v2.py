"""Contact-based depth correction + final v2 object files in the v1 layout.

Silhouettes fix an object's pose only up to a scaling about the camera centre: (s, t) and
(k s, k t) project identically. MoGe depth and the reconstructed mesh scale decide k, and
errors there put the object in front of / behind the actor. While the object moves it is
held, so the body (mostly the hands) must touch it. For each episode the body (work/body_v2, MHR,
mapped into the MoGe frame with the per-episode depth ratio) is compared with the posed object
on frames where the object moves and its mask touches the actor mask; k is the value closest
to 1 whose median body-object gap is within 1 cm of the best gap (range 0.8-1.25), used when it
closes the gap by more than 1 cm. Episodes without enough contact evidence keep k = 1 unless
their body-frame object scale is more than 15 % off the median of the class's episodes, in
which case k brings it to that median (the same physical object appears in all three).

Inputs : WORK/objects_v2/track/episode_XXXXXX.npz (track_object_v2.py)
Outputs: WORK/objects_v2/episode_XXXXXX_object.ply, _pose.json (key "s"), _track.npz
         (R, t, valid, scale, iou, ref_frame), same convention and frame as WORK/objects (raw MoGe).
         WORK/objects_v2/contact_scale.json

Runs in the sam3do env (MHR TorchScript + GPU). Usage: python finalize_objects_v2.py [--episodes ...] [--no-contact]
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, load_meta, unpack_masks  # noqa: E402

V2 = os.environ.get("OBJ_V2", os.path.join(WORK, "objects_v2"))


BODY_DIR = os.path.join(WORK, "body_v2")
RATIO_JSON = os.path.join(WORK, "moge_to_body_ratio_v2.json")


def body_vertices(ep, mhr, frames):
    """Body vertices (body frame, metres) of the given frames; body shape/scales = episode median."""
    b = np.load(os.path.join(BODY_DIR, f"episode_{ep:06d}.npz"))
    valid = b["valid"].astype(bool)
    sc = b["scales"] if b["scales"].ndim == 1 else np.median(b["scales"][valid], 0)
    sh = b["shape"] if b["shape"].ndim == 1 else np.median(b["shape"][valid], 0)
    frames = frames[valid[frames]]
    v, _ = mhr(b["pose"][frames], sc, sh)
    return v, frames


def contact_frames(ep, tr, min_speed=0.004):
    md = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
    h, w = int(md["mask_h"]), int(md["mask_w"])
    om = unpack_masks(md["obj_masks"], h, w)
    pm = unpack_masks(md["person_masks"], h, w)
    T = len(om)
    t = tr["t"]
    speed = np.linalg.norm(np.gradient(t, axis=0), axis=1)
    moving = (tr["seg_id"] < 0) & (speed > min_speed) & tr["valid"]
    ker = np.ones((11, 11), np.uint8)
    touch = np.array([(cv2.dilate(pm[f].astype(np.uint8), ker) > 0)[om[f]].any() if moving[f] else False
                      for f in range(T)])
    return np.flatnonzero(moving & touch)


def choose_k(P, R, t, s, verts, frames, ks):
    from scipy.spatial import cKDTree
    D = np.zeros((len(ks), len(frames)))
    for j, f in enumerate(frames):
        tree = cKDTree(verts[f])
        Pc = s * P @ R[f].T + t[f]
        for i, k in enumerate(ks):
            d = tree.query(Pc[::3] * k, distance_upper_bound=0.5)[0]
            D[i, j] = min(float(d.min()), 0.5)
    med = np.median(D, 1)
    ok = np.flatnonzero(med <= med.min() + 0.01)
    i = ok[np.argmin(np.abs(np.log(ks[ok])))]
    return float(ks[i]), med


def contact_k(ep, tr, mhr, ratio, ks, min_frames):
    import trimesh
    fr = contact_frames(ep, tr)
    if len(fr) < min_frames:
        return None, f"only {len(fr)} contact frames"
    fr = fr[:: max(1, len(fr) // 80)]
    verts, fr = body_vertices(ep, mhr, fr)
    P = np.asarray(trimesh.sample.sample_surface(trimesh.load(str(tr["mesh"]), process=False), 3000, seed=0)[0])
    k_best, med = choose_k(P, tr["R"][fr], tr["t"][fr], float(tr["s"]), verts * ratio, np.arange(len(fr)), ks)
    d1 = float(np.interp(0.0, np.log(ks), med))
    dk = float(med[np.argmin(np.abs(ks - k_best))])
    info = f"{len(fr)} contact frames, gap {d1 * 100:.1f} cm at k=1 -> {dk * 100:.1f} cm at k={k_best:.3f}"
    if d1 - dk > 0.01:
        return k_best, info
    return 1.0, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--no-contact", action="store_true")
    ap.add_argument("--min-frames", type=int, default=20)
    ap.add_argument("--size-tol", type=float, default=1.15,
                    help="episodes without contact evidence whose body-frame size is off the class median "
                         "by more than this factor are rescaled (about the camera) to the median")
    ap.add_argument("--body", default=BODY_DIR, help="MHR body estimates (body frame)")
    ap.add_argument("--ratio", default=RATIO_JSON, help="per-episode MoGe/body depth ratio json")
    a = ap.parse_args()
    globals()["BODY_DIR"] = a.body
    import trimesh
    meta = load_meta()
    ratio = json.load(open(a.ratio))["ratio"]
    mhr = None
    if not a.no_contact:
        from mhr_utils import DEFAULT_MHR, MHRForward
        mhr = MHRForward(DEFAULT_MHR, device="cuda" if torch.cuda.is_available() else "cpu")
    ks = np.exp(np.linspace(np.log(0.8), np.log(1.25), 46))
    log_path = os.path.join(V2, "contact_scale.json")
    summary = json.load(open(log_path)) if os.path.exists(log_path) else {}
    tracks, ks_ep, infos = {}, {}, {}
    for ep in a.episodes:
        p = os.path.join(V2, "track", f"episode_{ep:06d}.npz")
        if not os.path.exists(p):
            print(f"[ep {ep}] no v2 track", flush=True)
            continue
        tracks[ep] = dict(np.load(p))
        if mhr is not None:
            ks_ep[ep], infos[ep] = contact_k(ep, tracks[ep], mhr, ratio[str(ep)], ks, a.min_frames)
        else:
            ks_ep[ep], infos[ep] = None, "no contact step"
    # class size consensus (body frame) for episodes without contact evidence
    by_cls = {}
    for ep in tracks:
        by_cls.setdefault(meta[ep]["object"], []).append(ep)
    for cls, eps in by_cls.items():
        sizes = {ep: float(tracks[ep]["s"]) * (ks_ep[ep] or 1.0) / ratio[str(ep)] for ep in eps}
        med = float(np.median(list(sizes.values())))
        for ep in eps:
            if ks_ep[ep] is None:
                f = med / sizes[ep]
                if abs(np.log(f)) > np.log(a.size_tol):
                    ks_ep[ep] = f
                    infos[ep] += f"; size {sizes[ep]:.3f} vs class median {med:.3f} -> k={f:.3f}"
    for ep, tr in tracks.items():
        k = ks_ep[ep] or 1.0
        mesh = str(tr["mesh"])
        R, t, s = tr["R"], tr["t"], float(tr["s"])
        stem = os.path.join(V2, f"episode_{ep:06d}")
        shutil.copy(mesh, stem + "_object.ply")
        sk, tk = s * k, t * k
        iou = tr["iou"]
        ref = int(np.argmax(np.where(tr["valid"], iou, -1)))
        Vm = np.asarray(trimesh.load(mesh, process=False).vertices)
        pose = dict(episode=ep, object=meta[ep]["object"], frame=ref,
                    convention="point_cam = s * R @ point_mesh + t; camera frame OpenCV (x right, y down, "
                               "z forward); metres (raw MoGe-2 frame); point_mesh = vertices of "
                               "episode_XXXXXX_object.ply",
                    R=R[ref].tolist(), t=tk[ref].tolist(), s=sk, contact_k=k, contact_info=infos[ep],
                    mesh_source=mesh, iou=float(iou[ref]),
                    object_extent_m=(sk * (Vm.max(0) - Vm.min(0))).tolist())
        json.dump(pose, open(stem + "_pose.json", "w"), indent=1)
        np.savez(stem + "_track.npz", R=R, t=tk, valid=tr["valid"], scale=sk, iou=iou, ref_frame=ref,
                 seg_id=tr["seg_id"])
        summary[str(ep)] = dict(k=k, info=infos[ep], scale_moge=sk, scale_body=float(sk / ratio[str(ep)]))
        print(f"[ep {ep}] k={k:.3f} ({infos[ep]}) s={sk:.3f} body-frame s={sk / ratio[str(ep)]:.3f}", flush=True)
    json.dump(summary, open(log_path, "w"), indent=1)

if __name__ == "__main__":
    main()
