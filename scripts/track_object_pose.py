"""Per-frame object pose tracking by render-and-compare (skeleton / first version).

Starting from the SAM 3D Objects pose at its reference frame, the pose is propagated
forward and backward frame by frame. At each frame the rotation (axis-angle increment)
and translation are optimised with a soft-silhouette IoU loss against the tracked mask
(pixels covered by the actor are ignored, so occluded parts are not penalised), a depth
term against the per-frame MoGe-2 depth inside the visible mask, and a small motion prior.
Frames where the object is not visible are interpolated afterwards and the trajectory is
smoothed. Scale is kept fixed (from the reference reconstruction).

Inputs:  WORK/objects/episode_XXXXXX_{object.ply,pose.json}, WORK/masks, WORK/depth
Output:  WORK/objects/track/episode_XXXXXX.npz
           R [T,3,3], t [T,3] (point_cam = s * R @ point_mesh + t, OpenCV camera, metres),
           s, valid [T] (optimised from an observation), iou [T], R_raw, t_raw (before smoothing)

Usage: python track_object_pose.py --episodes 16 [--iters 40]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, unpack_masks  # noqa: E402


def so3_exp(w):
    theta = torch.linalg.norm(w) + 1e-12
    k = w / theta
    Kx = torch.zeros(3, 3, dtype=w.dtype, device=w.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 0] = -k[2], k[1], k[2]
    Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[0], -k[1], k[0]
    I = torch.eye(3, dtype=w.dtype, device=w.device)
    return I + torch.sin(theta) * Kx + (1 - torch.cos(theta)) * (Kx @ Kx)


def load_mesh(ep, target_faces=1500):
    import trimesh
    m = trimesh.load(os.path.join(WORK, "objects", f"episode_{ep:06d}_object.ply"), process=False)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    if len(F) > target_faces:
        try:
            import open3d as o3d
            om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
            om = om.simplify_quadric_decimation(target_faces)
            V, F = np.asarray(om.vertices), np.asarray(om.triangles)
        except Exception as e:  # fall back to the full mesh
            print("decimation failed:", e)
    return V.astype(np.float32), F.astype(np.int64)


class Renderer:
    def __init__(self, K, H, W, device):
        from pytorch3d.renderer import (BlendParams, MeshRasterizer, RasterizationSettings,
                                        SoftSilhouetteShader)
        from pytorch3d.utils import cameras_from_opencv_projection
        self.cam = cameras_from_opencv_projection(
            R=torch.eye(3, device=device)[None], tvec=torch.zeros(1, 3, device=device),
            camera_matrix=torch.tensor(K, dtype=torch.float32, device=device)[None],
            image_size=torch.tensor([[H, W]], device=device))
        # sigma acts on squared NDC distances (per triangle); 1e-5 keeps small triangles opaque
        sigma = 1e-5
        rs = RasterizationSettings(image_size=(H, W), blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
                                   faces_per_pixel=30, bin_size=0)
        self.rast = MeshRasterizer(cameras=self.cam, raster_settings=rs)
        self.shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=sigma, gamma=1e-4))

    def __call__(self, verts, faces):
        from pytorch3d.structures import Meshes
        mesh = Meshes(verts=[verts], faces=[faces])
        frag = self.rast(mesh)
        sil = self.shader(frag, mesh)[0, ..., 3]
        z = frag.zbuf[0, ..., 0]
        return sil, z


def track(ep, iters=40, device="cuda", wide_reinit=False):
    pose = json.load(open(os.path.join(WORK, "objects", f"episode_{ep:06d}_pose.json")))
    md = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
    h, w = int(md["mask_h"]), int(md["mask_w"])
    T = int(md["num_frames"])
    obj_masks_vis = unpack_masks(md["obj_masks"], h, w)
    # optimise at quarter resolution (2x2 "any"-pooling of the half-resolution masks)
    T_ = obj_masks_vis.shape[0]
    obj_masks = obj_masks_vis[:, :h // 2 * 2, :w // 2 * 2].reshape(T_, h // 2, 2, w // 2, 2).any((2, 4))
    pm = unpack_masks(md["person_masks"], h, w)
    per_masks = pm[:, :h // 2 * 2, :w // 2 * 2].reshape(T_, h // 2, 2, w // 2, 2).any((2, 4))
    del pm
    h, w = h // 2, w // 2
    score = md["obj_score"]
    dpath = os.path.join(WORK, "depth", f"episode_{ep:06d}.npz")
    depth = np.load(dpath)["depth"] if os.path.exists(dpath) else None

    full_w = int(pose["width"])
    sc = w / full_w
    K = np.array(pose["K"], np.float64)
    K[:2] *= sc
    s = float(pose["s"])
    V, F = load_mesh(ep)
    Vt = torch.tensor(V, device=device) * s
    Ft = torch.tensor(F, device=device)
    ren = Renderer(K, h, w, device)

    R_all = np.zeros((T, 3, 3)); t_all = np.zeros((T, 3))
    valid = np.zeros(T, bool); ious = np.zeros(T)
    t0 = pose["frame"]
    R0 = np.array(pose["R"]); tr0 = np.array(pose["t"])

    def fit(fi, R_init, t_init, iters=iters):
        m = torch.tensor(obj_masks[fi], device=device, dtype=torch.float32)
        occ = torch.tensor(per_masks[fi] & ~obj_masks[fi], device=device)
        keep = (~occ).float()
        Rb = torch.tensor(R_init, device=device, dtype=torch.float32)
        tb = torch.tensor(t_init, device=device, dtype=torch.float32)
        wv = torch.zeros(3, device=device, requires_grad=True)
        dt = torch.zeros(3, device=device, requires_grad=True)
        opt = torch.optim.Adam([{"params": [wv], "lr": 0.02}, {"params": [dt], "lr": 0.01}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-4)
        dref = None
        if depth is not None:
            d = cv2.resize(depth[fi].astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            # sample depth only well inside the mask (MoGe depth bleeds across object boundaries)
            inner = cv2.erode(obj_masks[fi].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            d[~inner] = 0
            dref = torch.tensor(d, device=device)
        for _ in range(iters):
            R = so3_exp(wv) @ Rb
            verts = Vt @ R.T + tb + dt
            sil, z = ren(verts, Ft)
            inter = (sil * m * keep).sum()
            union = ((sil + m - sil * m) * keep).sum() + 1e-6
            loss = 1 - inter / union
            if dref is not None:
                vis = (z > 0) & (m > 0) & (dref > 0) & (~occ)
                if vis.sum() > 20:
                    # robust: align the median rendered depth with the median observed depth
                    loss = loss + 0.5 * torch.abs(torch.median(z[vis]) - torch.median(dref[vis]))
            loss = loss + 1e-3 * (dt ** 2).sum() + 1e-3 * (wv ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            R = so3_exp(wv) @ Rb
            verts = Vt @ R.T + tb + dt
            from reconstruct_object import silhouette
            sb = silhouette(verts.cpu().numpy().astype(np.float64), F, K, h, w)
            mm = obj_masks[fi]
            kp = ~(per_masks[fi] & ~mm)  # ignore pixels where the actor may occlude the object
            iou = (sb & mm & kp).sum() / max(((sb | mm) & kp).sum(), 1)
        return R.detach().cpu().numpy().astype(np.float64), (tb + dt).detach().cpu().numpy().astype(np.float64), float(iou)

    def reinit_candidates(fi, Rp, tp):
        from scipy.spatial.transform import Rotation
        ys, xs = np.nonzero(obj_masks[fi])
        tc = tp.copy()
        if depth is not None:
            d = cv2.resize(depth[fi].astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)[ys, xs]
            d = d[d > 0]
            if len(d) > 10:
                z = float(np.median(d)) + 0.5 * s * float(np.ptp(V, 0).mean())  # surface -> centre
                tc = np.array([(xs.mean() - K[0, 2]) / K[0, 0] * z, (ys.mean() - K[1, 2]) / K[1, 1] * z, z])
        cands = [(Rp, tc), (R0, tc)]
        for ax in np.eye(3):
            for ang in ((np.pi, np.pi / 2, -np.pi / 2) if wide_reinit else (np.pi,)):
                cands.append((Rotation.from_rotvec(ax * ang).as_matrix() @ Rp, tc))
        return cands

    def visible(fi):
        a = obj_masks[fi].sum()
        return score[fi] > 0 and a > 30

    for order in (range(t0, T), range(t0 - 1, -1, -1)):
        Rp, tp = R0.copy(), tr0.copy()
        skip = 0
        for fi in order:
            if visible(fi):
                Rn, tn, iou = fit(fi, Rp, tp)
                if iou < 0.5 and skip <= 0:  # re-initialise from mask + depth, several rotations
                    improved = False
                    for Rc, tc in reinit_candidates(fi, Rp, tp):
                        Rc2, tc2, iou2 = fit(fi, Rc, tc, iters=20)
                        if iou2 > iou + 0.05:
                            Rn, tn, iou = Rc2, tc2, iou2
                            improved = True
                    if not improved:  # avoid paying for re-initialisation on every frame of a hard segment
                        skip = 15
                skip -= 1
                R_all[fi], t_all[fi], ious[fi] = Rn, tn, iou
                valid[fi] = iou > 0.3
                if valid[fi]:  # failed fits do not propagate
                    Rp, tp = Rn, tn
    R_raw, t_raw = R_all.copy(), t_all.copy()
    R_s, t_s = smooth_and_fill(R_all, t_all, valid)
    out = os.path.join(WORK, "objects", "track")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, f"episode_{ep:06d}.npz"), R=R_s, t=t_s, s=s, valid=valid, iou=ious,
             R_raw=R_raw, t_raw=t_raw, ref_frame=t0)
    # hand-off file next to the mesh: point_cam = scale * R[t] @ point_mesh + t[t] (OpenCV camera, MoGe metres)
    np.savez(os.path.join(WORK, "objects", f"episode_{ep:06d}_track.npz"), R=R_s, t=t_s, valid=valid, scale=s,
             iou=ious, ref_frame=t0)
    save_vis(ep, V, F, s, R_s, t_s, R_raw, t_raw, valid, np.array(pose["K"]), obj_masks_vis)
    return valid, ious


def save_vis(ep, V, F, s, R, t, R_raw, t_raw, valid, K, obj_masks):
    from common import VIDEO_TMPL, read_video
    from reconstruct_object import silhouette
    frames = read_video(VIDEO_TMPL.format(ep), scale=0.5)
    Kh = K.copy(); Kh[:2] *= 0.5
    T = len(frames); h, w = frames[0].shape[:2]
    tiles = []
    for fi in np.linspace(0, T - 1, 12).astype(int):
        im = frames[fi].copy()
        sil = silhouette(s * V @ R[fi].T + t[fi], F, Kh, h, w)
        im[sil] = (0.5 * im[sil] + [0, 127, 0]).astype(np.uint8)
        cnt, _ = cv2.findContours(obj_masks[fi].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cnt, -1, (255, 0, 0), 1)
        ys, xs = np.nonzero(sil | obj_masks[fi])
        if len(xs):
            cx, cy = int(xs.mean()), int(ys.mean())
            im = im[max(0, cy - 120):cy + 120, max(0, cx - 160):cx + 160]
        im = cv2.resize(im, (320, 240))
        cv2.putText(im, f"f{fi} {'obs' if valid[fi] else 'interp'}", (4, 18), 0, 0.55, (255, 255, 0), 2)
        tiles.append(im)
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)])
    os.makedirs(os.path.join(WORK, "objects", "track", "vis"), exist_ok=True)
    cv2.imwrite(os.path.join(WORK, "objects", "track", "vis", f"episode_{ep:06d}_track.jpg"), sheet[:, :, ::-1])


def smooth_and_fill(R, t, valid, sigma=2.0):
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation, Slerp
    T = len(t)
    idx = np.nonzero(valid)[0]
    if len(idx) == 0:
        return R, t
    tt = np.stack([np.interp(np.arange(T), idx, t[idx, k]) for k in range(3)], 1)
    if len(idx) > 1:
        sl = Slerp(idx, Rotation.from_matrix(R[idx]))
        q = sl(np.clip(np.arange(T), idx[0], idx[-1])).as_quat()
    else:
        q = np.repeat(Rotation.from_matrix(R[idx]).as_quat(), T, 0)
    for i in range(1, T):  # consistent quaternion hemisphere
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    q = gaussian_filter1d(q, sigma, axis=0, mode="nearest")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    tt = gaussian_filter1d(tt, sigma, axis=0, mode="nearest")
    return Rotation.from_quat(q).as_matrix(), tt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--shard", type=str, default=None)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--wide-reinit", action="store_true", help="also try +-90 deg rotations when re-initialising")
    a = ap.parse_args()
    eps = a.episodes if a.episodes else list(range(30))
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        eps = eps[i::n]
    for ep in eps:
        if not os.path.exists(os.path.join(WORK, "objects", f"episode_{ep:06d}_pose.json")):
            continue
        t0 = time.time()
        valid, ious = track(ep, a.iters, wide_reinit=a.wide_reinit)
        print(f"[ep {ep}] valid={valid.mean():.2f} median_iou={np.median(ious[valid]) if valid.any() else 0:.3f} "
              f"time={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
