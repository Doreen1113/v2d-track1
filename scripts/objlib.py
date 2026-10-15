"""Shared helpers for the v2 object stage (shared meshes, batched render-and-compare, trajectories).

Conventions: point_cam = s * R @ point_mesh + t, OpenCV camera, metres in the raw MoGe-2 frame.
All image-space work is done at quarter resolution (288 x 384), the resolution of WORK/depth.
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, unpack_masks  # noqa: E402

V2 = os.environ.get("OBJ_V2", os.path.join(WORK, "objects_v2"))
DS = 4  # quarter resolution w.r.t. the 1536 x 1152 video


# ---------------------------------------------------------------- rotations
def so3_exp(w):
    """Batched axis-angle [...,3] -> rotation [...,3,3] (differentiable, safe at 0)."""
    theta = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / theta
    z = torch.zeros_like(k[..., 0])
    Kx = torch.stack([z, -k[..., 2], k[..., 1], k[..., 2], z, -k[..., 0], -k[..., 1], k[..., 0], z], -1)
    Kx = Kx.reshape(*w.shape[:-1], 3, 3)
    th = theta[..., None]
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(Kx)
    return eye + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


# ---------------------------------------------------------------- meshes
def load_mesh(path, target_faces=2000):
    import trimesh
    m = trimesh.load(path, process=False)
    V, F = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    if len(F) > target_faces:
        try:
            import open3d as o3d
            om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
            om = om.simplify_quadric_decimation(target_faces)
            V, F = np.asarray(om.vertices), np.asarray(om.triangles)
        except Exception as e:  # fall back to the full mesh
            print("decimation failed:", e)
    return V, F


def surface_points(path, n=4000, seed=0):
    import trimesh
    m = trimesh.load(path, process=False)
    return np.asarray(trimesh.sample.sample_surface(m, n, seed=seed)[0])


# ---------------------------------------------------------------- episode data
class Episode:
    """Quarter-resolution masks, person masks and scale-stabilised MoGe depth of one episode."""

    def __init__(self, ep, load_depth=True):
        self.ep = ep
        md = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
        h, w = int(md["mask_h"]), int(md["mask_w"])
        self.T = int(md["num_frames"])
        om = unpack_masks(md["obj_masks"], h, w)
        pm = unpack_masks(md["person_masks"], h, w)
        h2, w2 = h // 2, w // 2
        self.obj = om[:, :h2 * 2, :w2 * 2].reshape(self.T, h2, 2, w2, 2).any((2, 4))
        self.per = pm[:, :h2 * 2, :w2 * 2].reshape(self.T, h2, 2, w2, 2).any((2, 4))
        self.obj_half = om
        self.H, self.W = h2, w2
        self.score = md["obj_score"]
        self.area = self.obj.sum((1, 2))
        self.visible = (self.score > 0) & (self.area > 8)
        # pixels where the actor may hide the object: not compared
        self.occ = self.per & ~self.obj
        self.K_full = np.array([[807.66, 0, 768.0], [0, 807.66, 576.0], [0, 0, 1]])
        kp = os.path.join(WORK, "intrinsics", f"episode_{ep:06d}.json")
        if os.path.exists(kp):
            d = json.load(open(kp))
            self.K_full = np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]], np.float64)
        self.K = self.K_full.copy()
        self.K[:2] /= DS
        self.depth = None
        if load_depth:
            self.depth, self.depth_scale = load_stable_depth(ep, self)
        self._dt = {}

    def dt_out(self, f):
        """Distance (px) to the nearest object-mask pixel; 0 inside the mask or where the actor is."""
        if f not in self._dt:
            m = self.obj[f]
            if m.any():
                d = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
            else:
                d = np.zeros(m.shape, np.float32)
            d[self.occ[f]] = 0
            self._dt[f] = np.minimum(d, 30.0).astype(np.float32)
        return self._dt[f]


def load_stable_depth(ep, E):
    """Per-frame MoGe depth divided by its per-frame scale relative to the static background.

    The camera is fixed, so background depth should not change; MoGe's per-frame scale jitter
    (about 0.5 % frame to frame) is estimated on background pixels and removed (smoothed ratio).
    """
    from scipy.ndimage import gaussian_filter1d
    D = np.load(os.path.join(WORK, "depth", f"episode_{ep:06d}.npz"))["depth"].astype(np.float32)
    T = len(D)
    fg = E.obj | E.per
    ker = np.ones((9, 9), np.uint8)
    sub = np.linspace(0, T - 1, min(T, 120)).astype(int)
    ref = np.median(D[sub], 0)
    r = np.ones(T)
    for t in range(T):
        bg = ~(cv2.dilate(fg[t].astype(np.uint8), ker) > 0) & (ref > 0) & (D[t] > 0)
        if bg.sum() > 500:
            r[t] = np.median(D[t][bg] / ref[bg])
    r = gaussian_filter1d(r, 1.0, mode="nearest")
    return D / r[:, None, None].astype(np.float32), r


# ---------------------------------------------------------------- rendering
class Renderer:
    """Batched soft-silhouette + z-buffer rasteriser (PyTorch3D) for an OpenCV pinhole camera."""

    def __init__(self, K, H, W, device="cuda", faces_per_pixel=24, sigma=1e-6):
        from pytorch3d.renderer import BlendParams, MeshRasterizer, RasterizationSettings, SoftSilhouetteShader
        from pytorch3d.utils import cameras_from_opencv_projection
        self.device = device
        self.K, self.H, self.W = K, H, W
        self.cam = cameras_from_opencv_projection(
            R=torch.eye(3, device=device)[None], tvec=torch.zeros(1, 3, device=device),
            camera_matrix=torch.tensor(K, dtype=torch.float32, device=device)[None],
            image_size=torch.tensor([[H, W]], device=device))
        # A small sigma matters: with sub-pixel triangles a wide blur band fills the K-face list with
        # nearby non-covering faces and the soft silhouette shrinks (IoU vs. hard raster 0.4-0.8
        # at sigma 1e-5, >= 0.9 at 1e-6). Long-range gradients come from the distance term instead.
        rs = RasterizationSettings(image_size=(H, W), blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
                                   faces_per_pixel=faces_per_pixel, bin_size=0)
        self.rast = MeshRasterizer(cameras=self.cam, raster_settings=rs)
        self.shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=sigma, gamma=1e-4))

    def __call__(self, verts, faces):
        """verts [B,V,3] (camera frame), faces [F,3] -> sil [B,H,W], z [B,H,W] (-1 where empty)."""
        from pytorch3d.structures import Meshes
        B = verts.shape[0]
        mesh = Meshes(verts=verts, faces=faces[None].expand(B, -1, -1))
        frag = self.rast(mesh, cameras=self.cam)
        sil = self.shader(frag, mesh, cameras=self.cam)[..., 3]
        return sil, frag.zbuf[..., 0]


def hard_silhouette(Vc, F, K, H, W):
    z = np.clip(Vc[:, 2], 1e-4, None)
    uv = np.stack([K[0, 0] * Vc[:, 0] / z + K[0, 2], K[1, 1] * Vc[:, 1] / z + K[1, 2]], 1)
    ok = (Vc[F, 2] > 1e-3).all(1)
    polys = np.round(uv[F[ok]] * 4).astype(np.int32)
    sil = np.zeros((H, W), np.uint8)
    cv2.fillPoly(sil, list(polys), 1, lineType=cv2.LINE_8, shift=2)
    return sil > 0


def frame_iou(E, f, Vc, F, half=False):
    """Silhouette IoU against the object mask, ignoring actor pixels (quarter or half resolution)."""
    if half:
        K = E.K_full.copy(); K[:2] /= 2
        m = E.obj_half[f]
        occ = cv2.resize(E.occ[f].astype(np.uint8), (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        sil = hard_silhouette(Vc, F, K, m.shape[0], m.shape[1])
    else:
        m, occ = E.obj[f], E.occ[f]
        sil = hard_silhouette(Vc, F, E.K, E.H, E.W)
    keep = ~occ
    return float((sil & m & keep).sum() / max(((sil | m) & keep).sum(), 1))


# ---------------------------------------------------------------- losses
class FrameTargets:
    """GPU tensors of the per-frame targets for a set of frames."""

    def __init__(self, E, frames, device="cuda", use_depth=True):
        fr = np.asarray(frames)
        self.frames = fr
        self.m = torch.tensor(E.obj[fr], device=device, dtype=torch.float32)
        self.keep = torch.tensor(~E.occ[fr], device=device, dtype=torch.float32)
        self.dt = torch.tensor(np.stack([E.dt_out(int(f)) for f in fr]), device=device)
        self.has = torch.tensor(E.visible[fr], device=device, dtype=torch.float32)
        # up to 400 non-occluded mask pixels per frame (x, y) for the 2-D chamfer term
        rng = np.random.default_rng(0)
        mp = np.zeros((len(fr), 400, 2), np.float32)
        mv = np.zeros((len(fr), 400), np.float32)
        for i, f in enumerate(fr):
            ys, xs = np.nonzero(E.obj[f])
            if len(xs):
                k = rng.choice(len(xs), min(400, len(xs)), replace=False)
                mp[i, :len(k)] = np.stack([xs[k], ys[k]], 1)
                mv[i, :len(k)] = 1
        self.mpix = torch.tensor(mp, device=device)
        self.mvalid = torch.tensor(mv, device=device)
        self.d = None
        if use_depth and E.depth is not None:
            ker = np.ones((3, 3), np.uint8)
            dd = []
            for f in fr:
                inner = cv2.erode(E.obj[f].astype(np.uint8), ker) > 0
                inner &= ~E.per[f]
                x = E.depth[f].copy()
                x[~inner] = 0
                dd.append(x)
            self.d = torch.tensor(np.stack(dd), device=device)

    def subset(self, idx):
        o = FrameTargets.__new__(FrameTargets)
        o.frames = self.frames[idx.cpu().numpy() if torch.is_tensor(idx) else idx]
        for k in ("m", "keep", "dt", "has", "d", "mpix", "mvalid"):
            v = getattr(self, k)
            setattr(o, k, None if v is None else v[idx])
        return o


def chamfer2d(uv, tg, clip=30.0):
    """Symmetric 2-D chamfer (pixels) between projected mesh points uv [B,M,2] and the mask.

    points -> mask: distance-to-mask map sampled at the points (0 inside the mask or on the actor);
    mask -> points: nearest projected point for sampled mask pixels. Gives smooth long-range
    gradients for thin objects (hoop) whose soft silhouette overlap is only a few pixels wide.
    """
    B, H, W = tg.dt.shape
    g = torch.stack([uv[..., 0] / (W - 1) * 2 - 1, uv[..., 1] / (H - 1) * 2 - 1], -1)[:, :, None]
    d_pm = torch.nn.functional.grid_sample(tg.dt[:, None], g, align_corners=True, padding_mode="border")[:, 0, :, 0]
    d_pm = d_pm.mean(1)
    dd = torch.cdist(tg.mpix, uv).amin(2).clamp(max=clip)
    d_mp = (dd * tg.mvalid).sum(1) / tg.mvalid.sum(1).clamp_min(1)
    return d_pm + d_mp


def project_points(P, K):
    """P [B,M,3] camera points -> pixel coordinates [B,M,2] (torch)."""
    z = P[..., 2].clamp_min(1e-3)
    return torch.stack([K[0, 0] * P[..., 0] / z + K[0, 2], K[1, 1] * P[..., 1] / z + K[1, 2]], -1)


def data_loss(sil, z, tg, zc=None, w_dt=0.03, w_depth=0.5, thin=False, uv=None, w_ch=0.05):
    """Per-frame loss [B]: soft IoU (actor pixels ignored) + outside-distance + depth.

    Depth: the rendered surface depth is compared with MoGe (medians over the object pixels seen
    in both), and the gap is applied to the object's centre depth zc [B] only, i.e. it moves the
    object along the optical axis. (Back-propagating through the z-buffer instead gives erratic
    rotation gradients and makes the fit diverge.)
    """
    m, keep = tg.m, tg.keep
    inter = (sil * m * keep).sum((1, 2))
    union = ((sil + m - sil * m) * keep).sum((1, 2)) + 1e-6
    loss = 1 - inter / union
    # pulls stray silhouette towards the mask (gradient also when there is no overlap: thin objects)
    area = sil.sum((1, 2)) + 1.0
    loss = loss + w_dt * (sil * tg.dt).sum((1, 2)) / area
    if uv is not None and w_ch > 0:
        loss = loss + w_ch * chamfer2d(uv, tg)
    if zc is not None and tg.d is not None and w_depth > 0 and not thin:
        with torch.no_grad():
            gap = torch.zeros_like(zc)
            ok = torch.zeros_like(zc, dtype=torch.bool)
            for i in range(len(zc)):
                vis = (z[i] > 0) & (tg.d[i] > 0)
                if vis.sum() > 15:
                    gap[i] = torch.median(z[i][vis]) - torch.median(tg.d[i][vis])
                    ok[i] = True
        e = (zc - zc.detach() + gap).abs()
        hub = torch.where(e < 0.05, 0.5 * e ** 2 / 0.05, e - 0.025)
        loss = loss + w_depth * torch.where(ok, hub, torch.zeros_like(hub))
    return loss * tg.has


# ---------------------------------------------------------------- trajectories
def accel_penalty(x):
    """Sum of squared second differences along dim 0."""
    if x.shape[0] < 3:
        return x.sum() * 0
    a = x[:-2] - 2 * x[1:-1] + x[2:]
    return (a ** 2).sum()


def load_gray_half(ep):
    from common import VIDEO_TMPL
    cap = cv2.VideoCapture(VIDEO_TMPL.format(ep))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(out)


def static_segments(E, min_len=15, tau=0.85, dil=1, min_cmp=40, gray=None, app_thr=7.0):
    """Segments [a, b) where the object mask does not change (object at rest).

    Half-resolution masks are compared (after a 1-px dilation, robust for thin objects) only on
    pixels not covered by the actor, both against the segment's first frame and against the
    last confirmed frame. A frame with too few comparable pixels neither confirms nor breaks a
    segment. If half-resolution grey frames are given, the appearance inside the (eroded) mask
    must also stay the same (catches objects held still in place but rotating, e.g. a textured
    block turned in the hands). Returns a list of (a, b) and a per-frame segment id (-1 = moving).
    """
    T = E.T
    ker = np.ones((2 * dil + 1, 2 * dil + 1), np.uint8)
    om = E.obj_half
    h, w = om.shape[1:]
    md = np.stack([cv2.dilate(m.astype(np.uint8), ker) > 0 for m in om])
    me = np.stack([cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0 for m in om])
    pd = np.stack([cv2.dilate(cv2.resize(p.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST),
                              np.ones((7, 7), np.uint8)) > 0 for p in E.per])

    def sim(f, g):
        keep = ~(pd[f] | pd[g])
        a, b = md[f] & keep, md[g] & keep
        u = (a | b).sum()
        if u < min_cmp or a.sum() < 0.3 * md[f].sum() or b.sum() < 0.3 * md[g].sum():
            return None
        iou = (a & b).sum() / u
        if gray is not None and iou >= tau:
            inner = me[f] & me[g] & keep
            if inner.sum() >= 20:
                diff = np.abs(gray[f][inner].astype(np.float32) - gray[g][inner].astype(np.float32))
                if np.median(diff) > app_thr:
                    return 0.0
        return iou

    seg_id = -np.ones(T, int)
    segs = []
    f = 0
    while f < T:
        if not E.visible[f]:
            f += 1
            continue
        a = f
        last_ok = f
        g = f + 1
        while g < T:
            s = sim(a, g) if E.visible[g] else None
            if s is None:
                if g - last_ok > 45:  # long gap: stop
                    break
                g += 1
                continue
            s2 = sim(last_ok, g)
            if s < tau or (s2 is not None and s2 < tau):
                break
            last_ok = g
            g += 1
        b = last_ok + 1
        if b - a >= min_len:
            segs.append((a, b))
            seg_id[a:b] = len(segs) - 1
            f = b
        else:
            f += 1
    return segs, seg_id


def load_ratio():
    return json.load(open(os.path.join(WORK, "moge_to_body_ratio_v2.json")))["ratio"]


def class_of(ep):
    from common import load_meta
    return load_meta()[ep]["object"]
