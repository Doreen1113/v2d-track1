"""Mask/depth agreement of object trajectories (no ground truth: proxy metrics).

For an objects directory with episode_XXXXXX_object.ply + episode_XXXXXX_track.npz
(R [T,3,3], t [T,3], scale; raw MoGe frame), per frame at half resolution:
  iou    silhouette IoU (pixel-centre rasterisation) vs the object mask, actor pixels ignored
  dz     median |rendered depth - MoGe depth| over object pixels not covered by the actor (m)
and per episode: median IoU, fraction of frames with IoU < 0.5, median dz, mean |accel(t)| (cm).

Usage: python evaluate_object_tracks.py --objects WORK/objects_v2 --out eval_v2.json [--episodes ...]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import objlib  # noqa: E402
from common import WORK  # noqa: E402


def evaluate(ep, obj_dir, device="cuda"):
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection
    tr = np.load(os.path.join(obj_dir, f"episode_{ep:06d}_track.npz"))
    R, t, s = tr["R"], tr["t"], float(tr["scale"])
    E = objlib.Episode(ep)
    V, F = objlib.load_mesh(os.path.join(obj_dir, f"episode_{ep:06d}_object.ply"), 20000)
    h, w = E.obj_half.shape[1:]
    K = E.K_full.copy(); K[:2] /= 2
    cam = cameras_from_opencv_projection(
        R=torch.eye(3, device=device)[None], tvec=torch.zeros(1, 3, device=device),
        camera_matrix=torch.tensor(K, dtype=torch.float32, device=device)[None],
        image_size=torch.tensor([[h, w]], device=device))
    rast = MeshRasterizer(cameras=cam, raster_settings=RasterizationSettings(
        image_size=(h, w), blur_radius=0.0, faces_per_pixel=1, bin_size=0))
    Vt = torch.tensor(V, dtype=torch.float32, device=device)
    Ft = torch.tensor(F, dtype=torch.int64, device=device)
    iou = np.full(E.T, np.nan)
    dz = np.full(E.T, np.nan)
    B = 32
    for a in range(0, E.T, B):
        fr = np.arange(a, min(a + B, E.T))
        Rb = torch.tensor(R[fr], dtype=torch.float32, device=device)
        tb = torch.tensor(t[fr], dtype=torch.float32, device=device)
        verts = s * torch.einsum("vj,bij->bvi", Vt, Rb) + tb[:, None]
        with torch.no_grad():
            frag = rast(Meshes(verts=verts, faces=Ft[None].expand(len(fr), -1, -1)))
        sil = (frag.pix_to_face[..., 0] >= 0).cpu().numpy()
        z = frag.zbuf[..., 0].cpu().numpy()
        for k, f in enumerate(fr):
            if not E.visible[f]:
                continue
            m = E.obj_half[f]
            occ = cv2.resize(E.occ[f].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            keep = ~occ
            iou[f] = (sil[k] & m & keep).sum() / max(((sil[k] | m) & keep).sum(), 1)
            d = cv2.resize(E.depth[f], (w, h), interpolation=cv2.INTER_NEAREST)
            inner = (cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & keep & sil[k] & (d > 0)
            if inner.sum() > 20:
                dz[f] = np.median(np.abs(z[k][inner] - d[inner]))
    acc = np.linalg.norm(t[:-2] - 2 * t[1:-1] + t[2:], axis=1) * 100
    v = ~np.isnan(iou)
    return dict(median_iou=float(np.nanmedian(iou)), mean_iou=float(np.nanmean(iou)),
                frac_below_05=float((iou[v] < 0.5).mean()), median_dz=float(np.nanmedian(dz)) if np.isfinite(dz).any() else None,
                mean_acc_cm=float(acc.mean()), scale=s, iou=iou.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default=os.path.join(WORK, "objects"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    a = ap.parse_args()
    res = {}
    for ep in a.episodes:
        if not os.path.exists(os.path.join(a.objects, f"episode_{ep:06d}_track.npz")):
            continue
        r = evaluate(ep, a.objects)
        res[ep] = r
        print(f"ep {ep:2d} {objlib.class_of(ep):18s} medIoU {r['median_iou']:.3f} meanIoU {r['mean_iou']:.3f} "
              f"<0.5 {r['frac_below_05']:.2f} dz {r['median_dz'] if r['median_dz'] is None else round(r['median_dz'], 3)} "
              f"acc {r['mean_acc_cm']:.3f}cm s {r['scale']:.3f}", flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
