"""Object mesh + camera-frame pose per episode with SAM 3D Objects.

For each episode:
  1. pick K candidate frames from the tracked object masks (WORK/masks) where the object
     is large, a single blob, away from the image border and not overlapped by the person;
  2. compute a metric point map of that frame with MoGe-2 using the shared episode
     intrinsics (WORK/intrinsics, fx=fy, cx, cy) so the object lives in the same metric
     camera frame as the body reconstruction;
  3. run SAM 3D Objects (mesh + layout, layout post-optimisation on) with that point map;
  4. keep the candidate whose posed mesh silhouette best matches the mask (IoU).

Outputs (WORK/objects):
  episode_XXXXXX_object.ply / .glb   mesh vertices in the object's local frame (as stored)
  episode_XXXXXX_pose.json           point_cam = s * R @ point_mesh + t
                                     camera = OpenCV (x right, y down, z forward), metres
  vis/episode_XXXXXX_overlay.jpg     posed-mesh silhouette/edges reprojected on the frame

Usage: python reconstruct_object.py --episodes 0 1   (or --shard i/n)
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import VIDEO_TMPL, WORK, load_intrinsics, load_meta, read_frame, unpack_masks  # noqa: E402

SAM3DO_ROOT = os.environ.get("SAM3DO_ROOT", "/work/doreen071/v2d/third_party/sam-3d-objects")
# SAM 3D Objects mesh export rotates z-up -> y-up: v_glb = v_local @ A
A_GLB = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)
# PyTorch3D camera (x left, y up, z forward) <-> OpenCV camera
F_P3D = np.diag([-1.0, -1.0, 1.0])


def select_frames(ep, k=3, min_gap=20):
    d = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
    h, w = int(d["mask_h"]), int(d["mask_w"])
    om, pm = d["obj_masks"], d["person_masks"]
    area = d["obj_area"].astype(np.float64)
    score = d["obj_score"]
    T = len(area)
    vis = (area > 50) & (score > 0)
    if vis.sum() == 0:
        return [], d
    amax = np.percentile(area[vis], 95)
    ker = np.ones((9, 9), np.uint8)
    rows = []
    cand = np.nonzero(vis & (area > 0.3 * amax))[0]
    if len(cand) > 150:
        cand = cand[np.linspace(0, len(cand) - 1, 150).astype(int)]
    for t in cand:
        m = unpack_masks(om, h, w, t)
        p = unpack_masks(pm, h, w, t)
        pd = cv2.dilate(p.astype(np.uint8), ker) > 0
        occl = (pd & m).sum() / max(m.sum(), 1)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        cc = stats[1:, cv2.CC_STAT_AREA].max() / max(m.sum(), 1) if n > 1 else 0.0
        ys, xs = np.nonzero(m)
        border = (xs.min() <= 2) or (ys.min() <= 2) or (xs.max() >= w - 3) or (ys.max() >= h - 3)
        s = min(area[t] / amax, 1.2) * max(0.0, 1 - 2.0 * occl) * cc * (0.3 if border else 1.0)
        rows.append((s, int(t), float(occl), float(cc), bool(border)))
    rows.sort(key=lambda r: -r[0])
    chosen = []
    for r in rows:
        if all(abs(r[1] - c[1]) >= min_gap for c in chosen):
            chosen.append(r)
        if len(chosen) >= k:
            break
    return chosen, d


def full_res_mask(d, t, H, W):
    h, w = int(d["mask_h"]), int(d["mask_w"])
    m = unpack_masks(d["obj_masks"], h, w, t).astype(np.uint8)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0


def project(V, K):
    z = np.clip(V[:, 2], 1e-4, None)
    return np.stack([K[0, 0] * V[:, 0] / z + K[0, 2], K[1, 1] * V[:, 1] / z + K[1, 2]], 1)


def silhouette(Vc, faces, K, H, W):
    uv = project(Vc, K)
    ok = (Vc[faces, 2] > 1e-3).all(1)
    polys = np.round(uv[faces[ok]] * 4).astype(np.int32)  # sub-pixel (shift=2)
    sil = np.zeros((H, W), np.uint8)
    cv2.fillPoly(sil, list(polys), 1, lineType=cv2.LINE_8, shift=2)
    return sil > 0


def overlay(img, sil, mask, text):
    out = img.copy()
    out[sil] = (0.45 * out[sil] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnt, -1, (255, 0, 0), 2)
    cnt2, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnt2, -1, (255, 255, 0), 1)
    cv2.putText(out, text, (20, 50), 0, 1.3, (255, 255, 0), 3)
    return out


class Recon:
    def __init__(self, device="cuda"):
        sys.path.insert(0, os.path.join(SAM3DO_ROOT, "notebook"))
        from inference import Inference
        self.inf = Inference(os.path.join(SAM3DO_ROOT, "checkpoints/hf/pipeline.yaml"), compile=False)
        self.device = device

    def run(self, img, mask, pts_cv_np, seed=42):
        # pts_cv_np: metric MoGe-2 point map (OpenCV camera frame) computed by prepare_object_frames.py
        pts_cv = torch.nan_to_num(torch.from_numpy(pts_cv_np.astype(np.float32)), nan=0.0, posinf=0.0,
                                  neginf=0.0).to(self.device)
        pts_p3d = pts_cv * torch.tensor([-1.0, -1.0, 1.0], device=pts_cv.device)
        rgba = self.inf.merge_mask_to_rgba(img, mask)
        out = self.inf._pipeline.run(
            rgba, None, seed, stage1_only=False, with_mesh_postprocess=False,
            with_texture_baking=False, with_layout_postprocess=True, use_vertex_color=True,
            pointmap=pts_p3d)
        return out, pts_cv


def pose_from_output(out):
    from pytorch3d.transforms import quaternion_to_matrix
    q = out["rotation"].reshape(1, 4).float().cpu()
    Rq = quaternion_to_matrix(q)[0].double().numpy()
    t = out["translation"].reshape(-1)[:3].double().cpu().numpy()
    s = float(out["scale"].reshape(-1)[0])
    # row-vector pose in PyTorch3D camera: p = s * v_local @ Rq + t ; v_local = A v_glb (column)
    R = F_P3D @ Rq.T @ A_GLB
    return R, F_P3D @ t, s


def process(ep, rec, k, out_dir, vis_dir):
    t0 = time.time()
    meta = load_meta()[ep]
    cpath = os.path.join(WORK, "objects", "cache", f"episode_{ep:06d}.npz")
    if not os.path.exists(cpath):
        print(f"[ep {ep}] no prepared frames ({cpath})", flush=True)
        return
    c = np.load(cpath)
    chosen = json.loads(str(c["chosen"]))[:k]
    d = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
    K, kjson = load_intrinsics(ep)
    best = None
    for j, (sc, t, occl, cc, border) in enumerate(chosen):
        img = read_frame(VIDEO_TMPL.format(ep), t)
        H, W = img.shape[:2]
        mask = full_res_mask(d, t, H, W)
        try:
            out, pts = rec.run(img, mask, c["points"][j])
        except Exception as e:  # keep going on failures
            import traceback
            traceback.print_exc()
            print(f"[ep {ep}] frame {t} failed: {e}", flush=True)
            continue
        glb = out["glb"]
        V = np.asarray(glb.vertices, np.float64)
        Fc = np.asarray(glb.faces, np.int64)
        R, tt, s = pose_from_output(out)
        Vc = s * V @ R.T + tt
        sil = silhouette(Vc, Fc, K, H, W)
        iou = (sil & mask).sum() / max((sil | mask).sum(), 1)
        # depth agreement with the MoGe point map inside the mask
        zm = pts[..., 2].cpu().numpy()[mask]
        zm = float(np.median(zm[zm > 0])) if (zm > 0).any() else float("nan")
        print(f"[ep {ep}] frame {t} sel={sc:.2f} occl={occl:.2f} iou={iou:.3f} s={s:.3f} "
              f"z_mesh={np.median(Vc[:, 2]):.2f} z_moge={zm:.2f}", flush=True)
        cand = dict(frame=t, iou=float(iou), R=R, t=tt, s=s, V=V, F=Fc, glb=glb, img=img, mask=mask,
                    sil=sil, sel=sc, occl=occl, z_moge=zm)
        if best is None or cand["iou"] > best["iou"]:
            best = cand
    if best is None:
        return
    stem = os.path.join(out_dir, f"episode_{ep:06d}")
    best["glb"].export(stem + "_object.glb")
    import trimesh
    tm = trimesh.Trimesh(vertices=best["V"], faces=best["F"],
                         vertex_colors=getattr(best["glb"].visual, "vertex_colors", None), process=False)
    tm.export(stem + "_object.ply")
    ext = best["V"].max(0) - best["V"].min(0)
    pose = dict(
        episode=ep, object=meta["object"], frame=best["frame"],
        convention="point_cam = s * R @ point_mesh + t; camera frame OpenCV (x right, y down, z forward); metres; "
                   "point_mesh = vertices as stored in episode_XXXXXX_object.ply/.glb",
        R=best["R"].tolist(), t=best["t"].tolist(), s=best["s"],
        K=K.tolist(), width=int(best["img"].shape[1]), height=int(best["img"].shape[0]),
        intrinsics_source="work/intrinsics json (MoGe-2 global median)" if kjson else "default f=807.66",
        depth_model="MoGe-2 vitl-normal with fixed fov_x from K (metric)",
        iou=best["iou"], select_score=best["sel"], person_overlap=best["occl"], z_moge_median=best["z_moge"],
        object_extent_m=(best["s"] * ext).tolist(), num_vertices=int(len(best["V"])), num_faces=int(len(best["F"])),
    )
    with open(stem + "_pose.json", "w") as f:
        json.dump(pose, f, indent=1)
    ov = overlay(best["img"], best["sil"], best["mask"],
                 f"ep{ep} {meta['object']} f{best['frame']} IoU={best['iou']:.2f} ext={np.round(pose['object_extent_m'], 2)}")
    cv2.imwrite(os.path.join(vis_dir, f"episode_{ep:06d}_overlay.jpg"), ov[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"[ep {ep}] DONE frame={best['frame']} iou={best['iou']:.3f} extent={pose['object_extent_m']} "
          f"time={time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--shard", type=str, default=None)
    ap.add_argument("--k", type=int, default=3, help="candidate frames per episode")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    eps = a.episodes if a.episodes else list(range(30))
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        eps = eps[i::n]
    out_dir = os.path.join(WORK, "objects")
    vis_dir = os.path.join(out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    rec = Recon()
    for ep in eps:
        if a.skip_existing and os.path.exists(os.path.join(out_dir, f"episode_{ep:06d}_pose.json")):
            continue
        process(ep, rec, a.k, out_dir, vis_dir)


if __name__ == "__main__":
    main()
