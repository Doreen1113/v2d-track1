"""Ground-truth-free proxy metrics for a body directory (episode_XXXXXX.npz with pose [T,136], scales, shape).

Per episode:
  reproj_px / reproj_px_scored   2D error vs ViTPose (conf > 0.5), all valid / scored frames
  iou / iou_p10                  silhouette IoU vs the person mask (half resolution, object pixels ignored),
                                 mean and 10th percentile over every 3rd valid frame
  moge_cv_pct                    per-frame ratio MoGe-depth / body-depth on visible person pixels, normalised by
                                 the frame's background MoGe scale; std/mean over scored frames (%)
  moge_drift_pct                 mean |ratio_t / ratio_first_scored - 1| over scored frames (%)
  dz_p99_cm, z_acc_cm            pelvis depth jitter
  skate_cm                       3D speed of feet that are stationary in the image (cm/frame)
  acc_cm / acc_root_xy / acc_root_z / acc_rel   22-joint acceleration magnitude on consecutive scored frames
                                 (cm/frame^2), total and split into root image-plane / root depth / root-relative
With --ref DIR also the change of the first scored frame w.r.t. that directory:
  first_mpjpe_cm (22 joints, camera frame), first_pa_v2v_cm (vertices after a similarity fit),
  first_depth_cm (pelvis depth change).

  python eval_body_proxies.py --body WORK/body_v3 --ref WORK/body_v2 --out WORK/body_v3/proxies
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, unpack_masks  # noqa: E402
from mhr_utils import DEFAULT_MHR, MHRForward  # noqa: E402

J22 = [1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42]
COCO_TO_70 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


def faces():
    m = torch.jit.load(DEFAULT_MHR, map_location="cpu")
    return dict(m.named_buffers())["character_torch.mesh.faces"].long()


def sim3(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, s, vt = np.linalg.svd(a.T @ b)
    c = np.ones(3)
    c[-1] = -1 if np.linalg.det(u @ vt) < 0 else 1
    R = (u * c) @ vt
    sc = (s * c).sum() / (a * a).sum()
    return sc * (src - ms) @ R + md


def acc(x):
    return x[2:] - 2 * x[1:-1] + x[:-2]


def kp70(verts, joints, kmap):
    vj = np.concatenate([verts, joints], 1)
    return np.einsum("kn,tnc->tkc", kmap, vj)


class Renderer:
    def __init__(self, K, h, w, device):
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        from pytorch3d.utils import cameras_from_opencv_projection
        cam = cameras_from_opencv_projection(
            R=torch.eye(3, device=device)[None], tvec=torch.zeros(1, 3, device=device),
            camera_matrix=torch.tensor(K, dtype=torch.float32, device=device)[None],
            image_size=torch.tensor([[h, w]], device=device))
        self.rast = MeshRasterizer(cameras=cam, raster_settings=RasterizationSettings(
            image_size=(h, w), blur_radius=0.0, faces_per_pixel=1, bin_size=0))
        self.dev = device

    def __call__(self, verts, F):
        from pytorch3d.structures import Meshes
        v = torch.as_tensor(verts, dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            frag = self.rast(Meshes(verts=v, faces=F[None].expand(len(v), -1, -1).to(self.dev)))
        return (frag.pix_to_face[..., 0] >= 0).cpu().numpy(), frag.zbuf[..., 0].cpu().numpy()


def stationary_feet_2d(kv, kpsam, valid):
    from scipy.ndimage import gaussian_filter1d
    C = np.zeros((len(kv), 2), bool)
    for s, (ia, sole) in enumerate(((15, [15, 16, 17]), (16, [18, 19, 20]))):
        a = gaussian_filter1d(kv[:, ia, :2], 1.0, axis=0, mode="nearest")
        b = gaussian_filter1d(kpsam[:, sole].mean(1), 1.0, axis=0, mode="nearest")
        va = np.r_[0, np.linalg.norm(np.diff(a, axis=0), axis=1)]
        vb = np.r_[0, np.linalg.norm(np.diff(b, axis=0), axis=1)]
        C[:, s] = (va < 1.2) & (vb < 1.2) & (kv[:, ia, 2] > 0.6) & valid
    return C


def evaluate(ep, body_dir, ref_dir, mhr, F, kmap, scored, device, stride=3):
    b = np.load(os.path.join(body_dir, f"episode_{ep:06d}.npz"))
    raw = np.load(os.path.join(WORK, "body", f"episode_{ep:06d}.npz"))
    valid = raw["valid"].astype(bool)
    K = np.array([float(raw["fx"]), float(raw["fy"]), float(raw["cx"]), float(raw["cy"])])
    pose = b["pose"]
    T = len(pose)
    V, J = mhr(pose, b["scales"], b["shape"])
    kp = kp70(V, J, kmap)
    vit = torch.load(os.path.join(WORK, "body_gvhmr/preproc", f"episode_{ep:06d}_feat.pt"), weights_only=False)
    kv = vit["kp2d"].numpy().astype(np.float64)
    det = vit["detected"].numpy().astype(bool)
    kv[~det | ~valid, :, 2] = 0
    sc = np.zeros(T, bool)
    sc[[f for f in scored if f < T]] = True
    out = {}
    uv = np.stack([K[0] * kp[..., 0] / kp[..., 2] + K[2], K[1] * kp[..., 1] / kp[..., 2] + K[3]], -1)
    e = np.linalg.norm(uv[:, COCO_TO_70] - kv[..., :2], axis=-1)
    m = kv[..., 2] > 0.5
    ho = b["holdout"].astype(bool) if "holdout" in b.files else np.zeros(T, bool)
    if ho.any():  # fitted on the other frames: report 2D error and IoU on the held-out frames only
        m &= ho[:, None]
    out["reproj_px"] = float(e[m].mean())
    out["reproj_px_scored"] = float(e[m & sc[:, None]].mean()) if (m & sc[:, None]).any() else float("nan")
    # depth jitter
    z = J[:, 1, 2]
    vz = valid[1:] & valid[:-1]
    out["dz_p99_cm"] = float(np.percentile(np.abs(np.diff(z))[vz] * 100, 99))
    vz2 = valid[2:] & valid[1:-1] & valid[:-2]
    out["z_acc_cm"] = float(np.abs(acc(z))[vz2].mean() * 100)
    # foot skating
    st = stationary_feet_2d(kv, raw["kp2d"].astype(np.float64), valid)
    sk = []
    for s, ids in enumerate(([13, 15, 16, 17], [14, 18, 19, 20])):
        f3 = kp[:, ids].mean(1)
        sp = np.linalg.norm(np.diff(f3, axis=0), axis=1) * 100
        sk.append(sp[st[1:, s] & st[:-1, s]])
    sk = np.concatenate(sk)
    out["skate_cm"] = float(sk.mean()) if len(sk) else float("nan")
    # accelerations on consecutive scored frames
    ok = sc[2:] & sc[1:-1] & sc[:-2]
    js = J[:, J22]
    out["acc_cm"] = float(np.linalg.norm(acc(js), axis=-1)[ok].mean() * 100)
    ar = acc(J[:, 1])
    out["acc_root_xy"] = float(np.linalg.norm(ar[:, :2], axis=-1)[ok].mean() * 100)
    out["acc_root_z"] = float(np.abs(ar[:, 2])[ok].mean() * 100)
    out["acc_rel"] = float(np.linalg.norm(acc(js - J[:, 1:2]), axis=-1)[ok].mean() * 100)
    # silhouettes and MoGe depth consistency
    mk = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
    mh, mw = int(mk["mask_h"]), int(mk["mask_w"])
    dd = np.load(os.path.join(WORK, "depth", f"episode_{ep:06d}.npz"))
    D = dd["depth"]
    ds = int(dd["downsample"])
    dh, dw = D.shape[1:]
    Kh = np.array([[K[0] / 2, 0, K[2] / 2], [0, K[1] / 2, K[3] / 2], [0, 0, 1]])
    rend = Renderer(Kh, mh, mw, device)
    frames = np.flatnonzero(valid)[::stride] if not ho.any() else np.flatnonzero(valid)
    first = scored[0] if scored else int(np.flatnonzero(valid)[0])
    frames = np.unique(np.r_[frames, np.flatnonzero(sc & valid)[::stride], first])
    frames = frames[frames < min(T, len(D))]
    if ho.any():
        frames = frames[ho[frames]]
    # static background reference depth
    bgs = []
    for f in np.linspace(0, len(D) - 1, 40).round().astype(int):
        pm = unpack_masks(mk["person_masks"], mh, mw, f) | unpack_masks(mk["obj_masks"], mh, mw, f)
        pm = binary_dilation(pm[::mh // dh, ::mw // dw][:dh, :dw], iterations=4)
        zz = D[f].astype(np.float32)
        bgs.append(np.where(pm | ~(zz > 0), np.nan, zz))
    Zbg = np.nanmedian(np.stack(bgs), 0)
    iou = {}
    ratio = {}
    for a0 in range(0, len(frames), 48):
        fr = frames[a0:a0 + 48]
        sil, zb = rend(V[fr], F)
        for k, f in enumerate(fr):
            P = unpack_masks(mk["person_masks"], mh, mw, f)
            O = unpack_masks(mk["obj_masks"], mh, mw, f)
            keep = ~(O & ~P)
            if P.sum() > 50:
                iou[int(f)] = float((sil[k] & P & keep).sum() / max(((sil[k] | P) & keep).sum(), 1))
            # depth ratio on eroded visible person pixels
            pe = cv2.erode(P.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            pe = (pe & sil[k])[::mh // dh, ::mw // dw][:dh, :dw]
            zr = zb[k][::mh // dh, ::mw // dw][:dh, :dw]
            Df = D[f].astype(np.float32)
            okp = pe & (Df > 0) & (zr > 0)
            bg = np.isfinite(Zbg) & (Df > 0)
            pm = unpack_masks(mk["person_masks"], mh, mw, f) | O
            bg &= ~binary_dilation(pm[::mh // dh, ::mw // dw][:dh, :dw], iterations=4)
            if okp.sum() > 30 and bg.sum() > 500:
                ratio[int(f)] = float(np.median(Df[okp] / zr[okp]) / np.median(Df[bg] / Zbg[bg]))
    iv = np.array(list(iou.values()))
    out["iou"] = float(iv.mean())
    out["iou_p10"] = float(np.percentile(iv, 10))
    rs = np.array([ratio[f] for f in sorted(ratio) if sc[f]])
    out["moge_cv_pct"] = float(rs.std() / rs.mean() * 100) if len(rs) > 2 else float("nan")
    if first in ratio:
        out["moge_drift_pct"] = float(np.mean(np.abs(rs / ratio[first] - 1)) * 100)
    out["iou_first"] = iou.get(first, float("nan"))
    if ref_dir:
        r = np.load(os.path.join(ref_dir, f"episode_{ep:06d}.npz"))
        Vr, Jr = mhr(r["pose"][[first]], r["scales"], r["shape"])
        out["first_mpjpe_cm"] = float(np.linalg.norm(Jr[0, J22] - J[first, J22], axis=-1).mean() * 100)
        out["first_pa_v2v_cm"] = float(np.linalg.norm(sim3(V[first], Vr[0]) - Vr[0], axis=-1).mean() * 100)
        out["first_depth_cm"] = float((J[first, 1, 2] - Jr[0, 1, 2]) * 100)
    series = dict(frames=sorted(ratio), ratio=[ratio[f] for f in sorted(ratio)], iou_frames=sorted(iou),
                  iou=[iou[f] for f in sorted(iou)], pelvis_z=z.tolist())
    return out, series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", required=True, help="directory for per-episode json")
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--scored", default=os.path.join(WORK, "body_v2/scored_frames.json"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHRForward(device=dev)
    F = faces()
    import scipy.sparse as sp
    kmap = sp.load_npz(os.path.join(WORK, "body_v2/kp70_mapping.npz")).toarray()
    scored = {int(k): v for k, v in json.load(open(a.scored)).items()}
    os.makedirs(a.out, exist_ok=True)
    for ep in a.episodes:
        out, s = evaluate(ep, a.body, a.ref, mhr, F, kmap, scored.get(ep, []), dev)
        print(ep, " ".join(f"{k}={v:.3f}" for k, v in out.items()), flush=True)
        with open(os.path.join(a.out, f"episode_{ep:06d}.json"), "w") as f:
            json.dump(dict(metrics=out, series=s), f)


if __name__ == "__main__":
    main()
