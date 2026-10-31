"""NLF (Neural Localizer Fields, Sarandi & Pons-Moll, NeurIPS 2024) on validation episodes: per-frame SMPL in the camera
frame, using the same person boxes (SAM 3D Body tracker, root/work/body_moge) and the same MoGe intrinsics as the
pipeline, so the absolute placement is comparable.

Model: nlf_l_multi_0.3.2.patch6.torchscript (github.com/isarandi/nlf releases; code MIT, weights non-commercial
research use). Needs torch >= 2.4 (run with the sam3db env).

Writes <out>/episode_XXXXXX.npz with, per frame (metres, camera frame, x right / y down / z forward):
  verts [T,6890,3]  parametric SMPL (neutral) vertices    joints [T,24,3]  SMPL joints
  pose [T,72], betas [T,10], trans [T,3]                    verts_np [T,6890,3] non-parametric vertices
  vert_unc [T,6890]  per-vertex uncertainty               valid [T]
  python val_nlf_infer.py --episodes 0 7 --out /work/doreen071/v2d_val/nlf/raw
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torchvision  # noqa: F401  (registers ops needed by the TorchScript model)

ROOT = "/work/doreen071/v2d_val/root"
MODEL = "/work/doreen071/v2d/third_party_new/nlf_models/nlf_l_multi_0.3.2.patch6.torchscript"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-dir", default=f"{ROOT}/data/track_1/videos/chunk-000/observation.images.exo_camera")
    ap.add_argument("--intr-dir", default=f"{ROOT}/work/intrinsics")
    ap.add_argument("--boxes-from", default=f"{ROOT}/work/body_moge")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num-aug", type=int, default=1)
    ap.add_argument("--box-scale", type=float, default=1.0)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    model = torch.jit.load(a.model).to(dev).eval()
    for ep in a.episodes:
        out = os.path.join(a.out, f"episode_{ep:06d}.npz")
        if a.skip_existing and os.path.exists(out):
            print(f"ep {ep}: exists", flush=True)
            continue
        t0 = time.time()
        intr = json.load(open(os.path.join(a.intr_dir, f"episode_{ep:06d}.json")))
        K = torch.tensor([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]],
                         dtype=torch.float32, device=dev)[None]
        bz = np.load(os.path.join(a.boxes_from, f"episode_{ep:06d}.npz"))
        bb = np.asarray(bz["bbox"], np.float64)  # xyxy
        bvalid = np.asarray(bz["valid"], bool) & np.isfinite(bb).all(1)
        cap = cv2.VideoCapture(os.path.join(a.video_dir, f"episode_{ep:06d}.mp4"))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        T = len(frames)
        idx = np.flatnonzero(bvalid[:T])
        # hold the nearest valid box on frames without one (marked invalid in the output)
        near = idx[np.abs(np.arange(T)[:, None] - idx[None]).argmin(1)]
        bb = bb[near]
        c = (bb[:, :2] + bb[:, 2:]) / 2
        wh = (bb[:, 2:] - bb[:, :2]) * a.box_scale
        xywh = np.concatenate([c - wh / 2, wh], 1).astype(np.float32)
        rec = {k: [] for k in ("verts", "joints", "pose", "betas", "trans", "verts_np", "vert_unc")}
        with torch.inference_mode(), torch.device(dev):
            for s in range(0, T, a.batch):
                e = min(T, s + a.batch)
                img = torch.from_numpy(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[s:e]])).to(dev)
                img = img.permute(0, 3, 1, 2).contiguous()
                boxes = [torch.from_numpy(xywh[t:t + 1]).to(dev) for t in range(s, e)]
                p = model.estimate_smpl_batched(img, boxes, intrinsic_matrix=K, num_aug=a.num_aug,
                                                model_name="smpl")
                for k, src in (("verts", "vertices3d"), ("joints", "joints3d"), ("pose", "pose"), ("betas", "betas"),
                               ("trans", "trans"), ("verts_np", "vertices3d_nonparam"),
                               ("vert_unc", "vertex_uncertainties")):
                    rec[k].append(torch.stack([x[0] for x in p[src]]).float().cpu().numpy())
        R = {k: np.concatenate(v) for k, v in rec.items()}
        # NLF returns millimetres for the point outputs; trans is in metres (SMPL)
        unit = 1000.0 if np.nanmedian(R["verts"][..., 2]) > 100 else 1.0
        for k in ("verts", "joints", "verts_np"):
            R[k] = (R[k] / unit).astype(np.float32)
        np.savez(out, **R, valid=bvalid[:T], fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"],
                 model=os.path.basename(a.model), num_aug=a.num_aug, unit_div=unit)
        print(f"ep {ep}: {T} frames, median pelvis z {np.median(R['joints'][:, 0, 2]):.3f} m, "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
