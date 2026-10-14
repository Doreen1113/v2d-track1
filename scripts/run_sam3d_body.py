"""Per-frame SAM 3D Body on Track 1 episode videos -> MHR params in the camera frame.

Output per episode: <out>/episode_XXXXXX.npz with
  pose   [T,136]  MHR model params 0:136; root translation set so that the scorer-convention forward
                  (verts = diag(1,-1,-1) @ MHR(...) / 100) lands the body in the OpenCV camera frame (metres)
  scales [T,68]   MHR model params 136:204 (per frame, raw)
  shape  [T,45]   MHR identity coefficients (per frame, raw)
  cam_t  [T,3]    camera-frame translation added to SAM 3D Body's root-centred vertices (metres)
  valid  [T]      person found in this frame (invalid rows are zero)
  bbox   [T,4]    xyxy of the tracked person
  kp2d   [T,70,2] SAM 3D Body 2D keypoints (pixels)
  fx, fy, cx, cy, focal_source, width, height, fps
Plus raw head outputs (global_rot, body_pose, hand, scale_pca) for later refinement.

Usage:
  python run_sam3d_body.py --episodes 0 1 2         # or --all, optionally --shard i/n
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

SAM3DB_ROOT = "/work/doreen071/v2d/third_party/sam-3d-body"
sys.path.insert(0, SAM3DB_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mhr_utils import cam_t_to_root_trans  # noqa: E402

VIDEO_DIR = "/work/doreen071/v2d/data/track_1/videos/chunk-000/observation.images.exo_camera"
INTR_DIR = "/work/doreen071/v2d/work/intrinsics"
OUT_DIR = "/work/doreen071/v2d/work/body"
VITDET_DIR = "/work/doreen071/v2d/assets/vitdet"
HF_REPO = "facebook/sam-3d-body-dinov3"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


class MainPersonTracker:
    """Keeps the main actor: initialised on the largest, most central box; then IoU / proximity tracking."""

    def __init__(self, width, height, max_gap=60):
        self.w, self.h, self.max_gap = width, height, max_gap
        self.last, self.gap = None, 0

    def _init_score(self, b):
        c = np.array([(b[0] + b[2]) / 2 / self.w - 0.5, (b[1] + b[3]) / 2 / self.h - 0.5])
        return area(b) * (1.0 - np.linalg.norm(c))

    def update(self, boxes):
        if len(boxes) == 0:
            self.gap += 1
            return None
        if self.last is None or self.gap > self.max_gap:
            pick = max(boxes, key=self._init_score)
        else:
            ious = [iou(b, self.last) for b in boxes]
            k = int(np.argmax(ious))
            if ious[k] > 0.2:
                pick = boxes[k]
            else:
                lc = np.array([(self.last[0] + self.last[2]) / 2, (self.last[1] + self.last[3]) / 2])
                diag = np.sqrt(area(self.last))
                best, best_d = None, None
                for b in boxes:
                    r = area(b) / max(area(self.last), 1e-6)
                    d = np.linalg.norm(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) - lc) / diag
                    if 0.4 < r < 2.5 and d < 1.5 and (best_d is None or d < best_d):
                        best, best_d = b, d
                if best is None:
                    self.gap += 1
                    return None
                pick = best
        self.last, self.gap = np.asarray(pick, dtype=np.float64), 0
        return self.last


def load_intrinsics(ep, width, height, intr_dir):
    path = os.path.join(intr_dir, f"episode_{ep:06d}.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]), f"json:{path}"
    f = float(np.hypot(width, height))  # SAM 3D Body default (image diagonal)
    return f, f, width / 2.0, height / 2.0, "default_diag"


def build(device):
    from huggingface_hub import snapshot_download
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from tools.build_detector import HumanDetector

    local = snapshot_download(repo_id=HF_REPO)
    model, cfg = load_sam_3d_body(os.path.join(local, "model.ckpt"), device=device,
                                  mhr_path=os.path.join(local, "assets", "mhr_model.pt"))
    vit_path = VITDET_DIR if os.path.exists(os.path.join(VITDET_DIR, "model_final_f05665.pkl")) else ""
    detector = HumanDetector(name="vitdet", device=device, path=vit_path)
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg, human_detector=None,
                             human_segmentor=None, fov_estimator=None)
    use_intrin_center = bool(cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False))
    return est, detector, use_intrin_center


def process_episode(ep, est, detector, use_intrin_center, args):
    from tools.build_detector import run_detectron2_vitdet

    vpath = os.path.join(VIDEO_DIR, f"episode_{ep:06d}.mp4")
    cap = cv2.VideoCapture(vpath)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fx, fy, cx, cy, fsrc = load_intrinsics(ep, W, H, args.intr_dir)
    K = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32)
    tracker = MainPersonTracker(W, H)
    vis_frames = set()
    rec = {k: [] for k in ("mhr_params", "shape", "cam_t", "bbox", "kp2d", "global_rot", "body_pose",
                           "hand", "scale_pca", "ndet")}
    valid, sam_verts = [], {}
    t0 = time.time()
    t = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if args.max_frames and t >= args.max_frames:
            break
        with torch.no_grad():
            boxes = run_detectron2_vitdet(detector.detector, bgr, det_cat_id=0, bbox_thr=args.det_thr,
                                          nms_thr=0.3, default_to_full_image=False)
        box = tracker.update(list(boxes))
        rec["ndet"].append(len(boxes))
        out = None
        if box is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with contextlib.redirect_stdout(io.StringIO()):
                res = est.process_one_image(rgb, bboxes=box[None].astype(np.float32), cam_int=K.clone())
            out = res[0] if len(res) else None
        if out is None:
            valid.append(False)
            for k, shp in (("mhr_params", 204), ("shape", 45), ("cam_t", 3), ("bbox", 4), ("global_rot", 3),
                           ("body_pose", 133), ("hand", 108), ("scale_pca", 28)):
                rec[k].append(np.zeros(shp, np.float32))
            rec["kp2d"].append(np.zeros((70, 2), np.float32))
        else:
            cam_t = out["pred_cam_t"].astype(np.float64).copy()
            if not use_intrin_center:
                # The camera head places the body assuming the principal point at the image centre;
                # shift it so it reprojects to the same pixels under the real (cx, cy).
                z = cam_t[2] + float(np.mean(out["pred_vertices"][:, 2]))
                cam_t[0] += (W / 2.0 - cx) * z / fx
                cam_t[1] += (H / 2.0 - cy) * z / fy
            valid.append(True)
            rec["mhr_params"].append(out["mhr_model_params"].astype(np.float32))
            rec["shape"].append(out["shape_params"].astype(np.float32))
            rec["cam_t"].append(cam_t.astype(np.float32))
            rec["bbox"].append(np.asarray(box, np.float32))
            rec["kp2d"].append(out["pred_keypoints_2d"].astype(np.float32))
            rec["global_rot"].append(out["global_rot"].astype(np.float32))
            rec["body_pose"].append(out["body_pose_params"].astype(np.float32))
            rec["hand"].append(out["hand_pose_params"].astype(np.float32))
            rec["scale_pca"].append(out["scale_params"].astype(np.float32))
            if len(sam_verts) < args.n_vis and t % max(1, n // args.n_vis) == 0:
                sam_verts[t] = (out["pred_vertices"] + cam_t[None]).astype(np.float32)
        t += 1
        if t % 100 == 0:
            print(f"ep {ep}: {t}/{n} frames, {(time.time() - t0) / t:.3f} s/frame", flush=True)
    cap.release()
    T = t
    valid = np.array(valid, bool)
    R = {k: np.stack(v) for k, v in rec.items()}
    pose = R["mhr_params"][:, :136].copy()
    assert np.abs(pose[valid, :3]).max(initial=0) < 1e-5, "SAM 3D Body root translation expected to be zero"
    pose[:, :3] = cam_t_to_root_trans(R["cam_t"]).astype(np.float32)
    pose[~valid, :3] = 0
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out, f"episode_{ep:06d}.npz"),
        pose=pose, scales=R["mhr_params"][:, 136:], shape=R["shape"], cam_t=R["cam_t"], valid=valid,
        bbox=R["bbox"], kp2d=R["kp2d"], global_rot=R["global_rot"], body_pose=R["body_pose"], hand=R["hand"],
        scale_pca=R["scale_pca"], n_detections=R["ndet"].astype(np.int16),
        fx=fx, fy=fy, cx=cx, cy=cy, focal=fx, focal_source=fsrc, width=W, height=H, fps=fps,
        use_intrin_center=use_intrin_center)
    if sam_verts:
        os.makedirs(os.path.join(args.out, "vis"), exist_ok=True)
        np.savez_compressed(os.path.join(args.out, "vis", f"episode_{ep:06d}_samverts.npz"),
                            frames=np.array(sorted(sam_verts)), verts=np.stack([sam_verts[k] for k in sorted(sam_verts)]))
    dt = time.time() - t0
    print(f"ep {ep}: done T={T} valid={valid.sum()} multi-det frames={(R['ndet'] > 1).sum()} "
          f"focal={fx:.1f} ({fsrc}) time={dt:.1f}s ({dt / max(T, 1):.3f} s/frame)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n: process every n-th episode starting at i")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--intr-dir", default=INTR_DIR)
    ap.add_argument("--det-thr", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--n-vis", type=int, default=4, help="frames per episode whose SAM vertices are kept for checks")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    eps = list(range(30)) if args.all or not args.episodes else args.episodes
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        eps = eps[i::n]
    if args.skip_existing:
        eps = [e for e in eps if not os.path.exists(os.path.join(args.out, f"episode_{e:06d}.npz"))]
    print("episodes:", eps, flush=True)
    if not eps:
        return
    est, det, uic = build("cuda")
    print("USE_INTRIN_CENTER =", uic, flush=True)
    for ep in eps:
        process_episode(ep, est, det, uic, args)


if __name__ == "__main__":
    main()
