"""Main-actor person boxes for every frame of each Track 1 episode (ViTDet + MainPersonTracker).

Runs in the sam3db env (detectron2 ViTDet from the SAM 3D Body repo, local weights only; no gated models).
Output: <out>/episode_XXXXXX_bbox.npz with
  bbox_raw  [T,4]  xyxy of the tracked main actor (NaN where no box was kept)
  detected  [T]    bool
  n_det     [T]    number of person detections in the frame
  width, height, fps

Usage:
  python track_person_bbox.py --episodes 0 1 2      # or --all [--shard i/n]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sam3d_body import SAM3DB_ROOT, VIDEO_DIR, VITDET_DIR, MainPersonTracker  # noqa: E402

sys.path.insert(0, SAM3DB_ROOT)

OUT_DIR = "/work/doreen071/v2d/work/body_gvhmr/preproc"


def process(ep, detector, args):
    from tools.build_detector import run_detectron2_vitdet

    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, f"episode_{ep:06d}.mp4"))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    tracker = MainPersonTracker(W, H)
    boxes_out, ndet = [], []
    t0 = time.time()
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        with torch.no_grad():
            boxes = run_detectron2_vitdet(detector.detector, bgr, det_cat_id=0, bbox_thr=args.det_thr,
                                          nms_thr=0.3, default_to_full_image=False)
        box = tracker.update(list(boxes))
        ndet.append(len(boxes))
        boxes_out.append(np.full(4, np.nan) if box is None else np.asarray(box, np.float64))
    cap.release()
    bbox = np.stack(boxes_out).astype(np.float32)
    det = np.isfinite(bbox[:, 0])
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, f"episode_{ep:06d}_bbox.npz"), bbox_raw=bbox, detected=det,
                        n_det=np.array(ndet, np.int16), width=W, height=H, fps=fps)
    print(f"ep {ep}: T={len(bbox)} detected={det.sum()} multi-det={(np.array(ndet) > 1).sum()} "
          f"{time.time() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--det-thr", type=float, default=0.5)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    eps = list(range(30)) if args.all or not args.episodes else args.episodes
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        eps = eps[i::n]
    if args.skip_existing:
        eps = [e for e in eps if not os.path.exists(os.path.join(args.out, f"episode_{e:06d}_bbox.npz"))]
    print("episodes:", eps, flush=True)
    if not eps:
        return
    from tools.build_detector import HumanDetector
    detector = HumanDetector(name="vitdet", device="cuda", path=VITDET_DIR)
    for ep in eps:
        process(ep, detector, args)


if __name__ == "__main__":
    main()
