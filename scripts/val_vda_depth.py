"""Temporally consistent metric depth with Video Depth Anything (metric model) in the compute_depth.py format.

Output: <out>/episode_XXXXXX.npz with depth [T, H/4, W/4] float16 metres, K (3x3 full resolution, from the episode
intrinsics json), downsample 4 -- the format val_depth_target.py / compute_moge_depth_target.py read.
The model (DepthAnything/Video-Depth-Anything, metric_depth checkpoints; Small: Apache-2.0, Base/Large:
CC-BY-NC-4.0) runs on the whole clip with its own temporal windows; frames are fed at --max-res on the long side.

  python val_vda_depth.py --videos ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera \
      --intr ROOT/work/intrinsics --out ROOT/work/depth_vda_vitl --episodes 0 7 --encoder vitl
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

VDA = "/work/doreen071/v2d/third_party_new/Video-Depth-Anything"
CKPT = "/work/doreen071/v2d/third_party_new/vda_checkpoints"
CFG = {"vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
       "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
       "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}}
DS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--intr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--encoder", default="vitl", choices=list(CFG))
    ap.add_argument("--max-res", type=int, default=1280)
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, VDA)
    from video_depth_anything.video_depth import VideoDepthAnything
    model = VideoDepthAnything(**CFG[a.encoder], metric=True)
    model.load_state_dict(torch.load(os.path.join(CKPT, f"metric_video_depth_anything_{a.encoder}.pth"),
                                     map_location="cpu"), strict=True)
    model = model.cuda().eval()
    os.makedirs(a.out, exist_ok=True)
    for ep in a.episodes:
        out = os.path.join(a.out, f"episode_{ep:06d}.npz")
        if a.skip_existing and os.path.exists(out):
            continue
        t0 = time.time()
        cap = cv2.VideoCapture(os.path.join(a.videos, f"episode_{ep:06d}.mp4"))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        s = min(1.0, a.max_res / max(W, H))
        w, h = int(round(W * s / 2) * 2), int(round(H * s / 2) * 2)
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA))
        frames = np.stack(frames)
        with torch.no_grad():
            depths, _ = model.infer_video_depth(frames, 30, input_size=a.input_size, device="cuda", fp32=False)
        D = np.stack([cv2.resize(np.asarray(d, np.float32), (W // DS, H // DS), interpolation=cv2.INTER_AREA)
                      for d in depths]).astype(np.float16)
        it = json.load(open(os.path.join(a.intr, f"episode_{ep:06d}.json")))
        K = np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1.0]])
        np.savez_compressed(out, depth=D, K=K, downsample=DS, model=f"metric_vda_{a.encoder}")
        print(f"ep {ep}: {len(D)} frames, median depth {np.median(D[D > 0]):.2f} m, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
