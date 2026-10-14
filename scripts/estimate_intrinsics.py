"""Estimate per-episode pinhole intrinsics with MoGe-2 on a handful of frames.

Writes work/intrinsics/episode_XXXXXX.json with fx, fy, cx, cy (pixels, principal point at the
image centre) and the per-frame estimates, plus a low-resolution metric point map of the
median frame (useful later for the ground plane).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel

ROOT = Path("/work/doreen071/v2d")
VIDEO = str(ROOT / "data/track_1/videos/chunk-000/observation.images.exo_camera") + "/episode_{:06d}.mp4"


def read_frames(path, count):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, n - 1, count).round().astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if ok:
            frames.append((int(i), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return frames, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "work/intrinsics")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
    for ep in args.episodes:
        frames, n = read_frames(VIDEO.format(ep), args.frames)
        fxs, fys, depths = [], [], []
        points = None
        for k, (i, img) in enumerate(frames):
            h, w = img.shape[:2]
            x = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1).cuda()
            with torch.no_grad():
                out = model.infer(x)
            K = out["intrinsics"].cpu().numpy()  # normalised
            fxs.append(float(K[0, 0] * w))
            fys.append(float(K[1, 1] * h))
            d = out["depth"].cpu().numpy()
            depths.append(float(np.nanmedian(d[np.isfinite(d)])))
            if k == len(frames) // 2:
                p = out["points"].cpu().numpy()
                points = cv2.resize(p, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST)
        fx, fy = float(np.median(fxs)), float(np.median(fys))
        f = float(np.sqrt(fx * fy))
        rec = {"episode": ep, "width": w, "height": h, "fx": f, "fy": f, "cx": w / 2, "cy": h / 2,
               "fx_frames": fxs, "fy_frames": fys, "median_depth_frames": depths,
               "frame_indices": [i for i, _ in frames], "num_frames": n}
        (args.out / f"episode_{ep:06d}.json").write_text(json.dumps(rec, indent=1))
        np.save(args.out / f"episode_{ep:06d}_points_lowres.npy", points.astype(np.float16))
        print(ep, f"f={f:.1f} spread={np.std(fxs):.1f} depth={np.median(depths):.2f}m", flush=True)


if __name__ == "__main__":
    main()
