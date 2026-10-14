"""Select candidate frames for object reconstruction and compute metric point maps.

Runs in the segmentation env (needs MoGe-2). For each episode it ranks frames by the
tracked object mask (large, single blob, not touching the image border, little overlap
with the actor) and runs MoGe-2 with the fixed episode intrinsics (fov_x from fx) to get a
metric point map in the OpenCV camera frame for each chosen frame.

Output: WORK/objects/cache/episode_XXXXXX.npz with
  chosen  json list of [select_score, frame, person_overlap, largest_cc_frac, touches_border]
  points  [k, H, W, 3] float16 metric point map (OpenCV camera frame)
  K       3x3 intrinsics used

Usage: python prepare_object_frames.py --episodes 0 1  (or --shard i/n) [--k 3]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import VIDEO_TMPL, WORK, load_intrinsics, read_frame  # noqa: E402
from reconstruct_object import select_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--shard", type=str, default=None)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    eps = a.episodes if a.episodes else list(range(30))
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        eps = eps[i::n]
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
    out_dir = os.path.join(WORK, "objects", "cache")
    os.makedirs(out_dir, exist_ok=True)
    for ep in eps:
        chosen, _ = select_frames(ep, k=a.k)
        K, _ = load_intrinsics(ep)
        pts = []
        for row in chosen:
            img = read_frame(VIDEO_TMPL.format(ep), row[1])
            H, W = img.shape[:2]
            fov_x = math.degrees(2 * math.atan(W / (2 * K[0, 0])))
            x = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1).cuda()
            with torch.no_grad():
                p = model.infer(x, fov_x=fov_x)["points"].float().cpu().numpy()
            pts.append(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16))
        np.savez(os.path.join(out_dir, f"episode_{ep:06d}.npz"), chosen=json.dumps(chosen),
                 points=np.stack(pts) if pts else np.zeros((0, 1, 1, 3), np.float16), K=K)
        print(ep, [(r[1], round(r[0], 2), round(r[2], 2)) for r in chosen], flush=True)


if __name__ == "__main__":
    main()
