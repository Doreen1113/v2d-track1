"""Diagnostics for refine_body_sequence.py outputs.

Per episode writes to <v2>/vis/:
  episode_XXXXXX_overlay.jpg   v1 (red) and v2 (green) projected vertices + ViTPose keypoints (yellow)
                               on the first scored frame and 3 more frames
  episode_XXXXXX_traj.png      pelvis depth / height over time (raw SAM, v1, v2) with foot contacts

  python vis_body_v2.py --episodes 0 2 9
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhr_utils import MHRForward, project  # noqa: E402
from refine_body_sequence import BODY_DIR, VIT_DIR, fill_invalid  # noqa: E402

VIDEO_DIR = "/work/doreen071/v2d/data/track_1/videos/chunk-000/observation.images.exo_camera"


def read_frames(path, frames):
    cap = cv2.VideoCapture(path)
    out = {}
    for f in sorted(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            out[int(f)] = img
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="/work/doreen071/v2d/work/body_v2")
    ap.add_argument("--episodes", type=int, nargs="+", default=list(range(30)))
    ap.add_argument("--scored", default="/work/doreen071/v2d/work/body_v2/scored_frames.json")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = os.path.join(args.v2, "vis")
    os.makedirs(out_dir, exist_ok=True)
    scored = {int(k): v for k, v in json.load(open(args.scored)).items()}
    mhr = MHRForward()
    for ep in args.episodes:
        p2 = os.path.join(args.v2, f"episode_{ep:06d}.npz")
        if not os.path.exists(p2):
            continue
        v2 = np.load(p2)
        b = np.load(os.path.join(BODY_DIR, f"episode_{ep:06d}.npz"))
        valid = b["valid"].astype(bool)
        K = (float(b["fx"]), float(b["fy"]), float(b["cx"]), float(b["cy"]))
        T = len(v2["pose"])
        pose0 = fill_invalid(b["pose"], valid)
        pose0[:, 3:] = np.unwrap(pose0[:, 3:], axis=0)
        pose_v1 = gaussian_filter1d(pose0, 2.0, axis=0, mode="nearest")
        sh1, sc1 = np.median(b["shape"][valid], 0), np.median(b["scales"][valid], 0)
        kv = torch.load(f"{VIT_DIR}/episode_{ep:06d}_feat.pt", map_location="cpu", weights_only=False)["kp2d"].numpy()

        sc = scored.get(ep, [0])
        frames = [sc[0]] + list(np.linspace(sc[0], sc[-1], 5).round().astype(int)[1:4])
        imgs = read_frames(os.path.join(VIDEO_DIR, f"episode_{ep:06d}.mp4"), frames)
        V1, _ = mhr(pose_v1[frames], sc1, sh1)
        V2, _ = mhr(v2["pose"][frames], v2["scales"], v2["shape"])
        tiles = []
        for i, f in enumerate(frames):
            img = imgs[f].copy()
            for V, col in ((V1, (0, 0, 255)), (V2, (0, 255, 0))):
                uv = project(V[i], *K)
                for u, v in uv[::6].astype(int):
                    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        cv2.circle(img, (u, v), 1, col, -1)
            for u, v, c in kv[f]:
                if c > 0.3:
                    cv2.circle(img, (int(u), int(v)), 5, (0, 255, 255), 2)
            uv = project(V2[i], *K)
            x0, y0 = np.clip(uv.min(0) - 80, 0, None).astype(int)
            x1, y1 = np.minimum(uv.max(0) + 80, [img.shape[1], img.shape[0]]).astype(int)
            crop = img[y0:y1, x0:x1]
            crop = cv2.resize(crop, (int(crop.shape[1] * 600 / max(crop.shape[0], 1)), 600))
            cv2.putText(crop, f"ep{ep} f{f}" + (" first scored" if i == 0 else ""), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(crop)
        cv2.imwrite(os.path.join(out_dir, f"episode_{ep:06d}_overlay.jpg"), np.concatenate(tiles, 1))

        # trajectories
        _, J1 = mhr(pose_v1, sc1, sh1)
        _, J2 = mhr(v2["pose"], v2["scales"], v2["shape"])
        n = v2["floor_n"]
        fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        t = np.arange(T)
        ax[0].plot(t, J1[:, 1, 2], color="tab:red", lw=0.8, label="v1 pelvis z")
        ax[0].plot(t, J2[:, 1, 2], color="tab:green", lw=1.2, label="v2 pelvis z")
        ax[0].set_ylabel("depth (m)")
        ax[0].legend(loc="upper right")
        ax[1].plot(t, J1[:, 1] @ n - v2["floor_d"], color="tab:red", lw=0.8, label="v1")
        ax[1].plot(t, J2[:, 1] @ n - v2["floor_d"], color="tab:green", lw=1.2, label="v2")
        ax[1].set_ylabel("pelvis height (m)")
        c = v2["contact"].astype(float)
        ax[2].fill_between(t, 0, c[:, 0] * 1.0, step="mid", alpha=0.5, label="left contact")
        ax[2].fill_between(t, 0, -c[:, 1] * 1.0, step="mid", alpha=0.5, label="right contact")
        ax[2].legend(loc="upper right")
        for a in ax:
            a.axvspan(sc[0], sc[-1], color="0.9", zorder=0)
        ax[2].set_xlabel("frame (grey = scored span)")
        fig.suptitle(f"episode {ep}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"episode_{ep:06d}_traj.png"), dpi=80)
        plt.close(fig)
        print("vis", ep, flush=True)


if __name__ == "__main__":
    main()
