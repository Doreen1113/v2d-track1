"""Check an episode's body npz: official-MHR forward of (pose, scales, shape) vs SAM 3D Body's own
camera-frame vertices, and overlay the projected mesh on the video frames.

  python verify_body.py --episode 0 [--frames 0 100 200]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhr_utils import MHRForward, project  # noqa: E402

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
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--body-dir", default="/work/doreen071/v2d/work/body")
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    args = ap.parse_args()

    d = np.load(os.path.join(args.body_dir, f"episode_{args.episode:06d}.npz"))
    fx, fy, cx, cy = (float(d[k]) for k in ("fx", "fy", "cx", "cy"))
    mhr = MHRForward()
    sv_path = os.path.join(args.body_dir, "vis", f"episode_{args.episode:06d}_samverts.npz")
    frames = args.frames
    if os.path.exists(sv_path):
        sv = np.load(sv_path)
        verts, _ = mhr(d["pose"][sv["frames"]], d["scales"][sv["frames"]], d["shape"][sv["frames"]])
        err = np.linalg.norm(verts - sv["verts"], axis=-1)
        print(f"official MHR vs SAM 3D Body camera-frame verts, frames {sv['frames'].tolist()}: "
              f"mean {err.mean() * 1000:.4f} mm, max {err.max() * 1000:.4f} mm")
        if frames is None:
            frames = sv["frames"].tolist()
    if frames is None:
        frames = np.linspace(0, len(d["pose"]) - 1, 4).astype(int).tolist()
    frames = [f for f in frames if d["valid"][f]]
    imgs = read_frames(os.path.join(VIDEO_DIR, f"episode_{args.episode:06d}.mp4"), frames)
    verts, joints = mhr(d["pose"][frames], d["scales"][frames], d["shape"][frames])
    os.makedirs(os.path.join(args.body_dir, "vis"), exist_ok=True)
    for i, f in enumerate(frames):
        img = imgs[f].copy()
        uv = project(verts[i], fx, fy, cx, cy)
        over = img.copy()
        for u, v in uv[::3].astype(int):
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(over, (u, v), 1, (0, 255, 0), -1)
        img = cv2.addWeighted(over, 0.6, img, 0.4, 0)
        for u, v in d["kp2d"][f][:21].astype(int):
            cv2.circle(img, (u, v), 4, (0, 0, 255), -1)
        x1, y1, x2, y2 = d["bbox"][f].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"ep{args.episode} f{f} depth {d['cam_t'][f][2]:.2f}m focal {fx:.0f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out = os.path.join(args.body_dir, "vis", f"episode_{args.episode:06d}_f{f:04d}.png")
        cv2.imwrite(out, img)
        print("wrote", out)


if __name__ == "__main__":
    main()
