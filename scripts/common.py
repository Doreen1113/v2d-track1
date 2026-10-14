"""Shared paths and helpers for the Track 1 pipeline."""
import json
import os

import numpy as np

ROOT = os.environ.get("V2D_ROOT", "/work/doreen071/v2d")
DATA = os.path.join(ROOT, "data/track_1")
WORK = os.path.join(ROOT, "work")
VIDEO_TMPL = os.path.join(DATA, "videos/chunk-000/observation.images.exo_camera/episode_{:06d}.mp4")
META = os.path.join(DATA, "meta/episodes_metadata.jsonl")


def load_meta():
    out = {}
    with open(META) as f:
        for line in f:
            d = json.loads(line)
            out[int(d["episode_index"])] = d
    return out


def load_intrinsics(ep, width=1536, height=1152):
    """Return 3x3 K (full-resolution pixels) and the raw json (or None)."""
    p = os.path.join(WORK, "intrinsics", f"episode_{ep:06d}.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        K = np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]], dtype=np.float64)
        return K, d
    f_ = 807.66
    K = np.array([[f_, 0, width / 2], [0, f_, height / 2], [0, 0, 1]], dtype=np.float64)
    return K, None


def read_video(path, scale=1.0):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if scale != 1.0:
            f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(np.ascontiguousarray(f[:, :, ::-1]))  # RGB
    cap.release()
    return frames


def read_frame(path, idx):
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {idx} of {path}")
    return f[:, :, ::-1].copy()


def pack_masks(masks):
    """masks: [T,H,W] bool -> packed uint8 [T, ceil(H*W/8)]."""
    T = masks.shape[0]
    return np.packbits(masks.reshape(T, -1), axis=1)


def unpack_masks(packed, h, w, idx=None):
    if idx is not None:
        return np.unpackbits(packed[idx])[: h * w].reshape(h, w).astype(bool)
    T = packed.shape[0]
    return np.unpackbits(packed, axis=1)[:, : h * w].reshape(T, h, w).astype(bool)


def load_masks(ep, key="obj"):
    """Load per-frame masks (half resolution) for an episode: returns [T,h,w] bool."""
    d = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
    h, w = int(d["mask_h"]), int(d["mask_w"])
    return unpack_masks(d[f"{key}_masks"], h, w)
