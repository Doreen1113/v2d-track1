"""Combine per-episode human and object estimates into the Track 1 episode NPZ + mesh.

Inputs (all in one OpenCV camera frame, metres):
  work/body/episode_XXXXXX.npz      per-frame MHR pose [T,136], scales [T,68], shape [T,45], valid [T]
  work/objects/episode_XXXXXX_object.{glb,ply,obj}
  work/objects/episode_XXXXXX_pose.json          single-frame pose (s, R, t)   -- fallback
  work/objects/episode_XXXXXX_track.npz          optional per-frame R [T,3,3], t [T,3], valid [T]

Output: OUT/episode_XXXXXX.npz and OUT/episode_XXXXXX_object.<ext>
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path("/work/doreen071/v2d")


def fill_invalid(x, valid):
    """Linear interpolation over invalid frames along axis 0 (hold at the ends)."""
    x = np.array(x, dtype=float)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        raise ValueError("no valid frames")
    flat = x.reshape(len(x), -1)
    t = np.arange(len(x))
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(t, idx, flat[idx, j])
    return flat.reshape(x.shape)


def smooth(x, sigma):
    if sigma <= 0:
        return x
    return gaussian_filter1d(x, sigma, axis=0, mode="nearest")


def fill_rotations(R, valid):
    idx = np.flatnonzero(valid)
    rots = Rotation.from_matrix(R[idx])
    if len(idx) == 1:
        return np.repeat(R[idx], len(R), axis=0)
    t = np.clip(np.arange(len(R)), idx[0], idx[-1])
    return Slerp(idx, rots)(t).as_matrix()


def smooth_rotations(R, sigma):
    """Gaussian smoothing of rotations via sign-consistent quaternion averaging."""
    if sigma <= 0:
        return R
    q = Rotation.from_matrix(R).as_quat()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    q = gaussian_filter1d(q, sigma, axis=0, mode="nearest")
    return Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()


def unwrap_euler(pose, euler_cols):
    pose = pose.copy()
    pose[:, euler_cols] = np.unwrap(pose[:, euler_cols], axis=0)
    return pose


def assemble(ep, body_dir, obj_dir, out_dir, sigma_h, sigma_o, euler_cols, k=1.0):
    body = dict(np.load(body_dir / f"episode_{ep:06d}.npz"))
    valid = body.get("valid", np.ones(len(body["pose"]), bool)).astype(bool)
    pose = fill_invalid(body["pose"], valid)
    pose = smooth(unwrap_euler(pose, euler_cols), sigma_h)
    scales = np.median(np.asarray(body["scales"])[valid], axis=0) if np.ndim(body["scales"]) == 2 else body["scales"]
    shape = np.median(np.asarray(body["shape"])[valid], axis=0) if np.ndim(body["shape"]) == 2 else body["shape"]
    T = len(pose)

    pose_json = json.loads((obj_dir / f"episode_{ep:06d}_pose.json").read_text())
    s = float(pose_json.get("s", pose_json.get("scale")))
    track = obj_dir / f"episode_{ep:06d}_track.npz"
    if track.exists():
        tr = dict(np.load(track))
        ov = tr["valid"].astype(bool)
        R = smooth_rotations(fill_rotations(tr["R"], ov), sigma_o)
        t = smooth(fill_invalid(tr["t"], ov), sigma_o)
        s = float(tr.get("scale", s))
    else:
        R = np.repeat(np.asarray(pose_json["R"], float)[None], T, axis=0)
        t = np.repeat(np.asarray(pose_json["t"], float)[None], T, axis=0)
    R = R[:T]
    t = t[:T] * k  # object estimates live in the MoGe depth frame; k maps them to the body frame
    s = s * k

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"episode_{ep:06d}.npz", pose=pose, scales=np.asarray(scales, float),
             shape=np.asarray(shape, float), object_rotation=R, object_translation=t,
             object_scale=np.float64(s))
    mesh = next(p for ext in (".ply", ".obj", ".glb", ".off", ".stl")
                if (p := obj_dir / f"episode_{ep:06d}_object{ext}").exists())
    shutil.copy(mesh, out_dir / mesh.name)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--body", type=Path, default=ROOT / "work/body")
    ap.add_argument("--objects", type=Path, default=ROOT / "work/objects")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sigma-h", type=float, default=2.0, help="Gaussian sigma (frames) on MHR pose")
    ap.add_argument("--sigma-o", type=float, default=3.0, help="Gaussian sigma (frames) on object pose")
    ap.add_argument("--euler-cols", type=str, default="3:136",
                    help="columns of pose holding angles, unwrapped before smoothing")
    ap.add_argument("--ratio", type=Path, default=ROOT / "work/moge_to_body_ratio.json",
                    help="per-episode MoGe/body depth ratio; object t and s are divided by it")
    args = ap.parse_args()
    ratio = json.loads(args.ratio.read_text())["ratio"] if args.ratio else {}
    a, b = (int(v) for v in args.euler_cols.split(":"))
    for ep in args.episodes:
        T = assemble(ep, args.body, args.objects, args.out, args.sigma_h, args.sigma_o, list(range(a, b)),
                     1.0 / ratio.get(str(ep), 1.0))
        print(f"episode {ep}: {T} frames")


if __name__ == "__main__":
    main()
