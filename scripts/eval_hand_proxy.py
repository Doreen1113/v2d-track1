"""Hand proxy: 2D error of the projected MHR hand keypoints against independent RTMPose hand keypoints
(compute_hand_kp2d.py), on the scored frames that have detections.

MHR70 hand points (SAM 3D Body order: tip, first, second, third joint per finger, then the wrist) are mapped to the
COCO-WholeBody hand order (wrist, then CMC/MCP .. tip per finger). Reported per run (mean over episodes):
  finger_px    mean pixel error over confident finger keypoints (score > --thr, default 0.25: RTMPose SimCC scores are low, median ~0.27), both hands
  wrist_px     same for the wrists
  finger_rel   finger error divided by the detected hand size (wrist -> middle-finger MCP), in %

  python eval_hand_proxy.py WORK/body_v2 WORK/body_v5 --sigma-h 3            # refined runs, assembly smoothing
  python eval_hand_proxy.py --assembled SUB/V2/episodes SUB/S1/episodes        # assembled episode npz as is
  python eval_hand_proxy.py --raw WORK/body                                    # per-frame SAM 3D Body (own identity)
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_body_proxies import assembled_pose  # noqa: E402
from mhr_utils import MHRForward  # noqa: E402

WORK = "/work/doreen071/v2d/work"
R70 = [41, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40, 39, 38, 37]
L70 = [62, 45, 44, 43, 42, 49, 48, 47, 46, 53, 52, 51, 50, 57, 56, 55, 54, 61, 60, 59, 58]


def project(kp, K):
    return np.stack([K[0] * kp[..., 0] / kp[..., 2] + K[2], K[1] * kp[..., 1] / kp[..., 2] + K[3]], -1)


def run(d, ep, mode, sigma_h, mhr, kmap, frames, sigma_f=None, sigma_a=None, raw_dir=None):
    b = np.load(os.path.join(d, f"episode_{ep:06d}.npz"))
    if mode == "raw":
        uv = b["kp2d"][frames].astype(np.float64)
        return uv
    pose = b["pose"] if mode == "assembled" else assembled_pose(b, sigma_h, sigma_f, sigma_a)
    scl, shp = b["scales"], b["shape"]
    if np.ndim(scl) == 2:
        scl, shp = scl[frames], shp[frames]
    V, J = mhr(pose[frames], scl, shp)
    kp = np.einsum("kn,tnc->tkc", kmap, np.concatenate([V, J], 1))
    raw = np.load(os.path.join(raw_dir or os.path.join(WORK, "body"), f"episode_{ep:06d}.npz"))  # intrinsics
    K = np.array([float(raw["fx"]), float(raw["fy"]), float(raw["cx"]), float(raw["cy"])])
    return project(kp, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--assembled", action="store_true", help="dirs hold assembled episodes (pose used as is)")
    ap.add_argument("--raw", action="store_true", help="dirs hold per-frame SAM 3D Body output (its own 2D points)")
    ap.add_argument("--sigma-h", type=float, default=0.0)
    ap.add_argument("--sigma-fingers", type=float, default=None)
    ap.add_argument("--sigma-arms", type=float, default=None)
    ap.add_argument("--raw-dir", default=None, help="per-frame run with the intrinsics of the dirs (default work/body)")
    ap.add_argument("--hands", default=f"{WORK}/hand_kp2d")
    ap.add_argument("--thr", type=float, default=0.25)  # SimCC scores: median ~0.27
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mode = "raw" if a.raw else ("assembled" if a.assembled else "refined")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHRForward(device=dev)
    kmap = sp.load_npz(os.path.join(WORK, "body_v2/kp70_mapping.npz")).toarray()
    res = {d: {} for d in a.dirs}
    for ep in a.episodes:
        p = os.path.join(a.hands, f"episode_{ep:06d}.npz")
        if not os.path.exists(p):
            continue
        h = np.load(p)
        fr = h["frames"]
        if len(fr) == 0:
            continue
        det = np.stack([h["right"], h["left"]], 1)  # [N,2,21,3]
        conf = det[..., 2] > a.thr
        size = np.linalg.norm(det[:, :, 9, :2] - det[:, :, 0, :2], axis=-1)  # wrist -> middle MCP
        for d in a.dirs:
            uv = run(d, ep, mode, a.sigma_h, mhr, kmap, fr, a.sigma_fingers, a.sigma_arms, a.raw_dir)
            pred = np.stack([uv[:, R70], uv[:, L70]], 1)  # [N,2,21,2]
            e = np.linalg.norm(pred - det[..., :2], axis=-1)
            fing, wr = conf.copy(), conf.copy()
            fing[..., 0] = False
            wr[..., 1:] = False
            rel = e / np.maximum(size[..., None], 5.0) * 100
            res[d][ep] = dict(finger_px=float(e[fing].mean()), wrist_px=float(e[wr].mean()),
                              finger_rel=float(rel[fing].mean()), n=int(fing.sum()))
        print(ep, " | ".join(f"{res[d][ep]['finger_px']:.2f}/{res[d][ep]['wrist_px']:.2f}" for d in a.dirs), flush=True)
    eps = sorted(set.intersection(*[set(r) for r in res.values()]))
    print(f"{len(eps)} episodes   (finger_px / wrist_px / finger_rel %)")
    for d in a.dirs:
        m = {k: np.mean([res[d][e][k] for e in eps]) for k in ("finger_px", "wrist_px", "finger_rel")}
        print(f"{d[-45:]:>45s}  {m['finger_px']:.3f}  {m['wrist_px']:.3f}  {m['finger_rel']:.2f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
