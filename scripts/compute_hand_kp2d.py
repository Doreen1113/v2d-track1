"""Independent 2D hand keypoints (RTMPose-m hand, COCO-WholeBody hand order, 21 per hand) on the scored frames.

Hand crops come from SAM 3D Body's own 2D hand keypoints in work/body (box around the 21 points, enlarged), so
the detector only needs a rough location. Runs on CPU with onnxruntime (separate venv: rtmlib, onnxruntime,
opencv-python-headless; no torch needed).

Output: <out>/episode_XXXXXX.npz with frames [N], right [N,21,3], left [N,21,3] (x, y, score; full-image pixels)
and the crop boxes.

  /work/doreen071/v2d/envs/rtm/bin/python compute_hand_kp2d.py --out WORK/hand_kp2d [--step 2]
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

ROOT = "/work/doreen071/v2d"
WORK = f"{ROOT}/work"
VIDEO = f"{ROOT}/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{{:06d}}.mp4"
HAND_MODEL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
              "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip")
RIGHT70, LEFT70 = list(range(21, 42)), list(range(42, 63))  # MHR70 hand points incl. wrist (41 / 62)


def hand_box(pts, W, H, margin=1.6, min_size=48):
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    s = max((hi - lo).max() * margin, min_size)
    b = np.r_[c - s / 2, c + s / 2]
    b[[0, 2]] = np.clip(b[[0, 2]], 0, W - 1)
    b[[1, 3]] = np.clip(b[[1, 3]], 0, H - 1)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{WORK}/hand_kp2d")
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--step", type=int, default=1, help="every n-th scored frame")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    import onnxruntime as ort
    _session = ort.InferenceSession

    def session(*args, **kw):  # fixed thread count (the default spawns one thread per core of a shared node)
        so = ort.SessionOptions()
        so.intra_op_num_threads = a.threads
        so.inter_op_num_threads = 1
        kw["sess_options"] = so
        return _session(*args, **kw)
    ort.InferenceSession = session
    from rtmlib import RTMPose
    model = RTMPose(onnx_model=HAND_MODEL, model_input_size=(256, 256), backend="onnxruntime", device="cpu")
    scored = {int(k): v for k, v in json.load(open(f"{WORK}/body_v2/scored_frames.json")).items()}
    os.makedirs(a.out, exist_ok=True)
    for ep in a.episodes:
        t0 = time.time()
        b = np.load(f"{WORK}/body/episode_{ep:06d}.npz")
        valid, kp = b["valid"].astype(bool), b["kp2d"]
        frames = [f for f in scored[ep][::a.step] if f < len(valid) and valid[f]]
        want = set(frames)
        cap = cv2.VideoCapture(VIDEO.format(ep))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = {"right": [], "left": [], "box_right": [], "box_left": []}
        done = []
        for t in range(max(frames) + 1 if frames else 0):
            if t not in want:
                cap.grab()
                continue
            ok, img = cap.read()
            if not ok:
                break
            for side, idx in (("right", RIGHT70), ("left", LEFT70)):
                box = hand_box(kp[t, idx], W, H)
                k, s = model(img, bboxes=[box.tolist()])
                out[side].append(np.concatenate([k[0], s[0][:, None]], 1))
                out[f"box_{side}"].append(box)
            done.append(t)
        cap.release()
        np.savez_compressed(f"{a.out}/episode_{ep:06d}.npz", frames=np.array(done),
                            **{k: np.asarray(v, np.float32) for k, v in out.items()})
        print(f"ep {ep}: {len(done)} frames, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
