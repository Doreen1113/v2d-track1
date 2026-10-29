"""Independent whole-body 2D keypoints (RTMW-x, COCO-WholeBody 133 points: body 0:17, feet 17:23, face 23:91,
left hand 91:112, right hand 112:133) on the scored frames, as a third detector next to ViTPose and Sapiens for
the proxy evaluation (feet in particular, which no other independent detector covers).

Person boxes as run_sapiens_pose.py (work/body bbox, xyxy full-image pixels). Runs on CPU with onnxruntime in the
rtm venv (rtmlib, onnxruntime, opencv-python-headless).

Output: <out>/episode_XXXXXX.npz with frames [N], kp [N,133,2] (full-image pixels), score [N,133].

  /work/doreen071/v2d/envs/rtm/bin/python compute_wholebody_kp2d.py --out WORK/sap_rtmw [--step 1]
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
MODEL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
         "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{WORK}/sap_rtmw")
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--step", type=int, default=1, help="every n-th scored frame")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    import onnxruntime as ort
    _session = ort.InferenceSession

    def session(*args, **kw):  # fixed thread count
        so = ort.SessionOptions()
        so.intra_op_num_threads = a.threads
        so.inter_op_num_threads = 1
        kw["sess_options"] = so
        return _session(*args, **kw)
    ort.InferenceSession = session
    from rtmlib import RTMPose
    model = RTMPose(onnx_model=MODEL, model_input_size=(288, 384), backend="onnxruntime", device="cpu")
    scored = {int(k): v for k, v in json.load(open(f"{WORK}/body_v2/scored_frames.json")).items()}
    os.makedirs(a.out, exist_ok=True)
    for ep in a.episodes:
        if os.path.exists(f"{a.out}/episode_{ep:06d}.npz"):
            continue
        t0 = time.time()
        b = np.load(f"{WORK}/body/episode_{ep:06d}.npz")
        valid, box = b["valid"].astype(bool), b["bbox"]
        frames = [f for f in scored[ep][::a.step] if f < len(valid) and valid[f]]
        want = set(frames)
        cap = cv2.VideoCapture(VIDEO.format(ep))
        kps, scs, done = [], [], []
        for t in range(max(frames) + 1 if frames else 0):
            if t not in want:
                cap.grab()
                continue
            ok, img = cap.read()
            if not ok:
                break
            k, s = model(img, bboxes=[box[t].tolist()])
            kps.append(k[0])
            scs.append(s[0])
            done.append(t)
        cap.release()
        np.savez_compressed(f"{a.out}/episode_{ep:06d}.npz", frames=np.array(done),
                            kp=np.asarray(kps, np.float32), score=np.asarray(scs, np.float32))
        print(f"ep {ep}: {len(done)} frames, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
