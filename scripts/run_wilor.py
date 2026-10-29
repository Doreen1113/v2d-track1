"""Per-frame 3D hands with WiLoR (hand detector + MANO regressor with feature refinement) on every frame of an episode.

The WiLoR YOLO detector finds hand boxes on the full frame (class-agnostic NMS, the left / right class and score are
kept). Every detection is regressed twice, once as a right hand and once as a left hand (mirrored crop), so the side
can be decided later from the body (transfer_hand_wilor.py) without re-running the network.

Output <out>/episode_XXXXXX.npz:
  T, width, height
  det_frame [N]            frame index of each detection
  det_box [N,4]            x0 y0 x1 y1 (full-image pixels), det_conf [N], det_cls [N] (1 = right, detector's class)
  crop_size [N]            side of the square crop that was fed to the network (pixels)
  per side s (0 = regressed as left, 1 = as right), arrays [N,2,...]:
    kp3d [N,2,21,3]        MANO joints (OpenPose hand order: wrist, thumb 1-4, index, middle, ring, little), metres,
                           crop-camera axes (x right, y down, z forward), mirrored back for s = 0
    kp2d [N,2,21,2]        WiLoR's own projection, full-image pixels
    global_orient [N,2,3,3], hand_pose [N,2,15,3,3] (network output, right-hand MANO space), betas [N,2,10],
    cam [N,2,3]            (s, tx, ty) weak-perspective crop camera

  python run_wilor.py --episodes 0 1 2 --out WORK/hand_wilor
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = "/work/doreen071/v2d"
WILOR = f"{ROOT}/third_party/WiLoR"
ASSETS = f"{ROOT}/assets/wilor"
VIDEO = f"{ROOT}/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{{:06d}}.mp4"
sys.path.insert(0, WILOR)


def load_model(device):
    from wilor.configs import get_config
    from wilor.models import WiLoR
    cfg = get_config(f"{WILOR}/pretrained_models/model_config.yaml", update_cachedir=True)
    cfg.defrost()
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
    cfg.MANO.DATA_DIR = f"{ASSETS}/mano_data"
    cfg.MANO.MODEL_PATH = f"{ASSETS}/mano_data"
    cfg.MANO.MEAN_PARAMS = f"{ASSETS}/mano_data/mano_mean_params.npz"
    cfg.MANO.EXT = "npz"  # MANO_RIGHT.npz: arrays only (converted from the pickle with a restricted unpickler)
    cfg.freeze()
    model = WiLoR.load_from_checkpoint(f"{ASSETS}/official/wilor_final.ckpt", strict=False, cfg=cfg,
                                       init_renderer=False, map_location="cpu")
    return model.to(device).eval(), cfg


def crops(img, boxes, rescale, cfg):
    """Square crops (256x256) around each box, as WiLoR's ViTDetDataset, for both sides."""
    from wilor.datasets.utils import expand_to_aspect_ratio, generate_image_patch_cv2
    mean = 255.0 * np.array(cfg.MODEL.IMAGE_MEAN)
    std = 255.0 * np.array(cfg.MODEL.IMAGE_STD)
    S = cfg.MODEL.IMAGE_SIZE
    out, centers, sizes = [], [], []
    for b in boxes:
        c = (b[:2] + b[2:4]) / 2.0
        size = expand_to_aspect_ratio(rescale * (b[2:4] - b[:2]), target_aspect_ratio=cfg.MODEL.BBOX_SHAPE).max()
        for right in (0, 1):
            patch, _ = generate_image_patch_cv2(img, c[0], c[1], size, size, S, S, right == 0, 1.0, 0,
                                                border_mode=cv2.BORDER_CONSTANT)
            patch = patch[:, :, ::-1].astype(np.float32)  # RGB
            out.append(((patch - mean) / std).transpose(2, 0, 1))
        centers.append(c)
        sizes.append(size)
    return np.stack(out).astype(np.float32), np.array(centers), np.array(sizes)


@torch.no_grad()
def regress(model, x, device, bs=128):
    res = {k: [] for k in ("kp3d", "kp2d", "global_orient", "hand_pose", "betas", "cam")}
    for a in range(0, len(x), bs):
        xb = torch.from_numpy(x[a:a + bs]).to(device)
        o = model.forward_step({"img": xb}, train=False)
        p = o["pred_mano_params"]
        n = len(xb)
        res["kp3d"].append(o["pred_keypoints_3d"].float().cpu().numpy())
        res["kp2d"].append(o["pred_keypoints_2d"].float().cpu().numpy())
        res["global_orient"].append(p["global_orient"].reshape(n, 3, 3).float().cpu().numpy())
        res["hand_pose"].append(p["hand_pose"].reshape(n, 15, 3, 3).float().cpu().numpy())
        res["betas"].append(p["betas"].reshape(n, 10).float().cpu().numpy())
        res["cam"].append(o["pred_cam"].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--out", default=f"{ROOT}/work/hand_wilor")
    ap.add_argument("--imgsz", type=int, default=1536, help="detector input size (hands are ~50 px)")
    ap.add_argument("--det-conf", type=float, default=0.1)
    ap.add_argument("--max-det", type=int, default=6)
    ap.add_argument("--rescale", type=float, default=2.0, help="crop = rescale x detector box (WiLoR demo: 2.0)")
    ap.add_argument("--frame-batch", type=int, default=16)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    from ultralytics import YOLO
    dev = torch.device("cuda")
    model, cfg = load_model(dev)
    det = YOLO(f"{ASSETS}/official/detector.pt")
    M = np.diag([-1.0, 1.0, 1.0])
    os.makedirs(a.out, exist_ok=True)
    for ep in a.episodes:
        path = f"{a.out}/episode_{ep:06d}.npz"
        if a.skip_existing and os.path.exists(path):
            continue
        t0 = time.time()
        cap = cv2.VideoCapture(VIDEO.format(ep))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        recs = {k: [] for k in ("det_frame", "det_box", "det_conf", "det_cls", "crop_size", "kp3d", "kp2d",
                                "global_orient", "hand_pose", "betas", "cam")}
        t, done = 0, False
        while not done:
            imgs = []
            for _ in range(a.frame_batch):
                ok, img = cap.read()
                if not ok:
                    done = True
                    break
                imgs.append(img)
            if not imgs:
                break
            dets = det(imgs, conf=a.det_conf, imgsz=a.imgsz, agnostic_nms=True, max_det=a.max_det, verbose=False)
            for k, (img, d) in enumerate(zip(imgs, dets)):
                f = t + k
                if len(d.boxes) == 0:
                    continue
                boxes = d.boxes.xyxy.cpu().numpy().astype(np.float64)
                x, centers, sizes = crops(img, boxes, a.rescale, cfg)
                r = regress(model, x, dev)
                n = len(boxes)
                kp3d = r["kp3d"].reshape(n, 2, 21, 3)
                kp3d[:, 0, :, 0] *= -1  # regressed as left: mirror back
                kp2d = r["kp2d"].reshape(n, 2, 21, 2).copy()
                kp2d[:, 0, :, 0] *= -1
                kp2d = kp2d * sizes[:, None, None, None] + centers[:, None, None, :]
                go = r["global_orient"].reshape(n, 2, 3, 3)
                go[:, 0] = M @ go[:, 0] @ M
                recs["det_frame"].append(np.full(n, f))
                recs["det_box"].append(boxes)
                recs["det_conf"].append(d.boxes.conf.cpu().numpy())
                recs["det_cls"].append(d.boxes.cls.cpu().numpy().astype(np.int8))
                recs["crop_size"].append(sizes)
                recs["kp3d"].append(kp3d)
                recs["kp2d"].append(kp2d)
                recs["global_orient"].append(go)
                recs["hand_pose"].append(r["hand_pose"].reshape(n, 2, 15, 3, 3))
                recs["betas"].append(r["betas"].reshape(n, 2, 10))
                recs["cam"].append(r["cam"].reshape(n, 2, 3))
            t += len(imgs)
        cap.release()
        out = {k: (np.concatenate(v) if v else np.zeros((0,))) for k, v in recs.items()}
        out = {k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in out.items()}
        np.savez_compressed(path, T=t, width=W, height=H, **out)
        nf = len(np.unique(out["det_frame"])) if len(out["det_frame"]) else 0
        print(f"ep {ep}: {t} frames, {len(out['det_frame'])} detections on {nf} frames, {time.time() - t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
