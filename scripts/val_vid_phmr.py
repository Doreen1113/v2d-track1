"""PromptHMR-Video (image model + GVHMR-style temporal head, camera-space SMPL-X) on the validation episodes -> MHR.

Follows PromptHMR's pipeline/phmr_vid.py (static camera) with our inputs instead of its detector / SAM 2 / ViTPose:
  - person boxes: root/work/body_gvhmr/preproc/episode_XXXXXX_bbox.npz (= the SAM 3D Body boxes)
  - person masks (mask prompt): root/work/masks (Grounding DINO + SAM 2.1, packed bits at half resolution)
  - COCO-17 ViTPose-H keypoints for the temporal head: root/work/body_gvhmr/preproc/episode_XXXXXX_feat.pt (kp2d)
  - intrinsics: root/work/intrinsics (MoGe-2)
As in the official pipeline, pose / shape come from the temporal head run without keypoints and the translation from
the run with keypoints.

Stages:
  predict  (env /work/doreen071/v2d/envs/phmr) -> <out>/smplx/episode_XXXXXX.npz with smplx_{global_orient,
           body_pose, betas, transl}, valid, K (same keys as run_gvhmr.py)
  convert  (env gvhmr) SMPL-X vertices (GVHMR "supermotion" body model, as PromptHMR's head) -> MHR via
           run_gvhmr.stage_convert -> <out>/episode_XXXXXX.npz (refined-body format, ready for val_assemble.py)

  /work/doreen071/v2d/envs/phmr/bin/python val_vid_phmr.py --stage predict --episodes 0 7
  /home/doreen071/.conda/envs/gvhmr/bin/python val_vid_phmr.py --stage convert --episodes 0 7
"""
import argparse
import json
import os
import sys
import time

import numpy as np

VAL = "/work/doreen071/v2d_val"
W = f"{VAL}/root/work"
PHMR = "/work/doreen071/v2d/third_party_new/PromptHMR"
VIDEO_DIR = f"{VAL}/root/data/track_1/videos/chunk-000/observation.images.exo_camera"
OUT = f"{W}/body_vid_phmr"
IMG = 896


def load_K(ep):
    d = json.load(open(f"{W}/intrinsics/episode_{ep:06d}.json"))
    return np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]], np.float64)


def fill_boxes(raw):
    det = np.isfinite(raw[:, 0])
    idx = np.nonzero(det)[0]
    t = np.arange(len(raw))
    return np.stack([np.interp(t, idx, raw[idx, j]) for j in range(4)], 1), det


def read_frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return out


def stage_predict(ep, a):
    import torch
    from PIL import Image, ImageOps
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
    from torchvision.transforms import Compose, Normalize, ToTensor

    out_path = f"{a.out}/smplx/episode_{ep:06d}.npz"
    if os.path.exists(out_path) and not a.overwrite:
        print(f"ep {ep}: exists", flush=True)
        return
    model, head = get_models()
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, normalize_kp2d
    from hmr4d.utils.geo_transform import compute_cam_angvel

    t0 = time.time()
    raw = np.load(f"{W}/body_gvhmr/preproc/episode_{ep:06d}_bbox.npz")["bbox_raw"].astype(np.float64)
    boxes, det = fill_boxes(raw)
    pre = torch.load(f"{W}/body_gvhmr/preproc/episode_{ep:06d}_feat.pt")
    kp2d = pre["kp2d"].float()  # [T,17,3]
    mz = np.load(f"{W}/masks/episode_{ep:06d}.npz")
    mh, mw = int(mz["mask_h"]), int(mz["mask_w"])
    frames = read_frames(f"{VIDEO_DIR}/episode_{ep:06d}.mp4")
    T = min(len(frames), len(boxes))
    K = torch.from_numpy(load_K(ep)).float()
    norm = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    H, Wd = frames[0].shape[:2]
    scale = IMG / max(H, Wd)
    offset = (IMG - scale * np.array([Wd, H])) / 2
    Kp = K.clone()
    Kp[:2] *= scale
    Kp[:2, 2] += torch.tensor(offset, dtype=torch.float32)
    msk = int(IMG / 14 * 4)
    feats = []
    for s in range(0, T, a.batch):
        batch = []
        for i in range(s, min(T, s + a.batch)):
            img = np.array(ImageOps.pad(ImageOps.contain(Image.fromarray(frames[i]), (IMG, IMG)), size=(IMG, IMG)))
            b = boxes[i] * scale
            b[:2] += offset
            b[2:] += offset
            m = np.unpackbits(mz["person_masks"][i])[: mh * mw].reshape(mh, mw).astype(bool)
            m = np.array(ImageOps.pad(ImageOps.contain(Image.fromarray(m), (msk, msk)), size=(msk, msk)))
            batch.append({"image": norm(img), "image_cv": torch.tensor(img),
                          "boxes": torch.tensor([[*b, 1.0]], dtype=torch.float32), "cam_int": Kp[None].clone(),
                          "kpts": None, "masks": torch.from_numpy(m).float()[None, None], "track_ids": [0]})
        with torch.no_grad(), torch.autocast("cuda"):
            outp = model(batch, mask_prompt=True)
        feats += [o["features"][0].float().cpu() for o in outp]
    f_img = torch.stack(feats)  # [T, ...]
    bbx_xys = get_bbx_xys_from_xyxy(torch.from_numpy(boxes[:T])).numpy()
    bbx_xys = np.array([signal.medfilt(p, 11) for p in bbx_xys.T]).T
    bbx_xys = torch.from_numpy(np.array([gaussian_filter1d(t, 3) for t in bbx_xys.T]).T).float()
    obs = normalize_kp2d(kp2d[:T], bbx_xys).float()
    vb = {"length": torch.tensor([T]), "obs": obs[None], "bbx_xys": bbx_xys[None],
          "K_fullimg": K[None, None].repeat_interleave(T, 1), "cam_angvel": compute_cam_angvel(torch.eye(3).repeat(T, 1, 1))[None],
          "f_imgseq": f_img[None]}
    vb = {k: v.cuda() for k, v in vb.items()}
    with torch.no_grad(), torch.autocast("cuda"):
        o_kp = head.pipeline.forward(vb, train=False, postproc=False, static_cam=True)
        vb["obs"] = torch.zeros_like(vb["obs"])
        o_nokp = head.pipeline.forward(vb, train=False, postproc=False, static_cam=True)
    pc = {k: v[0].float().cpu().numpy() for k, v in o_nokp["pred_smpl_params_incam"].items()}
    pc["transl"] = o_kp["pred_smpl_params_incam"]["transl"][0].float().cpu().numpy()
    valid = det[:T].copy()
    for x, y in zip(np.nonzero(det)[0][:-1], np.nonzero(det)[0][1:]):
        if 1 < y - x <= 61:
            valid[x + 1:y] = True
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, valid=valid, K=load_K(ep), **{f"smplx_{k}": v for k, v in pc.items()},
                        transl_nokp=o_nokp["pred_smpl_params_incam"]["transl"][0].float().cpu().numpy())
    print(f"ep {ep}: PromptHMR-Video T={T} depth median {np.median(pc['transl'][:, 2]):.2f} m {time.time() - t0:.1f}s",
          flush=True)


_M = {}


def get_models():
    if "m" not in _M:
        import torch  # noqa: F401
        from omegaconf import OmegaConf
        os.chdir(PHMR)
        sys.path.insert(0, f"{PHMR}/pipeline/gvhmr")
        sys.path.insert(0, PHMR)
        from prompt_hmr import load_model
        from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
        model = load_model(f"{PHMR}/data/pretrain/phmr/checkpoint.ckpt")
        cfg = OmegaConf.load(f"{PHMR}/data/pretrain/phmr_vid/prhmr_release_002.yaml")
        head = DemoPL(pipeline=cfg.model.pipeline, smplx_path=f"{PHMR}/data/body_models/smplx/SMPLX_NEUTRAL.npz")
        head.load_pretrained_model(f"{PHMR}/data/pretrain/phmr_vid/prhmr_release_002.ckpt")
        _M["m"] = (model, head.eval().cuda())
    return _M["m"]


def stage_convert(ep, a):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import val_vid_gvhmr  # noqa: F401  sets the GVHMR paths (third_party_new) for run_gvhmr
    import run_gvhmr as rg
    os.chdir(rg.GVHMR_ROOT)
    from hmr4d.utils.smplx_utils import make_smplx
    p = f"{a.out}/smplx/episode_{ep:06d}.npz"
    s = dict(np.load(p))
    if "verts" not in s:
        smplx = make_smplx("supermotion").cuda()
        prm = {k[6:]: torch.from_numpy(s[k]).float() for k in s if k.startswith("smplx_")}
        T = len(prm["transl"])
        with torch.no_grad():
            verts = torch.cat([smplx(**{k: v[i:i + 256].cuda() for k, v in prm.items()}).vertices.cpu()
                               for i in range(0, T, 256)]).numpy().astype(np.float32)
        s.update(verts=verts, faces=np.asarray(smplx.faces, np.int64))
        np.savez_compressed(p, **s)
    a2 = argparse.Namespace(out=a.out, overwrite=a.overwrite, smooth_sigma=0.0, precision="float32")
    rg.stage_convert(ep, a2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["predict", "convert"], required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(19)))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for ep in a.episodes:
        (stage_predict if a.stage == "predict" else stage_convert)(ep, a)


if __name__ == "__main__":
    main()
