"""GVHMR (SMPL-X, camera frame) on Track 1 episodes, converted to MHR parameters.

Fallback human pipeline that does not need SAM 3D Body. Runs in the gvhmr env.

Stages (per episode):
  preproc  main-actor boxes (from track_person_bbox.py) -> gap interpolation + smoothing,
           ViTPose-H COCO-17 keypoints and HMR2 ViT features (no body model needed)
  predict  GVHMR with the episode intrinsics and a static camera -> SMPL-X params in the camera frame
           (needs inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz in the GVHMR checkout)
  convert  SMPL-X vertices -> MHR lod1 topology (barycentric map shipped with Meta's MHR repo,
           tools/mhr_smpl_conversion/assets/smplx2mhr_mapping.npz) -> MHR params with one identity
           per episode (kit tools/track1/mesh_to_mhr_params.py)

Final output: <out>/episode_XXXXXX.npz with
  pose [T,136], scales [68], shape [45], valid [T]  (Track 1 / scorer frame: MHR forward lands in the
  OpenCV camera frame in metres), plus smplx params, per-frame conversion error, intrinsics.

Usage:
  python run_gvhmr.py --episodes 0 --stages preproc
  python run_gvhmr.py --all --shard 0/8 --stages preproc predict convert
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

GVHMR_ROOT = "/work/doreen071/v2d/third_party/GVHMR"
KIT_TOOLS = "/work/doreen071/v2d/kit/v2d_submission_kit/tools/track1"
MAPPING = "/work/doreen071/v2d/third_party/MHR/tools/mhr_smpl_conversion/assets/smplx2mhr_mapping.npz"
MHR_MODEL = "/work/doreen071/v2d/assets/mhr_model.pt"
VIDEO_DIR = "/work/doreen071/v2d/data/track_1/videos/chunk-000/observation.images.exo_camera"
INTR_DIR = "/work/doreen071/v2d/work/intrinsics"
OUT_DIR = "/work/doreen071/v2d/work/body_gvhmr"
SMPLX_FILE = os.path.join(GVHMR_ROOT, "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz")

sys.path.insert(0, GVHMR_ROOT)
sys.path.insert(0, KIT_TOOLS)


def load_K(ep):
    with open(os.path.join(INTR_DIR, f"episode_{ep:06d}.json")) as f:
        d = json.load(f)
    return np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]], np.float64)


def moving_average(x, w=5):
    """Centred moving average along axis 0 with edge padding (same as GVHMR's tracker smoothing)."""
    pad = w // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)])
    k = np.ones(w) / w
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def fill_boxes(raw, max_gap):
    """Interpolate missing boxes, hold at the ends. valid = detected or inside a gap of <= max_gap frames."""
    T = len(raw)
    det = np.isfinite(raw[:, 0])
    if det.sum() == 0:
        raise RuntimeError("no person detected in any frame")
    idx = np.nonzero(det)[0]
    t = np.arange(T)
    box = np.stack([np.interp(t, idx, raw[idx, j]) for j in range(4)], 1)
    valid = det.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            valid[a + 1:b] = True
    return box, valid


def gaussian_smooth(x, sigma):
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def stage_preproc(ep, args):
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
    from hmr4d.utils.preproc import Extractor, VitPoseExtractor

    pp = os.path.join(args.out, "preproc")
    out_path = os.path.join(pp, f"episode_{ep:06d}_feat.pt")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"ep {ep}: preproc exists", flush=True)
        return
    b = np.load(os.path.join(pp, f"episode_{ep:06d}_bbox.npz"))
    box, valid = fill_boxes(b["bbox_raw"].astype(np.float64), args.max_gap)
    box = moving_average(moving_average(box))
    bbx_xyxy = torch.from_numpy(box).float()
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
    video = os.path.join(VIDEO_DIR, f"episode_{ep:06d}.mp4")
    t0 = time.time()
    vitpose = VitPoseExtractor().extract(video, bbx_xys)
    feats = Extractor().extract_video_features(video, bbx_xys)
    torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys, "valid": torch.from_numpy(valid),
                "detected": torch.from_numpy(b["detected"]), "kp2d": vitpose, "f_imgseq": feats}, out_path)
    print(f"ep {ep}: preproc T={len(box)} valid={valid.sum()} {time.time() - t0:.1f}s", flush=True)


_MODEL = {}


def get_gvhmr():
    if "m" not in _MODEL:
        import hydra
        from hydra import compose, initialize_config_module
        from hmr4d.configs import register_store_gvhmr

        with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            register_store_gvhmr()
            cfg = compose(config_name="demo", overrides=["video_name=v2d", "static_cam=True"])
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        _MODEL["m"] = model.eval().cuda()
    return _MODEL["m"]


def stage_predict(ep, args):
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.net_utils import detach_to_cpu
    from hmr4d.utils.smplx_utils import make_smplx

    out_path = os.path.join(args.out, "smplx", f"episode_{ep:06d}.npz")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"ep {ep}: smplx exists", flush=True)
        return
    if not os.path.exists(SMPLX_FILE):
        raise FileNotFoundError(f"SMPL-X model missing: {SMPLX_FILE}")
    pre = torch.load(os.path.join(args.out, "preproc", f"episode_{ep:06d}_feat.pt"))
    T = len(pre["bbx_xys"])
    K = torch.from_numpy(load_K(ep)).float()
    data = {
        "length": torch.tensor(T),
        "bbx_xys": pre["bbx_xys"],
        "kp2d": pre["kp2d"],
        "K_fullimg": K[None].repeat(T, 1, 1),
        "cam_angvel": compute_cam_angvel(torch.eye(3)[None].repeat(T, 1, 1)),
        "f_imgseq": pre["f_imgseq"],
    }
    t0 = time.time()
    pred = detach_to_cpu(get_gvhmr().predict(data, static_cam=True))
    p = {k: v.float() for k, v in pred["smpl_params_incam"].items()}
    smplx = make_smplx("supermotion").cuda()
    with torch.no_grad():
        verts = []
        for a in range(0, T, 256):
            verts.append(smplx(**{k: v[a:a + 256].cuda() for k, v in p.items()}).vertices.cpu())
    verts = torch.cat(verts).numpy().astype(np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, verts=verts, faces=np.asarray(smplx.faces, np.int64),
                        valid=pre["valid"].numpy(), K=load_K(ep),
                        **{f"smplx_{k}": v.numpy() for k, v in p.items()},
                        **{f"global_{k}": v.float().numpy() for k, v in pred["smpl_params_global"].items()})
    print(f"ep {ep}: GVHMR T={T} depth median {np.median(p['transl'][:, 2].numpy()):.2f} m "
          f"{time.time() - t0:.1f}s", flush=True)


def smplx_to_mhr_topology(verts, faces):
    m = np.load(MAPPING)
    tri = faces[m["triangle_ids"]]  # [18439, 3]
    w = m["baryc_coords"].astype(np.float32)
    return (verts[:, tri] * w[None, :, :, None]).sum(2)


def stage_convert(ep, args):
    from mesh_to_mhr_params import convert

    out_path = os.path.join(args.out, f"episode_{ep:06d}.npz")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"ep {ep}: output exists", flush=True)
        return
    s = np.load(os.path.join(args.out, "smplx", f"episode_{ep:06d}.npz"))
    verts = s["verts"]
    if args.smooth_sigma > 0:  # smooth the whole-body translation only (keeps articulation intact)
        c = verts.mean(1)
        verts = verts + (gaussian_smooth(c, args.smooth_sigma) - c)[:, None]
    tgt = smplx_to_mhr_topology(verts, s["faces"]).astype(np.float64)  # OpenCV camera frame, metres
    t0 = time.time()
    out = convert(tgt, MHR_MODEL, precision=args.precision)
    err = np.asarray(out["per_frame_vertex_error_mm"])
    K = s["K"]
    np.savez_compressed(out_path, pose=out["pose"].astype(np.float32), scales=out["scales"].astype(np.float32),
                        shape=out["shape"].astype(np.float32), valid=s["valid"].astype(bool),
                        conv_err_mm=err.astype(np.float32), fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                        smooth_sigma=args.smooth_sigma,
                        **{k: s[k] for k in s.files if k.startswith("smplx_")})
    print(f"ep {ep}: SMPL-X->MHR conversion error mean {err.mean():.2f} mm median {np.median(err):.2f} "
          f"max {err.max():.2f} mm, {time.time() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--stages", nargs="+", default=["preproc", "predict", "convert"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--max-gap", type=int, default=60, help="longest detection gap still marked valid")
    ap.add_argument("--smooth-sigma", type=float, default=0.0, help="Gaussian sigma (frames) on body translation")
    ap.add_argument("--precision", default="float32", choices=["float32", "float64"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    eps = list(range(30)) if args.all or not args.episodes else args.episodes
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        eps = eps[i::n]
    os.chdir(GVHMR_ROOT)  # GVHMR loads its checkpoints from relative paths
    print("episodes:", eps, "stages:", args.stages, flush=True)
    for ep in eps:
        for st in args.stages:
            {"preproc": stage_preproc, "predict": stage_predict, "convert": stage_convert}[st](ep, args)


if __name__ == "__main__":
    main()
