"""Single-image calibration of the Track 1 frames with several pretrained calibrators.

Methods (choose with --methods):
  unidepth     UniDepthV2 ViT-L (lpiccinelli/unidepth-v2-vitl14), resolution_level 9: fx, fy, cx, cy
  depthpro     Depth Pro FoV head: fx (square pixels, centred principal point)
  geocalib     GeoCalib pinhole weights, pinhole model: fx, fy (centred principal point)
  geocalib_d   GeoCalib distorted weights, simple_radial model: f, k1
  anycalib     AnyCalib anycalib_pinhole, cam_id pinhole: fx, fy, cx, cy
  anycalib_g   AnyCalib anycalib_gen, cam_id simple_radial:1: f, cx, cy, k1
  geocalib_shared  GeoCalib pinhole with intrinsics shared over all background plates of a camera
                   and over 48 person-containing frames of a camera

Reads work/fcal_frames (fcal_extract.py); rows go to <out>/<method>.jsonl.

    python fcal_learned.py --methods unidepth depthpro geocalib anycalib
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/work/doreen071/v2d")
DEPTHPRO_CKPT = ROOT / "work/fcal_src/weights/depth_pro.pt"


def list_images(fr_dir, episodes=None):
    items = []
    for d in sorted(fr_dir.glob("episode_*")):
        m = json.loads((d / "meta.json").read_text())
        if episodes is not None and m["episode"] not in episodes:
            continue
        for i in m["frames"]:
            items.append((m["episode"], m["camera"], str(i), d / f"f_{i:06d}.png"))
        items.append((m["episode"], m["camera"], "bg", d / "bg.png"))
    return items


def load_rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def to_t(img):
    return torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1).cuda()


class UniDepth:
    name = "unidepth"

    def __init__(self):
        from unidepth.models import UniDepthV2
        self.m = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").cuda().eval()
        self.m.resolution_level = 9
        self.m.interpolation_mode = "bilinear"

    def __call__(self, img):
        x = torch.from_numpy(img).permute(2, 0, 1).cuda()
        with torch.no_grad():
            K = self.m.infer(x)["intrinsics"][0].float().cpu().numpy()
        return {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])}


class DepthPro:
    name = "depthpro"

    def __init__(self):
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        from dataclasses import replace
        cfg = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(DEPTHPRO_CKPT))
        self.m, self.tf = depth_pro.create_model_and_transforms(config=cfg, device=torch.device("cuda"),
                                                                precision=torch.half)
        self.m.eval()

    def __call__(self, img):
        with torch.no_grad():
            pred = self.m.infer(self.tf(img), f_px=None)
        return {"fx": float(pred["focallength_px"].float().item())}


class GeoCalibM:
    def __init__(self, weights="pinhole", model="pinhole"):
        from geocalib import GeoCalib
        self.m = GeoCalib(weights=weights).cuda().eval()
        self.model = model
        self.name = "geocalib" if weights == "pinhole" else "geocalib_d"

    def run(self, x, shared=False):
        with torch.no_grad():
            r = self.m.calibrate(x, camera_model=self.model, shared_intrinsics=shared)
        return r

    def __call__(self, img):
        r = self.run(to_t(img))
        cam = r["camera"]
        f = cam.f[0].float().cpu().numpy()
        c = cam.c[0].float().cpu().numpy()
        out = {"fx": float(f[0]), "fy": float(f[1]), "cx": float(c[0]), "cy": float(c[1])}
        if self.model != "pinhole":
            out["k1"] = float(cam.k1[0].item())
        g = r["gravity"]
        try:
            out["roll_deg"] = float(np.rad2deg(g.rp[0, 0].item()))
            out["pitch_deg"] = float(np.rad2deg(g.rp[0, 1].item()))
        except Exception:
            pass
        return out


class AnyCalibM:
    def __init__(self, model_id="anycalib_pinhole", cam_id="pinhole"):
        from anycalib import AnyCalib
        self.m = AnyCalib(model_id=model_id).cuda().eval()
        self.cam_id = cam_id
        self.name = "anycalib" if model_id == "anycalib_pinhole" else "anycalib_g"

    def __call__(self, img):
        with torch.no_grad():
            out = self.m.predict(to_t(img), cam_id=self.cam_id)
        k = out["intrinsics"].float().cpu().numpy()
        if self.cam_id == "pinhole":
            return {"fx": float(k[0]), "fy": float(k[1]), "cx": float(k[2]), "cy": float(k[3])}
        return {"fx": float(k[0]), "fy": float(k[0]), "cx": float(k[1]), "cy": float(k[2]), "k1": float(k[3])}


def geocalib_shared(fr_dir, out_dir):
    """GeoCalib with intrinsics shared across many images of one physical camera."""
    gc = GeoCalibM()
    cams = {}
    for d in sorted(fr_dir.glob("episode_*")):
        m = json.loads((d / "meta.json").read_text())
        cams.setdefault(m["camera"], []).append((m["episode"], d, m["frames"]))
    rng = np.random.default_rng(0)
    with open(out_dir / "geocalib_shared.jsonl", "a") as f:
        for cam, eps in sorted(cams.items()):
            plates = torch.stack([to_t(load_rgb(d / "bg.png")) for _, d, _ in eps])
            pool = [(d / f"f_{i:06d}.png") for _, d, fr in eps for i in fr]
            pick = [pool[j] for j in sorted(rng.choice(len(pool), size=min(48, len(pool)), replace=False))]
            frames = torch.stack([to_t(load_rgb(p)) for p in pick])
            for kind, x in (("plates", plates), ("frames48", frames)):
                r = gc.run(x, shared=True)
                fx = r["camera"].f[:, 0].float().cpu().numpy()
                fy = r["camera"].f[:, 1].float().cpu().numpy()
                rec = {"method": "geocalib_shared", "camera": cam, "kind": kind, "n": int(x.shape[0]),
                       "fx": float(np.median(fx)), "fy": float(np.median(fy)), "fx_all": fx.tolist()}
                f.write(json.dumps(rec) + "\n")
                print(rec, flush=True)
            # per-episode shared over that episode's frames (32 frames + plate)
            for ep, d, fr in eps:
                x = torch.stack([to_t(load_rgb(d / f"f_{i:06d}.png")) for i in fr] + [to_t(load_rgb(d / "bg.png"))])
                r = gc.run(x, shared=True)
                fx = r["camera"].f[:, 0].float().cpu().numpy()
                rec = {"method": "geocalib_shared", "camera": cam, "kind": "episode", "episode": ep,
                       "n": int(x.shape[0]), "fx": float(np.median(fx))}
                f.write(json.dumps(rec) + "\n")
                print(rec, flush=True)


def build(name):
    return {
        "unidepth": UniDepth,
        "depthpro": DepthPro,
        "geocalib": lambda: GeoCalibM("pinhole", "pinhole"),
        "geocalib_d": lambda: GeoCalibM("distorted", "simple_radial"),
        "anycalib": lambda: AnyCalibM("anycalib_pinhole", "pinhole"),
        "anycalib_g": lambda: AnyCalibM("anycalib_gen", "simple_radial:1"),
    }[name]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, default=ROOT / "work/fcal_frames")
    ap.add_argument("--out", type=Path, default=ROOT / "work/fcal_results")
    ap.add_argument("--methods", nargs="+", default=["unidepth", "depthpro", "geocalib", "anycalib"])
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N images")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    items = list_images(a.frames_dir, set(a.episodes) if a.episodes is not None else None)
    if a.limit:
        items = items[: a.limit]
    for name in a.methods:
        if name == "geocalib_shared":
            geocalib_shared(a.frames_dir, a.out)
            continue
        meth = build(name)
        with open(a.out / f"{name}.jsonl", "a") as f:
            for k, (ep, cam, fid, p) in enumerate(items):
                img = load_rgb(p)
                try:
                    r = meth(img)
                except Exception as e:  # keep going; log the failure
                    r = {"error": repr(e)[:300]}
                rec = {"method": name, "episode": ep, "camera": cam, "frame": fid, **r}
                f.write(json.dumps(rec) + "\n")
                if k % 50 == 0:
                    f.flush()
                    print(name, k, len(items), rec, flush=True)
        del meth
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
