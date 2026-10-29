"""MoGe-2 focal per frame / background plate, at several token budgets, with and without the foreground mask.

Reads work/fcal_frames (fcal_extract.py). For each image and token budget it records
  fx            focal (px) recovered exactly as MoGeModel.infer does (whole valid map)
  fx_masked     same, but the person/object pixels (dilated) are excluded from the fit
  pp_fit        (f, cx, cy) from a pinhole fit with a free principal point to the masked point map
Rows go to <out>/moge.jsonl (one JSON per line).

    python fcal_moge.py --tokens 3600 9027
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from moge.model.v2 import MoGeModel
from moge.utils.geometry_torch import recover_focal_shift
from scipy.optimize import least_squares

ROOT = Path("/work/doreen071/v2d")


def focal_px(focal, w, h):
    """MoGe focal (relative to half diagonal) -> fx in pixels (square pixels)."""
    ar = w / h
    fx_norm = focal / 2 * (1 + ar ** 2) ** 0.5 / ar
    return float(fx_norm * w)


def pp_fit(points, valid, w, h, f0, grid=96):
    """Fit u = f*x/(z+s)+cx, v = f*y/(z+s)+cy (pixels) on a coarse grid of the affine point map."""
    P = F.interpolate(points.permute(2, 0, 1)[None], (grid, int(grid * w / h)), mode="nearest")[0].permute(1, 2, 0)
    V = F.interpolate(valid[None, None].float(), P.shape[:2], mode="nearest")[0, 0] > 0
    gh, gw = P.shape[:2]
    us = (torch.arange(gw, device=P.device) + 0.5) * w / gw
    vs = (torch.arange(gh, device=P.device) + 0.5) * h / gh
    vv, uu = torch.meshgrid(vs, us, indexing="ij")
    P, uu, vv = P[V].cpu().numpy().astype(np.float64), uu[V].cpu().numpy(), vv[V].cpu().numpy()
    if len(P) < 50:
        return None
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    # initial shift: least squares with f fixed and centred principal point
    def res(p):
        f, cx, cy, s = p
        zz = np.maximum(z + s, 1e-3)
        return np.concatenate([f * x / zz + cx - uu, f * y / zz + cy - vv]) / w
    s0 = 0.0
    try:
        r0 = least_squares(lambda q: res([f0, w / 2, h / 2, q[0]]), [s0], method="lm")
        s0 = float(r0.x[0])
        r = least_squares(res, [f0, w / 2, h / 2, s0], method="lm")
        f, cx, cy, s = r.x
        return [float(f), float(cx), float(cy)]
    except Exception:
        return None


def list_images(fr_dir):
    items = []
    for d in sorted(fr_dir.glob("episode_*")):
        m = json.loads((d / "meta.json").read_text())
        for i in m["frames"]:
            items.append((m["episode"], m["camera"], str(i), d / f"f_{i:06d}.png", d / f"m_{i:06d}.png"))
        items.append((m["episode"], m["camera"], "bg", d / "bg.png", d / "bg_valid.png"))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, default=ROOT / "work/fcal_frames")
    ap.add_argument("--out", type=Path, default=ROOT / "work/fcal_results")
    ap.add_argument("--tokens", type=int, nargs="+", default=[3600, 9027])
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
    print("num_tokens_range", model.num_tokens_range, flush=True)
    items = list_images(a.frames_dir)
    if a.episodes is not None:
        items = [t for t in items if t[0] in set(a.episodes)]
    outf = open(a.out / f"moge{a.tag}.jsonl", "a")
    for k, (ep, cam, fid, ip, mp) in enumerate(items):
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        fg = (m > 0) if fid != "bg" else (m == 0)  # for the plate, the "invalid" region is masked
        x = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)[None].cuda()
        fgt = torch.from_numpy(fg).cuda()
        for nt in a.tokens:
            with torch.inference_mode():
                # identical to MoGeModel.infer's focal recovery
                ref = model.infer(x[0], num_tokens=nt)
                fx_ref = float(ref["intrinsics"][0, 0].item() * w)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model.forward(x, num_tokens=nt)
                pts = out["points"].float()
                mb = out["mask"].float() > 0.5
                focal, _ = recover_focal_shift(pts, mb)
                focal_m, _ = recover_focal_shift(pts, mb & ~fgt[None])
            fx_u, fx_m = focal_px(focal.item(), w, h), focal_px(focal_m.item(), w, h)
            ppf = pp_fit(pts[0], (mb & ~fgt[None])[0], w, h, fx_m)
            rec = {"method": f"moge_t{nt}", "episode": ep, "camera": cam, "frame": fid, "fx_infer": fx_ref,
                   "fx": fx_u, "fx_masked": fx_m, "pp_fit": ppf, "fg_frac": float(fg.mean())}
            outf.write(json.dumps(rec) + "\n")
        outf.flush()
        if k % 20 == 0:
            print(k, len(items), ep, cam, fid, rec, flush=True)


if __name__ == "__main__":
    main()
