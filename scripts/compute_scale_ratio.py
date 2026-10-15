"""Per-episode depth ratio between the MoGe-2 depth frame (objects) and the body frame.

For a set of frames, the body's MHR vertices are projected with the shared intrinsics; at pixels
covered by both the projected body and the person mask, the MoGe depth is compared with the depth
of the nearest-to-camera body vertex there. The ratio median(MoGe / body) over all such pixels and
frames is written to a JSON file read by assemble_episode.py.

    python compute_scale_ratio.py --body ../../work/body_v2 --out ../../work/moge_to_body_ratio_v2.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_masks  # noqa: E402
from mhr_utils import MHRForward, project  # noqa: E402

ROOT = Path("/work/doreen071/v2d")


def episode_ratio(ep, body_dir, mhr, n_frames):
    J = json.loads((ROOT / f"work/intrinsics/episode_{ep:06d}.json").read_text())
    body = np.load(body_dir / f"episode_{ep:06d}.npz")
    dz = np.load(ROOT / f"work/depth/episode_{ep:06d}.npz")
    depth, ds = dz["depth"], int(dz["downsample"])
    person = load_masks(ep, "person")
    T = min(len(body["pose"]), len(depth), len(person))
    frames = np.linspace(0, T - 1, n_frames).round().astype(int)
    scales = body["scales"] if body["scales"].ndim == 1 else np.median(body["scales"], 0)
    shape = body["shape"] if body["shape"].ndim == 1 else np.median(body["shape"], 0)
    verts, _ = mhr(body["pose"][frames], scales, shape)
    ratios = []
    for k, f in enumerate(frames):
        v = verts[k]
        uv = project(v, J["fx"], J["fy"], J["cx"], J["cy"])
        h, w = depth.shape[1:]
        ui = np.clip((uv[:, 0] / ds).astype(int), 0, w - 1)
        vi = np.clip((uv[:, 1] / ds).astype(int), 0, h - 1)
        # z-buffer: nearest body vertex per depth pixel
        zbuf = np.full((h, w), np.inf)
        np.minimum.at(zbuf, (vi, ui), v[:, 2])
        pm = person[f]
        pm = pm[(np.arange(h) * pm.shape[0] // h)[:, None], (np.arange(w) * pm.shape[1] // w)[None]]
        ok = np.isfinite(zbuf) & pm & np.isfinite(depth[f].astype(float)) & (depth[f] > 0)
        if ok.sum() > 50:
            ratios.append(np.median(depth[f][ok].astype(float) / zbuf[ok]))
    return float(np.median(ratios)), float(np.std(ratios)), len(ratios)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", type=Path, default=ROOT / "work/body_v2")
    ap.add_argument("--out", type=Path, default=ROOT / "work/moge_to_body_ratio_v2.json")
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()
    mhr = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for ep in range(30):
        r, sd, n = episode_ratio(ep, args.body, mhr, args.frames)
        out[str(ep)] = r
        print(f"episode {ep}: ratio {r:.3f} (std {sd:.3f}, {n} frames)", flush=True)
    args.out.write_text(json.dumps({"note": "median MoGe-2 depth / body depth on person pixels over "
                                    f"{args.frames} frames; body = {args.body}", "ratio": out}, indent=1))


if __name__ == "__main__":
    main()
