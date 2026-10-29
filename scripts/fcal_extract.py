"""Sample frames, foreground masks and a person-free background plate per episode for focal checks.

For every episode:
  <out>/episode_XXXXXX/f_NNNNNN.png      full-resolution RGB frames (evenly spaced)
  <out>/episode_XXXXXX/m_NNNNNN.png      foreground mask (person + target object, dilated), 255 = foreground
  <out>/episode_XXXXXX/bg.png            masked per-pixel median over many frames (person removed)
  <out>/episode_XXXXXX/bg_valid.png      255 where the plate had at least a few unmasked samples
  <out>/episode_XXXXXX/meta.json          camera, sampled frame indices

    python fcal_extract.py --frames 32 --plate-frames 120
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/work/doreen071/v2d")
VIDEO = str(ROOT / "data/track_1/videos/chunk-000/observation.images.exo_camera") + "/episode_{:06d}.mp4"
META = ROOT / "data/track_1/meta/episodes_metadata.jsonl"


def load_masks(ep, mask_dir):
    z = np.load(mask_dir / f"episode_{ep:06d}.npz")
    h, w = int(z["mask_h"]), int(z["mask_w"])

    def unpack(a):
        return np.unpackbits(a, axis=-1)[..., : h * w].reshape(-1, h, w).astype(bool)

    fg = unpack(z["person_masks"]) | unpack(z["obj_masks"])
    area_ok = np.allclose(unpack(z["person_masks"][:3]).reshape(3, -1).sum(1), z["person_area"][:3])
    return fg, area_ok


def read_frames(cap, idx):
    out = {}
    for i in sorted(set(int(v) for v in idx)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if ok:
            out[i] = img
    return out


def full_mask(m, shape, dilate):
    mm = cv2.resize(m.astype(np.uint8) * 255, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        mm = cv2.dilate(mm, k)
    return mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--plate-frames", type=int, default=120)
    ap.add_argument("--dilate", type=int, default=24, help="mask dilation in full-res pixels")
    ap.add_argument("--mask-dir", type=Path, default=ROOT / "work/masks")
    ap.add_argument("--out", type=Path, default=ROOT / "work/fcal_frames")
    a = ap.parse_args()
    meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(META)}
    a.out.mkdir(parents=True, exist_ok=True)
    for ep in a.episodes:
        d = a.out / f"episode_{ep:06d}"
        d.mkdir(exist_ok=True)
        fg, area_ok = load_masks(ep, a.mask_dir)
        cap = cv2.VideoCapture(VIDEO.format(ep))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = min(n, len(fg))
        samp = np.linspace(0, n - 1, a.frames).round().astype(int)
        frames = read_frames(cap, samp)
        names = []
        for i, img in frames.items():
            cv2.imwrite(str(d / f"f_{i:06d}.png"), img)
            cv2.imwrite(str(d / f"m_{i:06d}.png"), full_mask(fg[i], img.shape, a.dilate))
            names.append(i)
        # background plate: masked median, processed in row strips to bound memory
        pidx = np.linspace(0, n - 1, a.plate_frames).round().astype(int)
        pf = read_frames(cap, pidx)
        keys = sorted(pf)
        H, W = pf[keys[0]].shape[:2]
        stack = np.stack([pf[k] for k in keys])  # T,H,W,3 uint8
        masks = np.stack([full_mask(fg[k], (H, W), a.dilate) > 0 for k in keys])  # T,H,W
        del pf
        bg = np.zeros((H, W, 3), np.uint8)
        valid = np.zeros((H, W), np.uint8)
        for r0 in range(0, H, 64):
            s = stack[:, r0:r0 + 64].astype(np.float32)
            m = masks[:, r0:r0 + 64]
            s[m] = np.nan
            cnt = (~m).sum(0)
            med = np.nanmedian(s, axis=0)
            fallback = np.median(stack[:, r0:r0 + 64].astype(np.float32), axis=0)
            med = np.where(np.isfinite(med), med, fallback)
            bg[r0:r0 + 64] = np.clip(med, 0, 255).astype(np.uint8)
            valid[r0:r0 + 64] = np.where(cnt >= 5, 255, 0)
        cv2.imwrite(str(d / "bg.png"), bg)
        cv2.imwrite(str(d / "bg_valid.png"), valid)
        rec = {"episode": ep, "camera": meta[ep]["camera"], "frames": names, "num_frames": n,
               "plate_frames": len(keys), "mask_bits_ok": bool(area_ok), "dilate": a.dilate}
        (d / "meta.json").write_text(json.dumps(rec, indent=1))
        print(ep, meta[ep]["camera"], len(names), "frames; plate valid %.3f" % (valid > 0).mean(),
              "mask_ok", area_ok, flush=True)


if __name__ == "__main__":
    main()
