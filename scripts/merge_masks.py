"""Merge SAM 3 masks (precise, but can drop the object for long stretches) with the
Grounding DINO + SAM 2.1 masks (complete coverage) into the final WORK/masks.

Per frame, the SAM 3 object mask is used when it is present and lies inside the (dilated)
SAM 2 mask (precision > 0.7), otherwise the SAM 2 mask is used. If the two pipelines picked
different objects for the episode (median precision over co-visible frames < 0.7, or SAM 3
covers < 20% of frames) the SAM 2 masks are used throughout. Person masks: SAM 3 where it
agrees with SAM 2 (IoU > 0.5), else SAM 2.

Adds `obj_source` [T] uint8 (0 = SAM 2, 1 = SAM 3) and `person_source`.

Usage: python merge_masks.py --sam2 WORK/masks_sam2 --sam3 WORK/masks_sam3 --out WORK/masks
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, pack_masks, unpack_masks  # noqa: E402
from segment_objects import MASK_SCALE, mask_bbox  # noqa: E402


def merge_episode(ep, d2, d3, out):
    a = np.load(os.path.join(d2, f"episode_{ep:06d}.npz"))
    b = np.load(os.path.join(d3, f"episode_{ep:06d}.npz"))
    h, w = int(a["mask_h"]), int(a["mask_w"])
    T = int(a["num_frames"])
    m2, m3 = unpack_masks(a["obj_masks"], h, w), unpack_masks(b["obj_masks"], h, w)
    p2, p3 = unpack_masks(a["person_masks"], h, w), unpack_masks(b["person_masks"], h, w)
    ker = np.ones((11, 11), np.uint8)
    prec = np.full(T, np.nan)
    for t in range(T):
        if m3[t].any() and m2[t].any():
            prec[t] = (m3[t] & (cv2.dilate(m2[t].astype(np.uint8), ker) > 0)).sum() / m3[t].sum()
    cov3 = m3.reshape(T, -1).any(1).mean()
    ok_ep = np.isfinite(prec).sum() > 0 and np.nanmedian(prec) > 0.7 and cov3 > 0.2
    use3 = bool(ok_ep) & (np.nan_to_num(prec, nan=0.0) > 0.7)
    obj = np.where(use3[:, None, None], m3, m2)
    score = np.where(use3, b["obj_score"], a["obj_score"]).astype(np.float32)
    pinter = (p2 & p3).reshape(T, -1).sum(1)
    puni = (p2 | p3).reshape(T, -1).sum(1)
    puse3 = (pinter / np.maximum(puni, 1)) > 0.5
    per = np.where(puse3[:, None, None], p3, p2)

    def bboxes(m):
        return np.stack([mask_bbox(x) for x in m]) / MASK_SCALE

    info = dict(merged=True, sam3_episode_ok=bool(ok_ep), sam3_frac=float(use3.mean()),
                sam2_info=json.loads(str(a["info"])), sam3_info=json.loads(str(b["info"])))
    np.savez_compressed(
        os.path.join(out, f"episode_{ep:06d}.npz"),
        obj_masks=pack_masks(obj), obj_score=score, obj_area=obj.reshape(T, -1).sum(1).astype(np.int32),
        obj_bbox=bboxes(obj).astype(np.float32), obj_source=use3.astype(np.uint8),
        person_masks=pack_masks(per), person_score=np.where(puse3, b["person_score"], a["person_score"]).astype(np.float32),
        person_area=per.reshape(T, -1).sum(1).astype(np.int32), person_bbox=bboxes(per).astype(np.float32),
        person_source=puse3.astype(np.uint8),
        mask_h=h, mask_w=w, full_h=int(a["full_h"]), full_w=int(a["full_w"]), num_frames=T, info=json.dumps(info))
    return ok_ep, use3.mean(), puse3.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam2", default=os.path.join(WORK, "masks_sam2"))
    ap.add_argument("--sam3", default=os.path.join(WORK, "masks_sam3"))
    ap.add_argument("--out", default=os.path.join(WORK, "masks"))
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for ep in a.episodes:
        ok, f3, pf3 = merge_episode(ep, a.sam2, a.sam3, a.out)
        print(f"ep {ep}: sam3_ok={ok} obj_from_sam3={f3:.2f} person_from_sam3={pf3:.2f}", flush=True)


if __name__ == "__main__":
    main()
