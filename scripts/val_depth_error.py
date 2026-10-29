"""Characterise the depth (camera z) error of a validation run after the scorer's frame-0 Sim(3).

Per scored frame: aligned predicted pelvis p~ = s p R + t (Sim(3) of the first scored frame, all vertices, as in
val_score.py) and the GT pelvis g (MHR joint 1). The depth error is e_z = p~_z - g_z, reported as a fraction of the
GT depth. Outputs per run:
  - bias / RMS, and the RMS of the parts below 0.1 Hz (drift), 0.1-1 Hz and above 1 Hz (Gaussian low-pass, sigma
    30 / (2 pi f) frames, inside each contiguous stretch)
  - the depth-motion gain: slope of the aligned predicted depth change since f0 against the GT depth change since f0
    (1 = depth motion right, < 1 = the prediction under-reacts)
  - correlations (Spearman, per episode, then the median over episodes) of e_z / z and |e_z| / z with posture and
    visibility features of the frame: torso pitch (root -> neck against the clip's median up direction), pelvis
    height above the feet (along that up), the fraction of projected GT body vertices covered by the object mask
    (object occluding the body), the fraction inside the person mask, and the person mask area (relative to the
    episode median)

  python val_depth_error.py --runs moge va_moge --out /work/doreen071/v2d_val/results/depth_error.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import val_score as vs  # noqa: E402
from common import unpack_masks  # noqa: E402
from mhr_utils import MHRForward  # noqa: E402

VAL = "/work/doreen071/v2d_val"
MASKS = f"{VAL}/root/work/masks"
ROOT, NECK, LFOOT, RFOOT = 1, 110, 4, 20


def ident(z):
    sc, sh = np.asarray(z["scales"], np.float64), np.asarray(z["shape"], np.float64)
    return (np.median(sc, 0) if sc.ndim == 2 else sc), (np.median(sh, 0) if sh.ndim == 2 else sh)


def bands(x, frames):
    """Split a per-frame series into <0.1 Hz, 0.1-1 Hz, >1 Hz parts inside contiguous stretches."""
    lo01, lo1 = np.zeros_like(x), np.zeros_like(x)
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    for a, b in zip([0, *breaks], [*breaks, len(x)]):
        lo01[a:b] = gaussian_filter1d(x[a:b], 30 / (2 * np.pi * 0.1), mode="nearest")
        lo1[a:b] = gaussian_filter1d(x[a:b], 30 / (2 * np.pi * 1.0), mode="nearest")
    return lo01, lo1 - lo01, x - lo1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["moge", "va_moge"])
    ap.add_argument("--gt", default=f"{VAL}/gt")
    ap.add_argument("--scored", default="scored_frames.json")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    kit = vs.load_kit("CD-H")
    fw = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)
    scored = {int(k): np.asarray(v, int) for k, v in json.load(open(os.path.join(a.gt, a.scored))).items()}
    res = {}
    feats_cache = {}
    for run in a.runs:
        rows = {}
        for ep in sorted(scored):
            fr = scored[ep]
            zp = np.load(f"{VAL}/runs/{run}/episodes/episode_{ep:06d}.npz")
            zg = np.load(os.path.join(a.gt, f"episode_{ep:06d}.npz"))
            Vp, Jp = fw(np.asarray(zp["pose"], np.float64)[fr], *ident(zp))
            Vg, Jg = fw(np.asarray(zg["pose"], np.float64)[fr], *ident(zg))
            s, R, t = kit.fit_similarity_transform(Vp[0], Vg[0])
            pa = float(s) * Jp[:, ROOT] @ np.asarray(R) + np.asarray(t)
            g = Jg[:, ROOT]
            frac = (pa[:, 2] - g[:, 2]) / g[:, 2]
            dz_g, dz_p = g[:, 2] - g[0, 2], pa[:, 2] - pa[0, 2]
            gain = float(np.dot(dz_p, dz_g) / max(np.dot(dz_g, dz_g), 1e-12))
            b01, b1, bh = bands(frac, fr)
            if ep not in feats_cache:
                up = np.median(Jg[:, NECK] - Jg[:, ROOT], 0)
                up /= np.linalg.norm(up)
                sp = Jg[:, NECK] - Jg[:, ROOT]
                pitch = np.degrees(np.arccos(np.clip((sp @ up) / np.linalg.norm(sp, axis=1), -1, 1)))
                height = (Jg[:, ROOT] - 0.5 * (Jg[:, LFOOT] + Jg[:, RFOOT])) @ up
                mk = np.load(os.path.join(MASKS, f"episode_{ep:06d}.npz"))
                mh, mw = int(mk["mask_h"]), int(mk["mask_w"])
                sx = mw / float(zg["cx"] * 2)
                occ, inper, area = np.zeros(len(fr)), np.zeros(len(fr)), np.zeros(len(fr))
                for i, f in enumerate(fr):
                    v = Vg[i, ::4]
                    u = np.round((float(zg["fx"]) * v[:, 0] / v[:, 2] + float(zg["cx"])) * sx).astype(int)
                    w = np.round((float(zg["fy"]) * v[:, 1] / v[:, 2] + float(zg["cy"])) * sx).astype(int)
                    ok = (u >= 0) & (u < mw) & (w >= 0) & (w < mh)
                    O = unpack_masks(mk["obj_masks"], mh, mw, f)
                    Pm = unpack_masks(mk["person_masks"], mh, mw, f)
                    occ[i] = O[w[ok], u[ok]].mean() if ok.any() else np.nan
                    inper[i] = Pm[w[ok], u[ok]].mean() if ok.any() else np.nan
                    area[i] = Pm.sum()
                area = area / max(np.median(area), 1)
                feats_cache[ep] = dict(pitch=pitch, height=height, occ=occ, in_person=inper, mask_area=area,
                                       gt_dz=dz_g)
            F = feats_cache[ep]
            corr = {}
            for k, x in F.items():
                okk = np.isfinite(x)
                if np.nanstd(x[okk]) < 1e-9:
                    continue
                corr[f"{k}_vs_err"] = float(spearmanr(x[okk], frac[okk])[0])
                corr[f"{k}_vs_abserr"] = float(spearmanr(x[okk], np.abs(frac[okk]))[0])
            rows[ep] = dict(
                bias_pct=float(frac.mean() * 100), rms_pct=float(np.sqrt((frac ** 2).mean()) * 100),
                rms_below01hz_pct=float(np.sqrt((b01 ** 2).mean()) * 100),
                rms_01to1hz_pct=float(np.sqrt((b1 ** 2).mean()) * 100),
                rms_above1hz_pct=float(np.sqrt((bh ** 2).mean()) * 100),
                rms_cm=float(np.sqrt(((pa[:, 2] - g[:, 2]) ** 2).mean()) * 100),
                gt_depth_range_cm=float((g[:, 2].max() - g[:, 2].min()) * 100), depth_motion_gain=gain,
                pitch_p90=float(np.percentile(F["pitch"], 90)), height_range_cm=float(np.ptp(F["height"]) * 100),
                occ_mean=float(np.nanmean(F["occ"])), **corr)
            r = rows[ep]
            print(f"{run} ep {ep:2d}: bias {r['bias_pct']:+5.2f}% rms {r['rms_pct']:4.2f}% (<0.1Hz {r['rms_below01hz_pct']:4.2f}"
                  f" 0.1-1 {r['rms_01to1hz_pct']:4.2f} >1Hz {r['rms_above1hz_pct']:4.2f}) {r['rms_cm']:5.1f} cm, "
                  f"GT range {r['gt_depth_range_cm']:5.1f} cm, gain {gain:4.2f} | pitch90 {r['pitch_p90']:4.1f} "
                  f"occ {r['occ_mean']:.2f}", flush=True)
        keys = list(next(iter(rows.values())))
        summ = {k: float(np.nanmedian([rows[e].get(k, np.nan) for e in rows])) for k in keys}
        summ.update({f"mean_{k}": float(np.nanmean([rows[e].get(k, np.nan) for e in rows])) for k in
                     ("rms_pct", "rms_below01hz_pct", "rms_01to1hz_pct", "rms_above1hz_pct", "rms_cm", "bias_pct",
                      "depth_motion_gain")})
        print(f"{run} MEDIAN " + " ".join(f"{k}={v:.3g}" for k, v in summ.items()), flush=True)
        res[run] = dict(median=summ, episodes={str(k): v for k, v in rows.items()})
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
