"""Sanity check of the Sapiens2 keypoints (work/sap_kp) against ViTPose-H (COCO-17), RTMW-x and the projected body_vA
model (all 308 regressor rows, so the face points too): median / mean pixel distances per joint group.

  python check_sapiens_kp.py --episodes 0 1 2 [--ref WORK/body_vA]
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sap_proxies import GROUPS, RTMW_FEET, project, regress  # noqa: E402
from mhr_utils import MHRForward  # noqa: E402

WORK = "/work/doreen071/v2d/work"
COCO_TO_70 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
COCO_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri",
              "l_hip", "r_hip", "l_knee", "r_knee", "l_ank", "r_ank"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--ref", default=f"{WORK}/body_vA")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mhr = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu")
    kmap = sp.load_npz(f"{WORK}/sap_kp/kp308_mapping.npz").tocsr()
    rows = []
    per_joint = {n: [] for n in COCO_NAMES}
    for ep in a.episodes:
        p = f"{WORK}/sap_kp/episode_{ep:06d}.npz"
        if not os.path.exists(p):
            continue
        s = np.load(p)
        skp, ssc = s["kp"].astype(np.float64), s["score"].astype(np.float64)
        raw = np.load(f"{WORK}/body/episode_{ep:06d}.npz")
        valid = raw["valid"].astype(bool)
        K = np.array([float(raw["fx"]), float(raw["fy"]), float(raw["cx"]), float(raw["cy"])])
        vit = torch.load(f"{WORK}/body_gvhmr/preproc/episode_{ep:06d}_feat.pt", weights_only=False)
        kv = vit["kp2d"].numpy().astype(np.float64)
        kv[~vit["detected"].numpy().astype(bool) | ~valid, :, 2] = 0
        r = {"ep": ep, "T": len(skp)}
        e = np.linalg.norm(skp[:, COCO_TO_70] - kv[..., :2], axis=-1)
        m = (kv[..., 2] > 0.5) & (ssc[:, COCO_TO_70] > 0.5)
        r["sap_vs_vit_med"] = float(np.median(e[m]))
        r["sap_vs_vit_mean"] = float(e[m].mean())
        r["sap_vs_vit_p95"] = float(np.percentile(e[m], 95))
        for j, n in enumerate(COCO_NAMES):
            per_joint[n].append(e[:, j][m[:, j]])
        es = np.linalg.norm(raw["kp2d"][:, COCO_TO_70] - kv[..., :2], axis=-1)
        r["sam_vs_vit_med"] = float(np.median(es[m]))
        rp = f"{WORK}/sap_rtmw/episode_{ep:06d}.npz"
        if os.path.exists(rp):
            w = np.load(rp)
            fr = w["frames"].astype(int)
            for name, sidx, ridx in (("body", COCO_TO_70, list(range(17))),
                                     ("feet", list(RTMW_FEET.values()), list(RTMW_FEET))):
                e1 = np.linalg.norm(skp[fr][:, sidx] - w["kp"][:, ridx], axis=-1)
                mm = (w["score"][:, ridx] > 3.0) & (ssc[fr][:, sidx] > 0.5)
                r[f"sap_vs_rtmw_{name}_med"] = float(np.median(e1[mm])) if mm.any() else float("nan")
                e2 = np.linalg.norm(kv[fr][:, :17, :2] - w["kp"][:, :17], axis=-1) if name == "body" else None
                if e2 is not None:
                    mm2 = (w["score"][:, :17] > 3.0) & (kv[fr][:, :, 2] > 0.5)
                    r["vit_vs_rtmw_body_med"] = float(np.median(e2[mm2]))
        if a.ref and os.path.exists(f"{a.ref}/episode_{ep:06d}.npz"):
            b = np.load(f"{a.ref}/episode_{ep:06d}.npz")
            V, J = mhr(b["pose"], b["scales"], b["shape"])
            uv = project(regress(kmap, V, J), K)
            for g, ids in GROUPS.items():
                e = np.linalg.norm(uv[:, ids] - skp[:, ids], axis=-1)
                mm = (ssc[:, ids] > 0.5) & valid[:, None]
                r[f"ref_vs_sap_{g}_med"] = float(np.median(e[mm])) if mm.any() else float("nan")
                r[f"sap_{g}_conf_frac"] = float((ssc[:, ids] > 0.5)[valid].mean())
        rows.append(r)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    keys = [k for k in rows[0] if k not in ("ep", "T")]
    print("MEAN over episodes:", {k: round(float(np.nanmean([r.get(k, np.nan) for r in rows])), 2) for k in keys})
    print("per-joint median sap vs vit:", {n: round(float(np.median(np.concatenate(v))), 2) for n, v in per_joint.items() if v})
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
