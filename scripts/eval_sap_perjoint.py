"""Per-joint held-out 2D errors (ViTPose COCO-17 and RTMW body COCO-17) of refined runs, after the assembly smoothing:
which joints move when the Sapiens terms are added.

  python eval_sap_perjoint.py DEV/ho_base DEV/ho_all --episodes 0 3 --sigma-h 3
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_body_proxies import assembled_pose  # noqa: E402
from eval_sap_proxies import COCO_TO_70, WORK, project, regress  # noqa: E402
from mhr_utils import MHRForward  # noqa: E402

NAMES = ["nose", "leye", "reye", "lear", "rear", "lsho", "rsho", "lelb", "relb", "lwri", "rwri", "lhip", "rhip",
         "lkne", "rkne", "lank", "rank"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--sigma-h", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mhr = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu")
    kmap = sp.load_npz(os.path.join(WORK, "sap_kp/kp308_mapping.npz")).tocsr()
    acc = {d: {"vit": [], "rtmw": []} for d in a.dirs}
    for ep in a.episodes:
        raw = np.load(f"{WORK}/body/episode_{ep:06d}.npz")
        valid = raw["valid"].astype(bool)
        vit = torch.load(f"{WORK}/body_gvhmr/preproc/episode_{ep:06d}_feat.pt", weights_only=False)
        kv = vit["kp2d"].numpy().astype(np.float64)
        kv[~vit["detected"].numpy().astype(bool), :, 2] = 0
        rw = np.load(f"{WORK}/sap_rtmw/episode_{ep:06d}.npz")
        for d in a.dirs:
            b = np.load(f"{d}/episode_{ep:06d}.npz")
            src = b if "fx" in b.files else raw
            K = np.array([float(src["fx"]), float(src["fy"]), float(src["cx"]), float(src["cy"])])
            V, J = mhr(assembled_pose(b, a.sigma_h), b["scales"], b["shape"])
            uv = project(regress(kmap, V, J), K)[:, COCO_TO_70]
            ho = b["holdout"].astype(bool) if "holdout" in b.files else np.zeros(len(uv), bool)
            ev = valid & ho if ho.any() else valid
            e = np.linalg.norm(uv - kv[..., :2], axis=-1)
            e[~((kv[..., 2] > 0.5) & ev[:, None])] = np.nan
            acc[d]["vit"].append(np.nanmean(e, 0))
            fr = rw["frames"].astype(int)
            k = ev[fr]
            e = np.linalg.norm(uv[fr[k]] - rw["kp"][k][:, :17], axis=-1)
            e[~(rw["score"][k][:, :17] > 3.0)] = np.nan
            acc[d]["rtmw"].append(np.nanmean(e, 0) if len(e) else np.full(17, np.nan))
    out = {}
    for src in ("vit", "rtmw"):
        print(f"--- {src}: per-joint mean over episodes (first row absolute, others difference to it)")
        print(f"{'':>28s} " + " ".join(f"{n:>6s}" for n in NAMES))
        base = np.nanmean(np.stack(acc[a.dirs[0]][src]), 0)
        for i, d in enumerate(a.dirs):
            m = np.nanmean(np.stack(acc[d][src]), 0)
            out[f"{d}:{src}"] = m.tolist()
            row = m if i == 0 else m - base
            print(f"{d[-28:]:>28s} " + " ".join(f"{x:6.2f}" if i == 0 else f"{x:+6.2f}" for x in row))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
