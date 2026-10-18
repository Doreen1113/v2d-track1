"""Proxy metrics of assembled Track 1 episodes (no ground truth): per episode, on the scored frames,

  iou      mean silhouette IoU of the posed object vs the object masks (actor pixels ignored),
           before (and after, iou_fixed) the fix
  pen      scorer penetration of the hand vertices into the budgeted mesh before the fix (cm, body frame)
  penfr    fraction of scored frames with any penetrating hand vertex (before the fix)
  push     mean / max object displacement applied by the penetration fix (cm), if --fixed is given
  acc      mean |second difference| of the object translation (cm/frame^2) over scored stretches,
           before and after the fix

    python eval_object_candidate.py --prefix EPS_BEFORE_FIX [--fixed EPS_AFTER_FIX] [--out x.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contactlib as cl  # noqa: E402
import objlib  # noqa: E402
from mhr_utils import DEFAULT_MHR, MHRForward  # noqa: E402


def acc(t, frames):
    out = []
    for part in np.split(frames, np.flatnonzero(np.diff(frames) != 1) + 1):
        x = t[part]
        a = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=1)
        out.append(np.concatenate([a[:1], a, a[-1:]]))
    return float(np.concatenate(out).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=Path, required=True)
    ap.add_argument("--fixed", type=Path, default=None)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-iou", action="store_true")
    a = ap.parse_args()
    mhr = MHRForward(DEFAULT_MHR, device="cuda" if torch.cuda.is_available() else "cpu")
    hand_ids = cl.hand_ids_from(mhr.model)
    scored = cl.scored()
    rows = {}
    for ep in a.episodes:
        p = a.prefix / f"episode_{ep:06d}.npz"
        d = np.load(p)
        fr = scored[ep]
        mesh = next(q for s in (".ply", ".obj", ".glb") if (q := p.with_name(p.stem + "_object" + s)).exists())
        mv, mf = cl.budget(mesh)
        R, t, s = d["object_rotation"][fr], d["object_translation"][fr], float(d["object_scale"])
        v, _ = mhr(d["pose"][fr], d["scales"], d["shape"])
        local = np.einsum("tpi,tij->tpj", v[:, hand_ids] - t[:, None], R) / s
        dep = cl.penetration_depth(local.reshape(-1, 3), mv, mf).reshape(local.shape[:2]) * s
        r = dict(pen=float(dep.mean() * 100), penfr=float((dep > 0).any(1).mean()),
                 acc=acc(d["object_translation"], fr))
        if a.fixed is not None:
            q = np.load(a.fixed / f"episode_{ep:06d}.npz")
            push = np.linalg.norm(q["object_translation"][fr] - d["object_translation"][fr], axis=1) * 100
            r.update(push=float(push.mean()), push_max=float(push.max()), acc_fixed=acc(q["object_translation"], fr))
        if not a.no_iou:
            E = objlib.Episode(ep, load_depth=False)
            ious = [objlib.frame_iou(E, f, s * mv @ R[i].T + t[i], mf) for i, f in enumerate(fr)
                    if f < E.T and E.visible[f]]
            r["iou"] = float(np.mean(ious)) if ious else float("nan")
            if a.fixed is not None:
                Rq, tq = q["object_rotation"][fr], q["object_translation"][fr]
                ious = [objlib.frame_iou(E, f, s * mv @ Rq[i].T + tq[i], mf) for i, f in enumerate(fr)
                        if f < E.T and E.visible[f]]
                r["iou_fixed"] = float(np.mean(ious)) if ious else float("nan")
        rows[ep] = r
        print(f"ep {ep:2d} " + " ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)
    keys = list(next(iter(rows.values())).keys())
    mean = {k: float(np.nanmean([r[k] for r in rows.values()])) for k in keys}
    print("MEAN " + " ".join(f"{k} {v:.4f}" for k, v in mean.items()), flush=True)
    if a.out:
        json.dump(dict(rows={str(k): v for k, v in rows.items()}, mean=mean), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
