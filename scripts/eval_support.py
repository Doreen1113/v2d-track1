"""Support proxy of assembled Track 1 episodes: height of the object's lowest point above the body floor.

For each episode (body frame, the floor plane of the body directory), over the scored frames:
  floor_rest   median height (cm) over static floor-resting frames (|h| < 30 cm), if any
  table_rest   median height (cm) over static table-resting frames (60-100 cm), if any
  below_max    deepest floor penetration of any scored frame (cm)
  size         body-frame scale of the object (mesh units -> metres)
Static frames come from the object directory's track files (seg_id), which also gives the class sizes.

    python eval_support.py --episodes-dir EPS --body WORK/body_v3 --objects WORK/objects_v5 [--out x.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contactlib as cl  # noqa: E402
from common import load_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-dir", type=Path, required=True)
    ap.add_argument("--body", type=Path, required=True)
    ap.add_argument("--objects", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    import trimesh
    meta = load_meta()
    scored = cl.scored()
    rows = {}
    for ep in a.episodes:
        p = a.episodes_dir / f"episode_{ep:06d}.npz"
        d = np.load(p)
        b = np.load(a.body / f"episode_{ep:06d}.npz")
        n, fd = np.asarray(b["floor_n"], float), float(b["floor_d"])
        mesh = next(q for s in (".ply", ".obj", ".glb") if (q := p.with_name(p.stem + "_object" + s)).exists())
        V = np.asarray(trimesh.load(mesh, process=False).vertices)
        V = V[np.random.default_rng(0).choice(len(V), min(len(V), 4000), replace=False)]
        R, t, s = d["object_rotation"], d["object_translation"], float(d["object_scale"])
        seg = np.load(a.objects / f"episode_{ep:06d}_track.npz")["seg_id"]
        fr = scored[ep]
        fr = fr[fr < len(t)]
        h = np.array([((s * V @ R[f].T + t[f]) @ n - fd).min() for f in fr])
        st = seg[fr] >= 0
        fl = st & (np.abs(h) < 0.3)
        tb = st & (h > 0.6) & (h < 1.0)
        r = dict(cls=meta[ep]["object"], size=s,
                 floor_rest=float(np.median(h[fl]) * 100) if fl.any() else None,
                 table_rest=float(np.median(h[tb]) * 100) if tb.any() else None,
                 below_max=float(np.maximum(-h, 0).max() * 100))
        rows[ep] = r
        print(f"ep {ep:2d} {r['cls']:18s} size {s:.3f} floor {r['floor_rest'] if r['floor_rest'] is None else round(r['floor_rest'], 1)} "
              f"table {r['table_rest'] if r['table_rest'] is None else round(r['table_rest'], 1)} below_max {r['below_max']:.1f}", flush=True)
    fl = [abs(r["floor_rest"]) for r in rows.values() if r["floor_rest"] is not None]
    tb = [r["table_rest"] for r in rows.values() if r["table_rest"] is not None]
    mean = dict(floor_abs_mean=float(np.mean(fl)) if fl else None, table_std=float(np.std(tb)) if tb else None,
                below_max_mean=float(np.mean([r["below_max"] for r in rows.values()])))
    print("MEAN", mean)
    if a.out:
        json.dump(dict(rows={str(k): v for k, v in rows.items()}, mean=mean), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
