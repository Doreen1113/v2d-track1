"""Generate several SAM 3D Objects mesh candidates per episode (for the shared per-class mesh).

For each episode, frames are ranked as in reconstruct_object.select_frames (large, single
blob, away from the border, little actor overlap) and picked spread over time (the best
frames of each quarter of the video), so that objects seen in different orientations
(lying / standing, before / after manipulation) all get a reconstruction. For each picked
frame a metric MoGe-2 point map is computed with the fixed intrinsics and SAM 3D Objects is
run with it. Every candidate is kept (mesh + single-frame pose), not only the best one.

Output: WORK/objects_v2/candidates/episode_XXXXXX/fFFFFF.ply + fFFFFF.json
  json: point_cam = s * R @ point_mesh + t (OpenCV camera, MoGe metres), frame, iou, extent

Two stages (different environments):
  --stage prep   (sam3 env, MoGe-2)      frame choice + point maps -> candidates/episode_XXXXXX/points.npz
  --stage recon  (sam3do env, SAM 3D)    meshes for the prepared frames

Usage: python object_candidates.py --stage prep|recon --shard i/n [--per-quarter 2]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import VIDEO_TMPL, WORK, load_intrinsics, read_frame  # noqa: E402
from reconstruct_object import Recon, full_res_mask, pose_from_output, select_frames, silhouette  # noqa: E402


def pick_frames(ep, per_quarter=2, min_gap=30):
    rows, d = select_frames(ep, k=400, min_gap=1)
    T = int(d["num_frames"])
    chosen = []
    for q in range(4):
        a, b = q * T // 4, (q + 1) * T // 4
        n = 0
        for r in rows:  # rows are sorted by selection score
            if a <= r[1] < b and r[0] > 0.3 and all(abs(r[1] - c[1]) >= min_gap for c in chosen):
                chosen.append(r)
                n += 1
                if n >= per_quarter:
                    break
    return chosen, d


def prep(ep, per_quarter):
    from moge.model.v2 import MoGeModel
    global _MOGE
    if _MOGE is None:
        _MOGE = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
    out_dir = os.path.join(WORK, "objects_v2", "candidates", f"episode_{ep:06d}")
    os.makedirs(out_dir, exist_ok=True)
    chosen, _ = pick_frames(ep, per_quarter)
    K, _ = load_intrinsics(ep)
    pts = []
    for row in chosen:
        img = read_frame(VIDEO_TMPL.format(ep), row[1])
        W = img.shape[1]
        fov_x = math.degrees(2 * math.atan(W / (2 * K[0, 0])))
        x = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1).cuda()
        with torch.no_grad():
            p = _MOGE.infer(x, fov_x=fov_x)["points"].float().cpu().numpy()
        pts.append(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16))
    np.savez(os.path.join(out_dir, "points.npz"), chosen=json.dumps(chosen), points=np.stack(pts), K=K)
    print(f"[ep {ep}] frames {[r[1] for r in chosen]}", flush=True)


def recon(ep, rec):
    import trimesh
    t0 = time.time()
    out_dir = os.path.join(WORK, "objects_v2", "candidates", f"episode_{ep:06d}")
    c = np.load(os.path.join(out_dir, "points.npz"))
    chosen = json.loads(str(c["chosen"]))
    d = np.load(os.path.join(WORK, "masks", f"episode_{ep:06d}.npz"))
    K, _ = load_intrinsics(ep)
    for j, (sc, t, occl, cc, border) in enumerate(chosen):
        stem = os.path.join(out_dir, f"f{t:06d}")
        if os.path.exists(stem + ".json"):
            continue
        img = read_frame(VIDEO_TMPL.format(ep), t)
        H, W = img.shape[:2]
        mask = full_res_mask(d, t, H, W)
        try:
            out, _ = rec.run(img, mask, c["points"][j].astype(np.float32))
        except Exception as e:  # keep going on failures
            print(f"[ep {ep}] frame {t} failed: {e}", flush=True)
            continue
        glb = out["glb"]
        V = np.asarray(glb.vertices, np.float64)
        F = np.asarray(glb.faces, np.int64)
        R, tt, s = pose_from_output(out)
        sil = silhouette(s * V @ R.T + tt, F, K, H, W)
        iou = float((sil & mask).sum() / max((sil | mask).sum(), 1))
        trimesh.Trimesh(vertices=V, faces=F, vertex_colors=getattr(glb.visual, "vertex_colors", None),
                        process=False).export(stem + ".ply")
        ext = (V.max(0) - V.min(0)) * s
        json.dump(dict(episode=ep, frame=int(t), R=R.tolist(), t=tt.tolist(), s=float(s), K=K.tolist(),
                       width=W, height=H, iou=iou, select_score=float(sc), person_overlap=float(occl),
                       object_extent_m=ext.tolist()), open(stem + ".json", "w"), indent=1)
        print(f"[ep {ep}] frame {t} iou={iou:.3f} s={s:.3f} ext={np.round(ext, 2)}", flush=True)
    print(f"[ep {ep}] done {len(chosen)} candidates in {time.time() - t0:.0f}s", flush=True)


_MOGE = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "recon"], required=True)
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--shard", type=str, default=None)
    ap.add_argument("--per-quarter", type=int, default=2)
    a = ap.parse_args()
    eps = a.episodes if a.episodes else list(range(30))
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        eps = eps[i::n]
    rec = Recon() if a.stage == "recon" else None
    for ep in eps:
        if a.stage == "prep":
            prep(ep, a.per_quarter)
        else:
            recon(ep, rec)

if __name__ == "__main__":
    main()
