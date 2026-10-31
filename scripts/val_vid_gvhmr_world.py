"""GVHMR's world-frame (gravity-view) root trajectory mapped back into the static camera, as a pelvis trajectory source.

GVHMR predicts two root trajectories: the camera-frame one ("incam", per-frame transl from the box and the focal)
and a world one (integrated root velocity with static-contact post-processing). With a static camera the world-to-
camera map is one rigid transform. Per episode:
  R  = chordal mean over frames of R_incam(t) R_global(t)^T     (root orientations, axis-angle in the NPZ)
  c  = median over valid frames of transl_incam(t) - R transl_global(t)
  p_world_in_cam(t) = R transl_global(t) + c
The SMPL-X root joint sits at transl + a constant (betas) offset in both frames, so the difference
d(t) = p_world_in_cam(t) - transl_incam(t) is the pelvis shift. It is added to the root translation of the assembled
GVHMR run (runs/vid_gvhmr), giving runs/vid_gvhmrw: same body, world-derived trajectory. That run is scored alone and
used as --src in val_vid_traj.py.

  python val_vid_gvhmr_world.py --smplx /work/doreen071/v2d_val/root/work/body_vid_gvhmr/smplx \
      --run /work/doreen071/v2d_val/runs/vid_gvhmr/episodes --out /work/doreen071/v2d_val/runs/vid_gvhmrw/episodes
"""
import argparse
import os
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhr_utils import cam_t_to_root_trans  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smplx", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    eps = sorted(int(f[8:14]) for f in os.listdir(a.run) if f.startswith("episode_") and f.endswith(".npz"))
    for ep in eps:
        s = np.load(os.path.join(a.smplx, f"episode_{ep:06d}.npz"))
        valid = s["valid"].astype(bool)
        Ri = Rotation.from_rotvec(s["smplx_global_orient"]).as_matrix()
        Rg = Rotation.from_rotvec(s["global_global_orient"]).as_matrix()
        M = np.einsum("tij,tkj->tik", Ri, Rg)[valid].sum(0)
        U, _, Vt = np.linalg.svd(M)
        R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        spread = np.degrees(Rotation.from_matrix(np.einsum("tij,tkj->tik", Ri, Rg) @ R.T).magnitude())
        tg = s["global_transl"] @ R.T
        ti = s["smplx_transl"]
        c = np.median((ti - tg)[valid], 0)
        d = tg + c - ti
        z = dict(np.load(os.path.join(a.run, f"episode_{ep:06d}.npz")))
        T = len(z["pose"])
        pose = np.asarray(z["pose"], np.float64).copy()
        pose[:, 0:3] += cam_t_to_root_trans(d[:T])
        z.update(pose=pose)
        np.savez(os.path.join(a.out, f"episode_{ep:06d}.npz"), **z)
        for ext in (".ply", ".obj", ".glb", ".off", ".stl"):
            m = os.path.join(a.run, f"episode_{ep:06d}_object{ext}")
            if os.path.exists(m):
                shutil.copy(m, a.out)
        print(f"ep {ep}: R spread median {np.median(spread[valid]):.1f} deg (p90 {np.percentile(spread[valid], 90):.1f}); "
              f"|d| mean {np.linalg.norm(d, axis=1).mean() * 100:.1f} cm, dz mean abs {np.abs(d[:, 2]).mean() * 100:.1f} cm",
              flush=True)


if __name__ == "__main__":
    main()
