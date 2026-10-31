"""Replace the pelvis trajectory of an assembled validation run with NLF's (keeping its articulation, root orientation
and identity), or fuse the two depth trajectories.

The source trajectory is the pelvis (MHR joint 1) of an assembled NLF run (val_nlf_to_mhr.py -> val_assemble.py,
sigma_h 3), or with --src raw the SMPL pelvis of the raw NLF output smoothed with the same Gaussian. NLF's body has
its own size, so its depth is put on the base run's scale by one factor per episode, k = median_t(z_base / z_nlf);
only the shape of NLF's trajectory over time is used (relative depth z_nlf(t) / median z_nlf).

Modes (p = base pelvis, q = k * NLF pelvis, both in the camera frame; the whole body is shifted rigidly):
  z        p' = (p_x, p_y, q_z)                      depth only (like oracle b)
  ray      p' = p * q_z / p_z                         depth from NLF, moved along the base pelvis ray (same 2D)
  xyz      p' = q                                     whole pelvis path from NLF
  geo<w>   ray with z' = p_z^(1-w) q_z^w              geometric blend of the two depth paths (e.g. geo0.5)
  lp<s>    ray with z' = p_z * G_s(q_z / p_z)         NLF's depth correction low-passed with a Gaussian of s frames
  geolp<w>_<s>  ray with z' = p_z * exp(G_s(w log(q_z / p_z)))   geo<w> with its correction low-passed (less jitter)

  python val_nlf_traj.py --base /work/doreen071/v2d_val/runs/va_moge/episodes \
      --nlf /work/doreen071/v2d_val/runs/nlf/episodes --mode ray --out /work/doreen071/v2d_val/runs/nlftraj_ray_va/episodes
"""
import argparse
import os
import shutil
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhr_utils import MHRForward, cam_t_to_root_trans  # noqa: E402


def ident(z):
    sc, sh = np.asarray(z["scales"], np.float64), np.asarray(z["shape"], np.float64)
    return (np.median(sc, 0) if sc.ndim == 2 else sc), (np.median(sh, 0) if sh.ndim == 2 else sh)


def nlf_pelvis(a, fw, ep, T):
    if a.src == "raw":
        z = np.load(os.path.join(a.nlf, f"episode_{ep:06d}.npz"))
        q = np.asarray(z["joints"][:, 0], np.float64)
        v = np.asarray(z["valid"], bool)
        idx = np.flatnonzero(v)
        for c in range(3):
            q[:, c] = np.interp(np.arange(len(q)), idx, q[idx, c])
        return gaussian_filter1d(q, a.sigma, axis=0, mode="nearest")[:T]
    z = np.load(os.path.join(a.nlf, f"episode_{ep:06d}.npz"))
    sc, sh = ident(z)
    _, J = fw(np.asarray(z["pose"], np.float64), sc, sh)
    return J[:T, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="assembled run whose pose / identity is kept")
    ap.add_argument("--nlf", required=True, help="assembled NLF run (--src mhr) or raw NLF dir (--src raw)")
    ap.add_argument("--src", default="mhr", choices=["mhr", "raw"])
    ap.add_argument("--sigma", type=float, default=3.0, help="smoothing of the raw NLF pelvis (--src raw)")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--with-object", action="store_true",
                    help="shift the object by the same per-frame offset as the body (keeps their relative pose, "
                         "so contacts and a carved object mesh stay valid)")
    a = ap.parse_args()
    fw = MHRForward(device="cuda", dtype=torch.float64)
    eps = a.episodes if a.episodes is not None else sorted(
        int(f[8:14]) for f in os.listdir(a.base) if f.startswith("episode_") and f.endswith(".npz"))
    os.makedirs(a.out, exist_ok=True)
    for ep in eps:
        zp = dict(np.load(os.path.join(a.base, f"episode_{ep:06d}.npz")))
        pose = np.asarray(zp["pose"], np.float64)
        sc, sh = ident(zp)
        _, Jp = fw(pose, sc, sh)
        p = Jp[:, 1]
        T = len(p)
        q = nlf_pelvis(a, fw, ep, T)
        k = float(np.median(p[:, 2] / q[:, 2]))
        q = k * q
        m = a.mode
        if m == "z":
            new = p.copy()
            new[:, 2] = q[:, 2]
        elif m == "xyz":
            new = q
        else:
            if m == "ray":
                zn = q[:, 2]
            elif m.startswith("geolp"):
                w, sg = (float(x) for x in m[5:].split("_"))
                zn = p[:, 2] * np.exp(gaussian_filter1d(w * np.log(q[:, 2] / p[:, 2]), sg, mode="nearest"))
            elif m.startswith("geo"):
                w = float(m[3:])
                zn = p[:, 2] ** (1 - w) * q[:, 2] ** w
            elif m.startswith("lp"):
                zn = p[:, 2] * gaussian_filter1d(q[:, 2] / p[:, 2], float(m[2:]), mode="nearest")
            else:
                raise ValueError(m)
            new = p * (zn / p[:, 2])[:, None]
        P = pose.copy()
        P[:, 0:3] += cam_t_to_root_trans(new - p)
        rec = dict(zp)
        rec["pose"] = P
        if a.with_object:
            t = np.asarray(zp["object_translation"], np.float64)
            assert len(t) == T, (t.shape, T)
            rec["object_translation"] = (t + (new - p)).astype(np.asarray(zp["object_translation"]).dtype)
        np.savez(os.path.join(a.out, f"episode_{ep:06d}.npz"), **rec)
        for ext in (".ply", ".obj", ".glb", ".off", ".stl"):
            f = os.path.join(a.base, f"episode_{ep:06d}_object{ext}")
            if os.path.exists(f):
                shutil.copy(f, a.out)
        dz = new[:, 2] - p[:, 2]
        print(f"ep {ep}: k {k:.3f}  depth change mean {dz.mean() * 100:+.1f} cm, std {dz.std() * 100:.1f} cm, "
              f"corr(z_base, z_nlf) {np.corrcoef(p[:, 2], q[:, 2])[0, 1]:.2f}", flush=True)


if __name__ == "__main__":
    main()
