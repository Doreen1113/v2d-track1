"""Where does a refined run differ from a reference run in 3D? On the scored frames, after one similarity fitted on the
first scored frame (the CD-H alignment), split the mean vertex change into
  root     change of the pelvis trajectory (cm), and of its depth component
  rel      mean vertex change after removing the per-frame pelvis offset (pose / orientation)
  hands    the same, hand vertices only (contactlib.hand_ids_from)
plus the pelvis-depth change between first and last scored frame (drift) and the frame-0 similarity rotation
split into camera-frame components (x: lean towards / away from the camera, z: in the image plane).

  python diag_body_change.py --run WORK/sap_body --ref WORK/body_vA --sigma-h 3
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_body_proxies import assembled_pose  # noqa: E402
import contactlib  # noqa: E402
from mhr_utils import MHRForward  # noqa: E402

WORK = "/work/doreen071/v2d/work"


def sim3_params(a0, b0):
    ma, mb = a0.mean(0), b0.mean(0)
    u, sv, vt = np.linalg.svd((a0 - ma).T @ (b0 - mb))
    c = np.ones(3)
    c[-1] = -1 if np.linalg.det(u @ vt) < 0 else 1
    R = (u * c) @ vt
    s = (sv * c).sum() / ((a0 - ma) ** 2).sum()
    return s, R, ma, mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--sigma-h", type=float, default=3.0)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mhr = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu")
    scored = {int(k): v for k, v in json.load(open(f"{WORK}/body_v2/scored_frames.json")).items()}
    hv = np.asarray(contactlib.hand_ids_from(mhr.model))
    rows = []
    for ep in a.episodes:
        b, r = np.load(f"{a.run}/episode_{ep:06d}.npz"), np.load(f"{a.ref}/episode_{ep:06d}.npz")
        fr = np.array([f for f in scored[ep] if f < len(b["pose"])])
        V, J = mhr(assembled_pose(b, a.sigma_h)[fr], b["scales"], b["shape"])
        Vr, Jr = mhr(assembled_pose(r, a.sigma_h)[fr], r["scales"], r["shape"])
        s, R, ma, mb = sim3_params(V[0], Vr[0])
        Va = s * (V - ma) @ R + mb
        Ja = s * (J - ma) @ R + mb
        dv = np.linalg.norm(Va - Vr, axis=-1)
        droot = Ja[:, 1] - Jr[:, 1]
        rel = np.linalg.norm((Va - Ja[:, 1:2]) - (Vr - Jr[:, 1:2]), axis=-1)
        row = dict(ep=ep, v2v=dv.mean() * 100, root=np.linalg.norm(droot, axis=-1).mean() * 100,
                   root_z=np.abs(droot[:, 2]).mean() * 100, rel=rel.mean() * 100,
                   depth_drift_run=(J[-1, 1, 2] - J[0, 1, 2]) * 100, depth_drift_ref=(Jr[-1, 1, 2] - Jr[0, 1, 2]) * 100,
                   scale=s)
        # frame-0 similarity rotation (run -> ref) as a rotation vector in the camera frame (deg): x = pitch (lean
        # towards / away from the camera), y = yaw, z = roll about the viewing axis (in the image plane)
        from scipy.spatial.transform import Rotation
        rv = np.degrees(Rotation.from_matrix(R.T).as_rotvec())
        row.update(rot0_deg=float(np.linalg.norm(rv)), rot0_x=float(rv[0]), rot0_y=float(rv[1]), rot0_z=float(rv[2]))
        if hv is not None:
            nh = np.setdiff1d(np.arange(V.shape[1]), hv)
            row["rel_hands"] = rel[:, hv].mean() * 100
            row["rel_nonhands"] = rel[:, nh].mean() * 100
        rows.append(row)
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    keys = [k for k in rows[0] if k != "ep"]
    print("MEAN", " ".join(f"{k}={np.mean([r[k] for r in rows]):.3f}" for k in keys))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
