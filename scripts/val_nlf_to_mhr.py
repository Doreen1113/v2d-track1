"""NLF SMPL meshes (val_nlf_infer.py output) -> MHR parameters, the same way the validation GT was converted
(val_build_behave.py --stage gt): SMPL vertices in the camera frame -> MHR lod1 topology with MHR's SMPL conversion map
(smpl2mhr_mapping.npz) -> the kit's mesh_to_mhr_params.convert (float64, one identity per episode).

Writes a body dir in the refine output format (pose [T,136], scales [68], shape [45], valid [T], conv_err_mm [T]) that
val_assemble.py turns into scoreable episodes.

  python val_nlf_to_mhr.py --raw /work/doreen071/v2d_val/nlf/raw --out /work/doreen071/v2d_val/nlf/body_nlf --episodes 0
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from val_build_behave import KIT_TOOLS, MHR_MODEL, SMPL2MHR, SMPL_DIR  # noqa: E402


def smpl_faces():
    import pickle
    with open(os.path.join(SMPL_DIR, "smpl", "SMPL_NEUTRAL.pkl"), "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return np.asarray(d["f"], np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--verts", default="verts", choices=["verts", "verts_np"],
                    help="parametric SMPL vertices (default) or NLF's non-parametric vertex predictions")
    ap.add_argument("--precision", default="float64", choices=["float32", "float64"])
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, KIT_TOOLS)
    from mesh_to_mhr_params import convert
    os.makedirs(a.out, exist_ok=True)
    mp = np.load(SMPL2MHR)
    tri = smpl_faces()[mp["triangle_ids"]]
    bary = mp["baryc_coords"][None, :, :, None]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for ep in a.episodes:
        out = os.path.join(a.out, f"episode_{ep:06d}.npz")
        if a.skip_existing and os.path.exists(out):
            continue
        t0 = time.time()
        z = np.load(os.path.join(a.raw, f"episode_{ep:06d}.npz"))
        V = np.asarray(z[a.verts], np.float64)
        valid = np.asarray(z["valid"], bool) & np.isfinite(V).all((1, 2))
        V[~valid] = np.nan
        tgt = (V[:, tri] * bary).sum(2)  # [T, 18439, 3]
        res = convert(tgt, MHR_MODEL, device=dev, precision=a.precision, log=lambda *x, **k: None)
        err = np.asarray(res["per_frame_vertex_error_mm"], np.float64)
        np.savez(out, pose=res["pose"].astype(np.float32), scales=res["scales"].astype(np.float32),
                 shape=res["shape"].astype(np.float32), valid=valid, conv_err_mm=err)
        print(f"ep {ep}: SMPL->MHR residual mean {np.nanmean(err[valid]):.2f} mm, max {np.nanmax(err[valid]):.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
