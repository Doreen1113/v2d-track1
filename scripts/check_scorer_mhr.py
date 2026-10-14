"""Check that the Track 1 scorer's numpy MHR forward (metric_code/track_1/CD-H.py::_MHRBody) and the
official MHR v1.0.1 TorchScript model agree on vertices/joints for the same (pose, scales, shape),
up to the scorer's diag(1,-1,-1) and /100. The scorer's embedded asset is not distributed, so an
equivalent asset is rebuilt from the TorchScript buffers.

  python check_scorer_mhr.py
"""
import importlib.util
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhr_utils import DEFAULT_MHR, MHRForward  # noqa: E402

CDH = "/work/doreen071/v2d/kit/v2d_submission_kit/metric_code/track_1/CD-H.py"


def load_scorer():
    spec = importlib.util.spec_from_file_location("cdh", CDH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def asset_from_torchscript(path):
    m = torch.jit.load(path, map_location="cpu")
    b = {k: v.detach().double().numpy() if v.is_floating_point() else v.detach().numpy()
         for k, v in list(m.named_buffers()) + list(m.named_parameters())}
    c = "character_torch."
    nv = b[c + "mesh.rest_vertices"].shape[0]
    skin = np.zeros((nv, 127))
    np.add.at(skin, (b[c + "linear_blend_skinning.vert_indices_flattened"],
                     b[c + "linear_blend_skinning.skin_indices_flattened"]),
              b[c + "linear_blend_skinning.skin_weights_flattened"])
    import scipy.sparse as sp
    w2 = sp.csr_matrix(b["pose_correctives_model.pose_dirs_predictor.2.weight"])
    pt = b[c + "parameter_transform.parameter_transform"][:, :204]
    w1_idx = b["pose_correctives_model.pose_dirs_predictor.0.sparse_indices"]
    return {
        "joint_parents": b[c + "skeleton.joint_parents"],
        "joint_offsets": b[c + "skeleton.joint_translation_offsets"],
        "joint_prerotations": b[c + "skeleton.joint_prerotations"],
        "parameter_transform": pt,
        "inverse_bind_pose": b[c + "linear_blend_skinning.inverse_bind_pose"],
        "base_shape": b[c + "blend_shape.base_shape"],
        "shape_vectors": b[c + "blend_shape.shape_vectors"],
        "skin_dense": skin,
        "corr_w1_idx": w1_idx,
        "corr_w1_val": b["pose_correctives_model.pose_dirs_predictor.0.sparse_weight"],
        "corr_w1_shape": np.array([w1_idx[0].max() + 1, 125 * 6]),
        "corr_w2_data": w2.data, "corr_w2_indices": w2.indices, "corr_w2_indptr": w2.indptr,
        "corr_w2_shape": np.array(w2.shape),
        **{f"role_{k}": np.arange(10) for k in ("alignment", "body", "left_hand", "right_hand")},
    }


def main():
    cdh = load_scorer()
    body = cdh._MHRBody(asset_from_torchscript(DEFAULT_MHR))
    ref = MHRForward(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)
    rng = np.random.default_rng(0)
    T = 6
    pose = np.zeros((T, 136))
    pose[:, :3] = rng.normal(0, 5, (T, 3))         # up to ~1 m root offsets
    pose[:, 3:6] = rng.uniform(-np.pi, np.pi, (T, 3))
    pose[:, 6:] = rng.normal(0, 0.3, (T, 130))
    scales = rng.normal(0, 0.05, (T, 68))
    shape = rng.normal(0, 1.0, 45)
    pose[0] = 0
    scales[0] = 0
    v_s, j_s = body(np.concatenate([pose, scales], 1), shape)
    v_r, j_r = ref(pose, scales, shape)
    print(f"scorer vs official verts: max {np.abs(v_s - v_r).max() * 1000:.6f} mm; "
          f"joints: max {np.abs(j_s - j_r).max() * 1000:.6f} mm")

    # Root-translation unit: +1 in pose[:, k] for k=0..2 shifts every vertex rigidly.
    p1 = pose[1:2].copy()
    base, _ = ref(p1, scales[1], shape)
    for k in range(3):
        p2 = p1.copy()
        p2[0, k] += 1.0
        moved, _ = ref(p2, scales[1], shape)
        d = (moved - base)[0]
        print(f"pose[{k}] +1 -> vertex shift mean {np.round(d.mean(0), 6)} m, spread {d.std(0).max():.2e}")


if __name__ == "__main__":
    main()
