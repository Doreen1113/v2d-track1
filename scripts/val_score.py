"""Score Track 1 episode NPZs against the local 3D validation set (BEHAVE-derived ground truth in MHR).

The metric code is the kit's own (metric_code/track_1/CD-H.py and PEN.py, imported from file): the scorer's numpy
MHR forward (_MHRBody), mesh surface sampling (_sample_mesh_surface, seed = episode index), one Sim(3) fitted on
the alignment vertices of the first scored frame (fit_similarity_transform), symmetric Chamfer
(symmetric_chamfer_distance), per-stretch acceleration (acceleration_error_per_frame, _stretches) and the contact
penetration (_episode_penetration). Only the Kaggle row packing is skipped: arrays are read from the NPZs directly.
The Chamfer / acceleration step runs on torch tensors (the kit functions accept them) so it can use a GPU; the
result equals the numpy path (checked with --check-numpy).

The scorer's embedded MHR asset (and so its vertex roles) is not distributed. The asset is rebuilt from the official
TorchScript model (check_scorer_mhr.asset_from_torchscript); the roles are a choice (--roles):
  all     alignment = body = all 18,439 vertices (default)
  nohand  alignment = body = vertices not skinned to the wrists / fingers
  sub     alignment = all vertices, body = the 4,663-vertex subset shipped with MHR's SMPL conversion tool
Hand points (PEN): 256 per hand, evenly spaced over the hand vertices (left, then right).

Inputs
  --pred DIR   episode_XXXXXX.npz (pose [T,136], scales [68] or [T,68], shape [45] or [T,45]; optional
               object_rotation [T,3,3], object_translation [T,3], object_scale) + episode_XXXXXX_object.<ext>
               (the assemble_episode.py output format). Per-frame scales / shape are reduced by the median.
  --gt DIR     the same format plus scored_frames.json ({episode: [frame, ...]}) and gt_info.json.

  python val_score.py --pred /work/doreen071/v2d_val/runs/moge/episodes --gt /work/doreen071/v2d_val/gt
  python val_score.py --pred ... --gt ... --metrics cd_h acc_h --roles nohand --out result.json
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_scorer_mhr import asset_from_torchscript  # noqa: E402
from mhr_utils import DEFAULT_MHR  # noqa: E402

KIT_METRICS = "/work/doreen071/v2d/kit/v2d_submission_kit/metric_code/track_1"
SUBSET = "/work/doreen071/v2d/third_party/MHR/tools/mhr_smpl_conversion/assets/subsampled_vertex_indices.npy"
ALL_METRICS = ("cd_h", "acc_h", "cd_o", "acc_o", "cd_c", "pen")
TORCH_FORWARD = None


def load_kit(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), os.path.join(KIT_METRICS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hand_vertex_ids(asset, side_joints):
    parents = asset["joint_parents"].astype(int)
    hand = set(side_joints)
    for j in range(len(parents)):
        p = parents[j]
        while p >= 0:
            if p in side_joints:
                hand.add(j)
                break
            p = parents[p]
    w = asset["skin_dense"][:, sorted(hand)].sum(1)
    return np.flatnonzero(w > 0.5)


def build_body(kit, mhr_path, roles):
    asset = asset_from_torchscript(mhr_path)
    nv = asset["skin_dense"].shape[0]
    # wrists: 42 = r_wrist, 78 = l_wrist (MHR joint ids); everything below them is hand
    left, right = hand_vertex_ids(asset, [78]), hand_vertex_ids(asset, [42])
    pick = lambda ids, n: ids[np.linspace(0, len(ids) - 1, n).round().astype(int)]  # noqa: E731
    everything = np.arange(nv)
    if roles == "all":
        align = body = everything
    elif roles == "nohand":
        align = body = np.setdiff1d(everything, np.union1d(left, right))
    elif roles == "sub":
        align, body = everything, np.flatnonzero(np.load(SUBSET))
    else:
        raise ValueError(roles)
    asset.update(role_alignment=align, role_body=body, role_left_hand=pick(left, 256), role_right_hand=pick(right, 256))
    return kit._MHRBody(asset)


def load_mesh(d, ep):
    import trimesh
    for ext in (".ply", ".obj", ".glb", ".off", ".stl"):
        p = os.path.join(d, f"episode_{ep:06d}_object{ext}")
        if os.path.exists(p):
            m = trimesh.load(p, force="mesh", process=False)
            return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    return None, None


def episode_arrays(kit, body, d, ep, frames):
    """The scorer's _to_frames for one side, from an NPZ instead of long rows."""
    z = np.load(os.path.join(d, f"episode_{ep:06d}.npz"))
    pose = np.asarray(z["pose"], np.float64)
    if pose.shape[0] <= frames[-1]:
        raise ValueError(f"{d} episode {ep}: {pose.shape[0]} pose rows, scored frames reach {frames[-1]}")
    sc, sh = np.asarray(z["scales"], np.float64), np.asarray(z["shape"], np.float64)
    sc = np.median(sc, 0) if sc.ndim == 2 else sc
    sh = np.median(sh, 0) if sh.ndim == 2 else sh
    pose = pose[frames]
    params = np.concatenate([pose, np.broadcast_to(sc, (len(pose), 68))], 1)
    if TORCH_FORWARD is not None:  # official TorchScript MHR in float64 (same body as the kit forward to ~1e-9 m)
        vertices, joints = TORCH_FORWARD(pose, sc, sh)
    else:
        vertices, joints = body(params, sh)
    T = len(frames)
    if "object_rotation" in z.files:
        R = kit._rotations(np.asarray(z["object_rotation"], np.float64)[frames])
        t = np.asarray(z["object_translation"], np.float64)[frames]
        s = float(z["object_scale"])
    else:
        R, t, s = np.repeat(np.eye(3)[None], T, 0), np.zeros((T, 3)), 1.0
    mv, mf = load_mesh(d, ep)
    if mv is None:  # no object: a 1 cm cube at the origin keeps the arrays well formed
        import trimesh
        b = trimesh.creation.box((0.01, 0.01, 0.01))
        mv, mf = np.asarray(b.vertices, np.float64), np.asarray(b.faces, np.int64)
        has_obj = False
    else:
        has_obj = True
    local = kit._sample_mesh_surface(mv, mf, len(body.roles["body"]), ep)
    arrays = {
        "mhr_vertices": vertices[:, body.roles["alignment"]],
        "mhr_joints": joints[:, list(kit.MHR_TABLE3_BODY_JOINT_INDICES)],
        "human_surface_points": vertices[:, body.roles["body"]],
        "object_surface_points": s * np.einsum("tij,pj->tpi", R, local) + t[:, None],
        "object_translation": t,
    }
    scene = {"hands": vertices[:, np.concatenate([body.roles["left_hand"], body.roles["right_hand"]])],
             "mesh_vertices": mv, "mesh_faces": mf, "rotation": R, "translation": t, "scale": s}
    return arrays, scene, has_obj


def episode_metrics(kit, pred, target, frames, device, chunk, human_only=False):
    """kit._episode_metrics on torch tensors (same functions, same order of operations). human_only skips the two
    object Chamfers (cd_o, cd_c; they come back NaN): CD-H is then kit.symmetric_chamfer_distance on the same
    aligned human points that mhr_table3_metric_series would use."""
    tp = {k: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float64, device=device) for k, v in pred.items()}
    tt = {k: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float64, device=device) for k, v in target.items()}
    tr = kit.fit_similarity_transform(tp["mhr_vertices"][0], tt["mhr_vertices"][0])
    if human_only:
        ph = kit.apply_similarity_transform(tp["human_surface_points"], tr)
        cdh = kit.symmetric_chamfer_distance(ph, tt["human_surface_points"], chunk_size=chunk) * 100.0
        nan = torch.full_like(cdh, float("nan"))
        series = {"cd_h_cm": cdh, "cd_o_cm": nan, "cd_c_cm": nan}
    else:
        series = kit.mhr_table3_metric_series(tp, tt, body_joint_indices=None, chamfer_chunk_size=chunk)
    pj = kit.apply_similarity_transform(tp["mhr_joints"], tr)
    pt = kit.apply_similarity_transform(tp["object_translation"], tr)
    acc_h, acc_o = [], []
    for part in kit._stretches(np.asarray(frames)):
        if part.stop - part.start < 3:
            raise ValueError(f"a scored stretch of {part.stop - part.start} frames is too short for acceleration")
        acc_h.append(kit.acceleration_error_per_frame(pj[part], tt["mhr_joints"][part]))
        acc_o.append(kit.acceleration_error_per_frame(pt[part], tt["object_translation"][part]))
    series["acc_h_cm"] = torch.cat(acc_h) * 100.0
    series["acc_o_cm"] = torch.cat(acc_o) * 100.0
    out = {k: float(v.double().mean().cpu()) for k, v in series.items()}
    out["sim3_scale"] = float(tr[0].cpu())
    out["cd_h_per_frame"] = series["cd_h_cm"].cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", default="/work/doreen071/v2d_val/gt")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--scored", default="scored_frames.json",
                    help="scored-frame file in --gt: scored_frames.json (the validation set) or "
                         "scored_frames_framedrop.json (episodes whose Kinect video drops frames)")
    ap.add_argument("--roles", default="all", choices=["all", "nohand", "sub"])
    ap.add_argument("--metrics", nargs="*", default=["cd_h", "acc_h"],
                    help=f"subset of {ALL_METRICS}; any object metric switches on the full Table 3 series (3x slower); "
                         "pen is slow and only meaningful with a real object")
    ap.add_argument("--max-frames", type=int, default=0, help="score only the first N scored frames (quick look)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=4096, help="Chamfer chunk (does not change the result)")
    ap.add_argument("--check-numpy", action="store_true", help="also run kit._episode_metrics in numpy and compare")
    ap.add_argument("--forward", default="kit", choices=["kit", "torch"],
                    help="kit: the scorer's numpy MHR forward (exact, CPU-bound); torch: the official TorchScript "
                         "model in float64 on the GPU (much faster, agrees to ~1e-9 m)")
    ap.add_argument("--mhr", default=DEFAULT_MHR)
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    kit = load_kit("PEN" if "pen" in a.metrics else "CD-H")
    if a.forward == "torch":
        global TORCH_FORWARD
        from mhr_utils import MHRForward
        TORCH_FORWARD = MHRForward(path=a.mhr, device=a.device, dtype=torch.float64)
    body = build_body(kit, a.mhr, a.roles)
    scored = {int(k): np.asarray(v, int) for k, v in json.load(open(os.path.join(a.gt, a.scored))).items()}
    info = json.load(open(os.path.join(a.gt, "gt_info.json"))) if os.path.exists(os.path.join(a.gt, "gt_info.json")) else {}
    eps = a.episodes if a.episodes is not None else sorted(scored)
    rows = {}
    for ep in eps:
        t0 = time.time()
        frames = scored[ep][: a.max_frames] if a.max_frames else scored[ep]
        if not os.path.exists(os.path.join(a.pred, f"episode_{ep:06d}.npz")):
            print(f"episode {ep}: no prediction, skipped", flush=True)
            continue
        if not os.path.exists(os.path.join(a.gt, f"episode_{ep:06d}.npz")):
            print(f"episode {ep}: no ground truth yet, skipped", flush=True)
            continue
        target, _, gt_obj = episode_arrays(kit, body, a.gt, ep, frames)
        pred, scene, pred_obj = episode_arrays(kit, body, a.pred, ep, frames)
        r = episode_metrics(kit, pred, target, frames, a.device, a.chunk,
                            human_only=not ({"cd_o", "cd_c", "acc_o"} & set(a.metrics)))
        cdf = r.pop("cd_h_per_frame")
        r["cd_h_first_frame"] = float(cdf[0])
        # diagnostics in the raw camera frame (no alignment): pelvis depth ratio and pelvis error
        zp, zg = pred["mhr_joints"][:, 0, 2], target["mhr_joints"][:, 0, 2]
        r["z_ratio"] = float(np.median(zp / zg))
        r["pelvis_err_raw_cm"] = float(np.linalg.norm(pred["mhr_joints"][:, 0] - target["mhr_joints"][:, 0], axis=-1).mean() * 100)
        r["n_frames"] = len(frames)
        r["object"] = bool(pred_obj and gt_obj)
        if a.check_numpy:
            ref = kit._episode_metrics(pred, target, np.asarray(frames), None)
            r["numpy_max_abs_diff"] = max(abs(ref[k] - r[k]) for k in ref)
        if "pen" in a.metrics:
            r["interpenetration_cm"] = kit._episode_penetration(pred, scene, target) if pred_obj else float("nan")
        r["seconds"] = round(time.time() - t0, 1)
        r["sequence"] = info.get(str(ep), {}).get("sequence", "")
        rows[ep] = r
        if not a.quiet:
            print(f"episode {ep:2d} {r['sequence']:<34s} CD-H {r['cd_h_cm']:6.2f}  ACC-H {r['acc_h_cm']:.4f}  "
                  f"CD-O {r['cd_o_cm']:6.2f}  s {r['sim3_scale']:.3f}  z {r['z_ratio']:.3f}  "
                  f"({len(frames)} fr, {r['seconds']}s)" + (f"  numpy diff {r['numpy_max_abs_diff']:.2e}"
                                                            if a.check_numpy else ""), flush=True)
    keys = ["cd_h_cm", "acc_h_cm", "cd_o_cm", "acc_o_cm", "cd_c_cm", "sim3_scale", "z_ratio", "pelvis_err_raw_cm"] + (
        ["interpenetration_cm"] if "pen" in a.metrics else [])
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = {k: float(np.nanmean([r[k] for r in rows.values()])) for k in keys} if rows else {}
    print("MEAN " + "  ".join(f"{k} {v:.4f}" for k, v in mean.items()), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"pred": a.pred, "gt": a.gt, "roles": a.roles, "mean": mean,
                   "episodes": {str(k): v for k, v in rows.items()}}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
