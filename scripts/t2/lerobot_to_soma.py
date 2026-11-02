#!/usr/bin/env python3
"""Convert a Track 2 LeRobot tier dataset into per-sequence SOMA retargeting folders.

For every episode this writes the layout consumed by
``robotic_grounding/scripts/retarget/soma_to_g1.py``::

    <out_root>/whole_body/soma/sequence_id=<sequence_id>/
        soma_params.npz                 SOMA-X export schema (77 joints, mhr identity)
        poses.npy                       (T, 4, 4) world_T_object in the source OpenCV frame
        ground_plane.json               copied fitted plane
        reconstructed_mesh/output_aligned.glb
    <out_root>/whole_body/reconstructed_stage/<sequence_id>_support.usda

Frames where the object is not visible carry an all-zero pose in the release; those are
filled by linear interpolation (slerp for rotation) between the nearest visible frames so
the retargeter receives a dense trajectory. Scoring only uses visible frames.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R, Slerp


def fill_object_poses(pose_wxyz: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Return (T, 4, 4) matrices with invisible frames interpolated."""
    T = pose_wxyz.shape[0]
    idx = np.flatnonzero(visible)
    if idx.size == 0:
        raise ValueError("episode has no visible object frame")
    pos = pose_wxyz[:, :3].astype(np.float64)
    quat_wxyz = pose_wxyz[:, 3:7].astype(np.float64)
    rot = R.from_quat(quat_wxyz[idx], scalar_first=True)
    t = np.arange(T, dtype=np.float64)
    out_pos = np.stack([np.interp(t, idx, pos[idx, k]) for k in range(3)], axis=1)
    if idx.size == 1:
        out_rot = R.from_quat(np.repeat(quat_wxyz[idx], T, axis=0), scalar_first=True)
    else:
        slerp = Slerp(idx.astype(np.float64), rot)
        tc = np.clip(t, idx[0], idx[-1])
        out_rot = slerp(tc)
    mats = np.tile(np.eye(4), (T, 1, 1))
    mats[:, :3, :3] = out_rot.as_matrix()
    mats[:, :3, 3] = out_pos
    return mats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, type=Path, help="tier dataset root (data/, meta/, mesh/)")
    ap.add_argument("--out-root", required=True, type=Path, help="motion root (HUMAN_MOTION_DATA_DIR)")
    ap.add_argument("--joint-orient-npz", required=True, type=Path,
                    help="reference soma_params.npz supplying joint_orient (78,3,3) for the same rig")
    ap.add_argument("--episodes", type=int, nargs="*", help="subset of episode indices")
    args = ap.parse_args()

    ref = np.load(args.joint_orient_npz, allow_pickle=True)
    joint_orient = ref["joint_orient"]
    ref_joint_names = [str(n) for n in ref["joint_names"]]

    info = json.loads((args.dataset / "meta/info.json").read_text())
    pose_names = info["features"]["observation.human.pose"]["names"]
    joint_names = [n[:-3] for n in pose_names[::3]]
    assert all(n.endswith((".rx", ".ry", ".rz")) for n in pose_names)
    if joint_names != ref_joint_names:
        raise SystemExit("joint order differs from the reference rig")

    meta = [json.loads(line) for line in (args.dataset / "meta/episodes_metadata.jsonl").read_text().splitlines() if line.strip()]
    soma_root = args.out_root / "whole_body" / "soma"
    stage_root = args.out_root / "whole_body" / "reconstructed_stage"
    stage_root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for ep in meta:
        e = int(ep["episode_index"])
        if args.episodes and e not in args.episodes:
            continue
        seq = ep["sequence_id"]
        table = pq.read_table(args.dataset / f"data/chunk-{e // info['chunks_size']:03d}/episode_{e:06d}.parquet")
        frame_index = table["frame_index"].to_numpy()
        assert np.array_equal(frame_index, np.arange(len(frame_index))), "non-contiguous frame index"
        pose = np.stack(table["observation.human.pose"].to_numpy(zero_copy_only=False)).astype(np.float32)
        transl = np.stack(table["observation.human.translation"].to_numpy(zero_copy_only=False)).astype(np.float32)
        identity = np.stack(table["observation.human.identity_coeffs"].to_numpy(zero_copy_only=False)).astype(np.float32)
        scale = np.stack(table["observation.human.scale_params"].to_numpy(zero_copy_only=False)).astype(np.float32)
        obj = np.stack(table["observation.object.pose"].to_numpy(zero_copy_only=False)).astype(np.float64)
        visible = table["observation.object.visible"].to_numpy(zero_copy_only=False).astype(bool)
        T = pose.shape[0]
        seq_dir = soma_root / f"sequence_id={seq}"
        (seq_dir / "reconstructed_mesh").mkdir(parents=True, exist_ok=True)
        np.savez(
            seq_dir / "soma_params.npz",
            allow_pickle=np.array(False),
            poses=pose.reshape(T, len(joint_names), 3),
            transl=transl,
            joint_names=np.array(joint_names),
            identity_model_type=np.array("mhr"),
            identity_coeffs=identity,
            keep_root=np.array(False),
            absolute_pose=np.array(False),
            rotation_repr=np.array("rotvec"),
            unit=np.array("meters"),
            scale_params=scale,
            joint_orient=joint_orient.astype(np.float32),
        )
        np.save(seq_dir / "poses.npy", fill_object_poses(obj, visible))
        shutil.copyfile(args.dataset / ep["ground_plane"], seq_dir / "ground_plane.json")
        shutil.copyfile(args.dataset / ep["mesh"], seq_dir / "reconstructed_mesh" / "output_aligned.glb")
        shutil.copyfile(args.dataset / ep["support_surface"], stage_root / f"{seq}_support.usda")
        manifest[e] = {"sequence_id": seq, "object": ep["object"], "frames": T,
                       "visible_frames": int(visible.sum()), "task": ep["task"]}
        print(f"episode {e:3d} {seq} T={T} visible={int(visible.sum())}")
    (args.out_root / "episodes.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} sequences under {soma_root}")


if __name__ == "__main__":
    main()
