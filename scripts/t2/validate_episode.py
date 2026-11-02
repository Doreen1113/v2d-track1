#!/usr/bin/env python3
"""Check one evaluator trajectory parquet against the clean Track 2 reference grid.

Mirrors the packer's per-episode checks (world/frame/object grid, evaluator reference poses,
provenance hashes) so a single finished episode can be validated before all 30 exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

KIT = Path("/work/doreen071/v2d/kit/v2d_submission_kit")
sys.path.insert(0, str(KIT))

from v2dlb.submission_parquet import provenance, read_trajectory  # noqa: E402
from v2dlb.trajectory import episode_parquet, read_episode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parquet", type=Path)
    ap.add_argument("--reference", type=Path,
                    default=Path("/work/doreen071/v2d/data_t2/track_2/tier_1_multiview_caption"))
    a = ap.parse_args()
    t = read_trajectory(a.parquet)
    ep = t.episode_index
    ref = read_episode(episode_parquet(a.reference, ep), ep)
    worlds = np.unique(t.world_index)
    frames = np.unique(t.frame_index)
    print(f"episode {ep}: rows={len(t.frame_index)} worlds={worlds.tolist()} frames {frames.min()}..{frames.max()} "
          f"({frames.size}) slots={np.unique(t.object_slot).tolist()} valid={bool(t.world_valid.all())}")
    print(f"reference: frames {ref.frame_index.min()}..{ref.frame_index.max()} ({ref.frame_index.size}) objects={ref.object_names}")
    missing = np.setdiff1d(ref.frame_index, frames)
    print(f"missing reference frames: {missing.size}")
    print(f"has evaluator reference poses: {t.reference_xyzw is not None}")
    print(f"object_name column: {None if t.object_name is None else np.unique(np.asarray(t.object_name, dtype=object)).tolist()}")
    prov = provenance(t)
    print("provenance:", {k: (v[:16] + '...') if isinstance(v, str) and len(v) > 20 else v for k, v in prov.items()})
    if t.reference_xyzw is not None:
        sel = np.isin(t.frame_index, ref.frame_index) & (t.world_index == worlds[0])
        order = np.argsort(t.frame_index[sel])
        achieved = t.pose_xyzw[sel][order]
        reference = t.reference_xyzw[sel][order]
        err = np.linalg.norm(achieved[:, :3] - reference[:, :3], axis=1)
        print(f"rollout vs evaluator reference position error: mean {err.mean()*100:.2f} cm, max {err.max()*100:.2f} cm")
        # The released reference lives in a different world frame; a rigid transform aligns them.
        rel = ref.pose_xyzw[:, 0, :3]
        d_ref = np.linalg.norm(np.diff(rel, axis=0), axis=1)
        d_eval = np.linalg.norm(np.diff(reference[:, :3], axis=0), axis=1)
        n = min(d_ref.size, d_eval.size)
        print(f"frame-to-frame path length agreement (released vs evaluator reference): "
              f"{np.abs(d_ref[:n] - d_eval[:n]).max()*100:.3f} cm max diff")


if __name__ == "__main__":
    main()
