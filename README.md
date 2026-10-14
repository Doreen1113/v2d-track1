# V2D Challenge — Track 1: 4D Human-Object Reconstruction

Monocular third-person video → 4D human (MHR parameters) and object (mesh + per-frame pose).

## Pipeline

All estimates live in one OpenCV camera frame (metres) with a shared pinhole focal length.

| Step | Script | Notes |
|---|---|---|
| Intrinsics | `estimate_intrinsics.py` | MoGe-2 on 12 frames per episode; global median focal (all four cameras share a lens) |
| Human | `run_sam3d_body.py`, `verify_body.py` | ViTDet main-actor tracking + SAM 3D Body per frame → MHR `pose/scales/shape`; checked against the official MHR forward |
| Object masks | `segment_objects.py`, `segment_objects_sam3.py`, `merge_masks.py` | Grounding DINO + SAM 2.1 and SAM 3 text-prompted tracking, merged per frame |
| Depth | `compute_depth.py`, `prepare_object_frames.py` | MoGe-2 metric depth with the fixed focal |
| Object mesh | `reconstruct_object.py` | SAM 3D Objects on the cleanest frame, pose refined to the mask and depth |
| Object pose | `track_object_pose.py` | per-frame render-and-compare (silhouette + depth), interpolation and smoothing |
| Assembly | `assemble_episode.py` | constant shape/scales (median), smoothed pose, object rescaled from the depth frame to the body frame |
| Contact | `fix_penetration.py` | moves the object out of the hands using the scorer's own penetration test |
| Packing | kit `eval_reconstruction.py` | official submission kit |

## Layout

- `scripts/` — reconstruction and packing scripts
- `configs/` — run configurations

Data, model weights and outputs are not tracked (see `.gitignore`).
