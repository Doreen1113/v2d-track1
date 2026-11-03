# V2D Challenge Track 1: single-view human and object reconstruction

Input: one exocentric RGB video and the object's text prompt. Output: the person as MHR parameters, the
object mesh and the object pose per frame, all in the camera frame in metres. No model is trained or
fine-tuned; the scripts run released checkpoints and then fit the results together over the whole sequence.

Final submission: **N1pw03**, packed at commit `f2f17f7`.

| CD-H | CD-O | ACC-H | ACC-O | PEN |
|---|---|---|---|---|
| 14.537 | 22.276 | 0.1375 | 0.0731 | 0 |

## Pipeline of the final entry

| Step | Scripts | What it does |
|---|---|---|
| Camera | `estimate_intrinsics.py` | MoGe-2 focal on 12 frames per episode, one pooled focal (807.7 px) for all cameras |
| Masks | `segment_objects.py`, `segment_objects_sam3.py`, `merge_masks.py` | Grounding DINO + SAM 2.1 and SAM 3 text-prompted tracks, merged per frame |
| Depth | `compute_depth.py` | MoGe-2 metric depth with the pooled focal |
| Person | `track_person_bbox.py`, `run_sam3d_body.py` | ViTDet main-actor track, SAM 3D Body per frame (DINOv3 and ViT-H checkpoints) |
| Body fusion | `fuse_body_ensemble.py` | the two SAM 3D Body runs combined per body part where they agree in 2D (ViTPose-H) and 3D |
| Body refinement | `refine_body_v5.py` | shared shape and bone scales per episode, 2D keypoint fit, acceleration penalty, static floor plane |
| Scale | `make_floor_ratio.py` | MoGe-to-body scale from the floor plane and foot contacts |
| Objects | `reconstruct_object.py`, `object_candidates.py`, `select_class_mesh.py`, `track_object_v2.py`, `track_object_v4.py`, `track_object_v5.py`, `compose_objects_v6.py` | SAM 3D Objects meshes (one per class, chosen by silhouette IoU), whole-sequence render-and-compare pose fit, hand-aware tracks on held frames |
| Assembly | `assemble_episode.py` | body smoothing sigma 3 frames, object smoothing sigma 12 frames |
| Contact carving | `carve_object_contacts.py`, `check_pen_exact.py` | CSG edit of the 4096-face object mesh around penetrating hand points, checked with the scorer's inside test |
| Depth fusion | `val_nlf_infer.py`, `val_nlf_traj.py --mode geolp0.3_5 --with-object` | pelvis depth = weighted geometric mean of the MHR and NLF depths (weight 0.3, low-pass 5 frames); the object is shifted with the body |
| Packing | kit `eval_reconstruction.py` | official submission kit |

The full command sequence, with the exact arguments, is in [`container/reproduce.sh`](container/reproduce.sh).
Model environments, checkpoints and licences are listed in [`container/README.md`](container/README.md).

## Layout

- `scripts/`: reconstruction, validation and packing scripts (`val_*` are the local BEHAVE validation tools)
- `container/`: Dockerfile, Apptainer definition, pinned requirements, environment freezes, reproduction script

Data, model weights and outputs are not tracked.
