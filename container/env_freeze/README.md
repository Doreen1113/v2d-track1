# Reference package lists

`pip freeze` of the five conda environments the N1pc submission was produced with (Python versions in
brackets). They are references for rebuilding the model-stage environments, not install files: the
`-e git+...` lines are the source checkouts listed in `../README.md`, and the CUDA wheels differ per
environment (the cluster mixed cu121 / cu126 / cu128 / cu13 builds).

| file | environment | Python | used for |
|---|---|---|---|
| `v2d.txt` | v2d | 3.11 | assembly, fusion, refinement, composition, NLF trajectory fusion, carving, scorer check, packing (the container's `../requirements.txt` is this list trimmed) |
| `sam3db.txt` | sam3db | 3.11 | SAM 3D Body (two checkpoints), ViTDet person tracking, NLF inference, body refinement runs |
| `sam3.txt` | sam3 | 3.12 | SAM 3 / SAM 2.1 / Grounding DINO masks, MoGe-2 intrinsics and depth, object frame preparation |
| `sam3do.txt` | sam3do | 3.11 | SAM 3D Objects reconstruction, PyTorch3D render-and-compare object tracking (v2, v4, v5) |
| `gvhmr.txt` | gvhmr | 3.10 | GVHMR preprocessing only (ViTPose-H 2D keypoints used by the body refinement) |
