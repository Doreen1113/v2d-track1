# V2D Challenge — Track 1: 4D Human-Object Reconstruction

Monocular third-person video → 4D human (MHR parameters) and object (mesh + per-frame pose).

## Pipeline (in progress)

1. **Human:** per-frame MHR recovery with SAM 3D Body, temporal smoothing, metric-scale alignment.
2. **Object:** segmentation and tracking with SAM, mesh reconstruction with SAM 3D Objects, per-frame pose by aligning the mesh to the tracked mask and depth.
3. **Interaction:** hand-object contact constraints to reduce penetration.
4. **Packing:** per-episode NPZ + object mesh → official submission kit (`eval_reconstruction.py`).

## Layout

- `scripts/` — reconstruction and packing scripts
- `configs/` — run configurations

Data, model weights and outputs are not tracked (see `.gitignore`).
