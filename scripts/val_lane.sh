#!/bin/bash
# One lane of a validation-set body run with the Track 1 settings: SAM 3D Body (dinov3, its own person detector and
# tracker) with the given intrinsics -> refine_body_v5.py --no-upright -> assembly with sigma_h 3 -> val_score.py.
#   val_lane.sh SETTING INTR_DIR EP [EP ...]
#     SETTING   output tag: root/work/body_SETTING (raw), root/work/body_v5_SETTING (refined),
#               /work/doreen071/v2d_val/runs/SETTING/episodes (assembled, scoreable)
#     INTR_DIR  intrinsics directory name under root/work (intrinsics = MoGe estimate, intrinsics_gt, intrinsics_gt095...)
# Waits for the episode's video, for the shared preprocessing (per-episode markers root/work/_done/<step>_<ep> written
# by val_preproc.sbatch): MoGe intrinsics (only when INTR_DIR=intrinsics), then masks, depth and ViTPose before the
# refinement.
VAL=/work/doreen071/v2d_val
ROOT=$VAL/root
W=$ROOT/work
S=$ROOT/v2d-track1/scripts
P=/home/doreen071/.conda/envs/sam3db/bin/python
PV=/home/doreen071/.conda/envs/v2d/bin/python
SET=$1; INTR=$2; shift 2
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=3
cd $S
V=$ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera
for ep in "$@"; do
  until [ -f $V/episode_$(printf %06d $ep).mp4 ]; do sleep 20; done
  [ "$INTR" = "intrinsics" ] && until [ -f $W/_done/intr_$ep ]; do sleep 20; done
done
$P run_sam3d_body.py --episodes "$@" --skip-existing --intr-dir $W/$INTR --out $W/body_$SET || exit 1
echo "RAW_DONE $(date +%T)"
for ep in "$@"; do
  until [ -f $W/_done/masks_$ep ] && [ -f $W/_done/depth_$ep ] && [ -f $W/_done/vit_$ep ]; do sleep 30; done
done
$P refine_body_v5.py --episodes "$@" --no-upright --body-in $W/body_$SET --out $W/body_v5_$SET || exit 1
echo "REFINE_DONE $(date +%T)"
cd /work/doreen071/v2d/v2d-track1/scripts
$PV val_assemble.py --body $W/body_v5_$SET --out $VAL/runs/$SET/episodes --sigma-h 3 --episodes "$@" || exit 1
echo "LANE_DONE $(date +%T)"
