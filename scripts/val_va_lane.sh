#!/bin/bash
# One episode of the V-A body on the validation set (the A1 Kaggle candidate's body):
#   member A = SAM3DB dinov3 with the MoGe focal (root/work/body_moge, from val_lane.sh moge intrinsics EP)
#   member B = SAM3DB vith, body decoder only, on A's person boxes, all frames (root/work/body_vith_moge)
#   fuse_body_ensemble.py (defaults, --window -1: every frame is scored) -> refine_body_v5.py --no-upright
#   -> val_assemble.py sigma_h 3 -> /work/doreen071/v2d_val/runs/va_moge/episodes
#   val_va_lane.sh EP
VAL=/work/doreen071/v2d_val; W=$VAL/root/work; S=$VAL/root/v2d-track1/scripts
P=/home/doreen071/.conda/envs/sam3db/bin/python; PV=/home/doreen071/.conda/envs/v2d/bin/python
EP=$1
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cd $S
$P run_sam3d_body.py --episodes $EP --hf-repo facebook/sam-3d-body-vith --body-only --batch 16 \
  --boxes-from $W/body_moge --intr-dir $W/intrinsics --out $W/body_vith_moge --skip-existing || exit 1
echo "VITH_DONE $(date +%T)"
$P fuse_body_ensemble.py --a $W/body_moge --b $W/body_vith_moge --out $W/body_ens_moge --episodes $EP --window -1 || exit 1
echo "FUSE_DONE $(date +%T)"
$P refine_body_v5.py --episodes $EP --no-upright --body-in $W/body_ens_moge --out $W/body_v5_ens_moge || exit 1
echo "REFINE_DONE $(date +%T)"
cd /work/doreen071/v2d/v2d-track1/scripts
$PV val_assemble.py --body $W/body_v5_ens_moge --out $VAL/runs/va_moge/episodes --sigma-h 3 --episodes $EP || exit 1
echo "LANE_DONE $(date +%T)"
