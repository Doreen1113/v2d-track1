#!/bin/bash
# One lane of the NLF body on the validation set: NLF (val_nlf_infer.py, sam3db env, boxes from root/work/body_moge,
# MoGe intrinsics) -> SMPL -> MHR (val_nlf_to_mhr.py, gvhmr env, as the GT conversion) -> val_assemble.py sigma_h 3
# -> /work/doreen071/v2d_val/runs/nlf/episodes
#   val_nlf_lane.sh EP [EP ...]
VAL=/work/doreen071/v2d_val; N=$VAL/nlf
PS=/home/doreen071/.conda/envs/sam3db/bin/python; PG=/home/doreen071/.conda/envs/gvhmr/bin/python
PV=/home/doreen071/.conda/envs/v2d/bin/python
cd /work/doreen071/v2d/v2d-track1/scripts
$PS -u val_nlf_infer.py --episodes "$@" --out $N/raw --skip-existing || exit 1
echo "INFER_DONE $(date +%T)"
for EP in "$@"; do
  $PG -u val_nlf_to_mhr.py --raw $N/raw --out $N/body_nlf --episodes $EP --skip-existing || exit 1
  echo "MHR_DONE $EP $(date +%T)"
  $PV val_assemble.py --body $N/body_nlf --out $VAL/runs/nlf/episodes --sigma-h 3 --episodes $EP || exit 1
  echo "LANE_DONE $EP $(date +%T)"
done
