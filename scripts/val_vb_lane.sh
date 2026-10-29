#!/bin/bash
# One episode of a V-B style body run on the validation set: a MoGe relative-depth target for the pelvis, built from
# a refined body (compute_moge_depth_target.py, or val_depth_target.py when TARGET_SCRIPT is set), then
# refine_body_v5.py --no-upright --depth-target on the per-frame input, then assembly with sigma_h 3.
#   val_vb_lane.sh TAG BODY_IN REFINED EP [refine args...]
#     TAG      output tag: root/work/depth_target_<TAG>, root/work/body_v5_<TAG>, /work/doreen071/v2d_val/runs/<TAG>
#     BODY_IN  per-frame input under root/work (body_moge, body_ens_moge)
#     REFINED  refined body under root/work the target is built from (body_v5_moge, body_v5_ens_moge)
#   env: TARGET_SCRIPT (default compute_moge_depth_target.py), TARGET_ARGS (extra args for it), TARGET_TAG (reuse the
#        target of another tag)
VAL=/work/doreen071/v2d_val; W=$VAL/root/work; S=$VAL/root/v2d-track1/scripts
E=/home/doreen071/.conda/envs
TAG=$1; BIN=$2; REF=$3; EP=$4; shift 4
TT=${TARGET_TAG:-$TAG}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cd $S
if [ ! -f $W/depth_target_$TT/episode_$(printf %06d $EP).npz ]; then
  if [ -n "$TARGET_SCRIPT" ]; then TS=/work/doreen071/v2d/v2d-track1/scripts/$TARGET_SCRIPT; else TS=compute_moge_depth_target.py; fi
  V2D_ROOT=$VAL/root $E/sam3do/bin/python $TS --body $W/$REF --out $W/depth_target_$TT --episodes $EP $TARGET_ARGS || exit 1
fi
echo "TARGET_DONE $(date +%T)"
$E/sam3db/bin/python refine_body_v5.py --episodes $EP --no-upright --body-in $W/$BIN --depth-target $W/depth_target_$TT \
  --out $W/body_v5_$TAG "$@" || exit 1
echo "REFINE_DONE $(date +%T)"
cd /work/doreen071/v2d/v2d-track1/scripts
$E/v2d/bin/python val_assemble.py --body $W/body_v5_$TAG --out $VAL/runs/$TAG/episodes --sigma-h 3 --episodes $EP || exit 1
echo "LANE_DONE $(date +%T)"
