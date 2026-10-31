#!/bin/bash
# One lane of the Date03 held-out set (val_nlf_build_d03.py), per episode in order, the same commands as the main set:
#   preprocessing as val_preproc.sbatch (MoGe intrinsics + depth, person boxes + ViTPose, masks)
#   baseline   SAM3DB dinov3 (MoGe intrinsics) -> refine_body_v5 --no-upright -> assemble sigma_h 3  -> runs_d03/moge
#   V-A        + SAM3DB vith body-only on the same boxes -> fuse --window -1 -> refine_v5 -> assemble -> runs_d03/va_moge
#   NLF        val_nlf_infer.py (same boxes / intrinsics) -> val_nlf_to_mhr.py -> assemble        -> runs_d03/nlf
#   val_nlf_d03_lane.sh EP [EP ...]
VAL=/work/doreen071/v2d_val; ROOT=$VAL/root_d03; W=$ROOT/work; S=$ROOT/v2d-track1/scripts; RUNS=$VAL/runs_d03
N=$VAL/nlf/d03; E=/home/doreen071/.conda/envs
P=$E/sam3db/bin/python; PV=$E/v2d/bin/python; PG=$E/gvhmr/bin/python; P3=$E/sam3/bin/python
REPO=/work/doreen071/v2d/v2d-track1/scripts
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=3
mkdir -p $W/_done $N
for EP in "$@"; do
  cd $S
  [ -f $W/_done/intr_$EP ] || { $P3 estimate_intrinsics.py --episodes $EP --out $W/intrinsics && touch $W/_done/intr_$EP; } || exit 1
  [ -f $W/_done/depth_$EP ] || { $P3 compute_depth.py --episodes $EP && touch $W/_done/depth_$EP; } || exit 1
  [ -f $W/_done/vit_$EP ] || { $P track_person_bbox.py --episodes $EP --skip-existing && \
      $PG run_gvhmr.py --episodes $EP --stages preproc && touch $W/_done/vit_$EP; } || exit 1
  [ -f $W/_done/masks_$EP ] || { $P3 segment_objects.py --episodes $EP --skip-existing --out-dir $W/masks && touch $W/_done/masks_$EP; } || exit 1
  echo "PRE_DONE $EP $(date +%T)"
  $P run_sam3d_body.py --episodes $EP --skip-existing --intr-dir $W/intrinsics --out $W/body_moge || exit 1
  $P refine_body_v5.py --episodes $EP --no-upright --body-in $W/body_moge --out $W/body_v5_moge || exit 1
  (cd $REPO && $PV val_assemble.py --body $W/body_v5_moge --out $RUNS/moge/episodes --sigma-h 3 --episodes $EP) || exit 1
  echo "MOGE_DONE $EP $(date +%T)"
  $P run_sam3d_body.py --episodes $EP --hf-repo facebook/sam-3d-body-vith --body-only --batch 16 \
    --boxes-from $W/body_moge --intr-dir $W/intrinsics --out $W/body_vith_moge --skip-existing || exit 1
  $P fuse_body_ensemble.py --a $W/body_moge --b $W/body_vith_moge --out $W/body_ens_moge --episodes $EP --window -1 || exit 1
  $P refine_body_v5.py --episodes $EP --no-upright --body-in $W/body_ens_moge --out $W/body_v5_ens_moge || exit 1
  (cd $REPO && $PV val_assemble.py --body $W/body_v5_ens_moge --out $RUNS/va_moge/episodes --sigma-h 3 --episodes $EP) || exit 1
  echo "VA_DONE $EP $(date +%T)"
  cd $REPO
  $P -u val_nlf_infer.py --episodes $EP --out $N/raw --skip-existing \
    --video-dir $ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera \
    --intr-dir $W/intrinsics --boxes-from $W/body_moge || exit 1
  $PG -u val_nlf_to_mhr.py --raw $N/raw --out $N/body_nlf --episodes $EP --skip-existing || exit 1
  $PV val_assemble.py --body $N/body_nlf --out $RUNS/nlf/episodes --sigma-h 3 --episodes $EP || exit 1
  echo "LANE_DONE $EP $(date +%T)"
done
