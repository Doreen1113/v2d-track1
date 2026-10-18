#!/bin/bash
# v4 candidate build: hand-aware object trajectories (objects_v4, raw MoGe frame) on top of the v3
# objects, assembled with the body and its floor-plane ratio, then the penetration fixes.
#   usage: bash run_v4.sh [stage ...]      stages (default all): track assemble fix eval
#   track     objects_v4.sbatch: budgeted mesh + SDF grid, track_object_v4.py       (4 jobs)
#   assemble  assemble_episode.py -> $OUT/episodes_prefix, copied to $OUT/$EPS       (1 job)
#   fix       FINGERS=1: fix_penetration_fingers.py first (finger pose only); then
#             fix_penetration.py --sigma 6 (object push)                             (3 jobs)
#   eval      eval_object_candidate.py (proxy metrics, before / after the fix)        (1 job)
# Defaults: BODY=work/body_v3, RATIO=work/moge_to_body_ratio_floor_b3.json, SIGMA_O=3.
set -e
ROOT=/work/doreen071/v2d
S=$ROOT/v2d-track1/scripts
W=$ROOT/work
OUT=${OUT:-$ROOT/submissions/v4_candidate}
EPS=${EPS:-episodes}
BODY=${BODY:-$W/body_v3}
RATIO=${RATIO:-$W/moge_to_body_ratio_floor_b3.json}
SIGMA_O=${SIGMA_O:-3}
PY=/home/doreen071/.conda/envs/v2d/bin/python
GROUPS_=("27 15 22 0 5 10 24 26" "16 17 4 1 6 12 23" "9 11 2 7 13 18 25" "3 8 14 19 20 21 28 29")
THIRDS=("0 1 2 3 4 5 6 7 8 9" "10 11 12 13 14 15 16 17 18 19" "20 21 22 23 24 25 26 27 28 29")
DEP=""
sub() {
  sbatch --parsable ${DEP:+--dependency=afterok:$DEP} "$@"
}
STAGES=${@:-track assemble fix eval}
cd $ROOT/jobs
for st in $STAGES; do case $st in
  track)
    ids=""
    for g in "${GROUPS_[@]}"; do
      id=$(sub -J objv4 $S/slurm/objects_v4.sbatch "$g" --body $BODY --ratio $RATIO); ids=${ids:+$ids:}$id
    done
    DEP=$ids ;;
  assemble)
    DEP=$(sub -J asm4 --wrap="cd $S && $PY assemble_episode.py --body $BODY --sigma-h 0 \
      --objects $W/objects_v4 --sigma-o $SIGMA_O --ratio $RATIO \
      --out $OUT/episodes_prefix && mkdir -p $OUT/$EPS && cp $OUT/episodes_prefix/* $OUT/$EPS/" \
      --account=GOV114009 --partition=dev --cpus-per-task=4 --time=00:30:00 \
      --output=$ROOT/jobs/asm4_%j.log) ;;
  fix)
    if [ -n "$FINGERS" ]; then
      DEP=$(sub -J fing4 $S/slurm/v2d_run.sbatch fix_penetration_fingers.py --episodes-dir $OUT/$EPS)
    fi
    ids=""
    for g in "${THIRDS[@]}"; do
      id=$(sub -J fixpen4 $S/slurm/v2d_run.sbatch fix_penetration.py --episodes-dir $OUT/$EPS --sigma 6 \
        --episodes $g); ids=${ids:+$ids:}$id
    done
    DEP=$ids ;;
  eval)
    DEP=$(sub -J eval4 $S/slurm/v2d_run.sbatch eval_object_candidate.py --prefix $OUT/episodes_prefix \
      --fixed $OUT/$EPS --out $OUT/eval_$EPS.json) ;;
esac; done
echo "last job: $DEP"
