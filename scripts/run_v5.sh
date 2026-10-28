#!/bin/bash
# v5 candidate build: support-calibrated object trajectories (objects_v5) assembled with a body and its
# floor-plane ratio, then the finger-only penetration fix and the proxy evaluation.
#   usage: bash run_v5.sh [stage ...]      stages (default all): track assemble fix eval
#   track     track_object_v5.py from objects_v4 (floor / table calibration + rest terms)   (4 jobs)
#   assemble  assemble_episode.py -> $OUT/episodes_prefix, copied to $OUT/$EPS               (1 job)
#   fix       fix_penetration_fingers.py (finger pose only, no object push)                  (1 job)
#   eval      eval_object_candidate.py (proxy metrics before / after the fix)                 (1 job)
# Variables: BODY (work/body_v3), RATIO (moge_to_body_ratio_floor_b3.json), OBJ (work/objects_v5),
#            OUT (submissions_v5/<name>), SIGMA_O (3), LOCAL=1 runs everything in this shell on the local
#            GPU instead of Slurm (used when the account has no credits).
set -e
ROOT=/work/doreen071/v2d
S=$ROOT/v2d-track1/scripts
W=$ROOT/work
BODY=${BODY:-$W/body_v3}
RATIO=${RATIO:-$W/moge_to_body_ratio_floor_b3.json}
OBJ=${OBJ:-$W/objects_v5}
OUT=${OUT:-$ROOT/submissions_v5/A_body3_obj5}
EPS=${EPS:-episodes}
SIGMA_O=${SIGMA_O:-3}
PY=/home/doreen071/.conda/envs/v2d/bin/python
E3D=/home/doreen071/.conda/envs/sam3do
GROUPS_=("27 15 22 0 5 10 24 26" "16 17 4 1 6 12 23" "9 11 2 7 13 18 25" "3 8 14 19 20 21 28 29")
DEP=""
sub() { sbatch --parsable ${DEP:+--dependency=afterok:$DEP} "$@"; }
STAGES=${@:-track assemble fix eval}
mkdir -p $ROOT/jobs/local
cd $ROOT/jobs
for st in $STAGES; do case $st in
  track)
    if [ -n "$LOCAL" ]; then
      export PATH=$E3D/bin:$PATH CUDA_HOME=$E3D CONDA_PREFIX=$E3D
      # one process, all episodes in sequence (the login node has a 20 GB per-user memory limit)
      (cd $S && nice $E3D/bin/python track_object_v5.py --init $W/objects_v4 --out $OBJ --sdf $W/objects_v4/sdf \
        --body $BODY --ratio $RATIO --episodes ${EPISODES:-$(seq -s ' ' 0 29)} > $ROOT/jobs/local/objv5_$(basename $OBJ).log 2>&1)
    else
      ids=""
      for g in "${GROUPS_[@]}"; do
        id=$(OBJ_OUT=$OBJ sub -J objv5 $S/slurm/objects_v5.sbatch "$g" --body $BODY --ratio $RATIO); ids=${ids:+$ids:}$id
      done
      DEP=$ids
    fi ;;
  assemble)
    CMD="cd $S && $PY assemble_episode.py --body $BODY --sigma-h 0 --objects $OBJ --sigma-o $SIGMA_O --ratio $RATIO \
      --out $OUT/episodes_prefix && mkdir -p $OUT/$EPS && cp $OUT/episodes_prefix/* $OUT/$EPS/"
    if [ -n "$LOCAL" ]; then bash -c "$CMD"; else
      DEP=$(sub -J asm5 --wrap="$CMD" --account=GOV114009 --partition=dev --cpus-per-task=4 --time=00:30:00 \
        --output=$ROOT/jobs/asm5_%j.log); fi ;;
  fix)
    if [ -n "$LOCAL" ]; then (cd $S && $PY fix_penetration_fingers.py --episodes-dir $OUT/$EPS); else
      DEP=$(sub -J fing5 $S/slurm/v2d_run.sbatch fix_penetration_fingers.py --episodes-dir $OUT/$EPS); fi ;;
  eval)
    if [ -n "$LOCAL" ]; then (cd $S && $PY eval_object_candidate.py --prefix $OUT/episodes_prefix --fixed $OUT/$EPS \
        --out $OUT/eval_$EPS.json); else
      DEP=$(sub -J eval5 $S/slurm/v2d_run.sbatch eval_object_candidate.py --prefix $OUT/episodes_prefix \
        --fixed $OUT/$EPS --out $OUT/eval_$EPS.json); fi ;;
esac; done
[ -n "$DEP" ] && echo "last job: $DEP" || true
