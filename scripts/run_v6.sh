#!/bin/bash
# v6 candidates: body v2 (per-episode identity) with extra temporal smoothing of the MHR pose and of the
# object trajectory, then the finger-only penetration fix.
#   usage: bash run_v6.sh NAME OBJECTS_DIR [SIGMA_H] [SIGMA_O]
#   V1: bash run_v6.sh V1 /work/doreen071/v2d/work/objects_v4 3 6
#   V2: bash run_v6.sh V2 /work/doreen071/v2d/work/objects_v5 3 6
set -e
S=/work/doreen071/v2d/v2d-track1/scripts
W=/work/doreen071/v2d/work
O=${OUT_ROOT:-/work/doreen071/v2d/submissions_v6}
PY=/home/doreen071/.conda/envs/v2d/bin/python
NAME=$1; OBJ=$2; SH=${3:-3}; SO=${4:-6}
cd $S
$PY assemble_episode.py --body $W/body_v2 --sigma-h $SH --objects $OBJ --sigma-o $SO \
  --ratio $W/moge_to_body_ratio_floor.json --out $O/$NAME/episodes_prefix
rm -rf $O/$NAME/episodes && cp -r $O/$NAME/episodes_prefix $O/$NAME/episodes
$PY fix_penetration_fingers.py --episodes-dir $O/$NAME/episodes
