#!/bin/bash
# v3 submission build: body v2 + objects v2 tracks re-finalised with the floor-plane scale ratio.
set -e
cd /work/doreen071/v2d/v2d-track1/scripts
W=/work/doreen071/v2d/work
OUT=/work/doreen071/v2d/submissions/v3/episodes
/home/doreen071/.conda/envs/v2d/bin/python make_floor_ratio.py
mkdir -p $W/objects_v3 && ln -sfn $W/objects_v2/track $W/objects_v3/track
OBJ_V2=$W/objects_v3 CONDA_PREFIX=/home/doreen071/.conda/envs/sam3do \
  /home/doreen071/.conda/envs/sam3do/bin/python finalize_objects_v2.py --ratio $W/moge_to_body_ratio_floor.json
/home/doreen071/.conda/envs/v2d/bin/python assemble_episode.py --body $W/body_v2 --sigma-h 0 \
  --objects $W/objects_v3 --sigma-o 2 --ratio $W/moge_to_body_ratio_floor.json --out $OUT
sbatch --wait slurm_fix_penetration.sbatch $OUT --sigma 6
