#!/bin/bash
# Object v4 for a list of episodes: budgeted mesh + SDF grid (v2d env), then the joint
# hand-aware trajectory fit (sam3do env).
#   sbatch objects_v4.sbatch "<episodes>" [extra track_object_v4.py args]
#SBATCH --job-name=objv4
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/doreen071/v2d/jobs/objv4_%j.log
set -e
W=/work/doreen071/v2d/work
INIT=${INIT:-$W/objects_v3}
OBJ_OUT=${OBJ_OUT:-$W/objects_v4}
cd /work/doreen071/v2d/v2d-track1/scripts
items=""
for e in $1; do items="$items $INIT/$(printf episode_%06d_object.ply $e):$e"; done
/home/doreen071/.conda/envs/v2d/bin/python contactlib.py $OBJ_OUT/sdf $items
E3D=/home/doreen071/.conda/envs/sam3do
export PATH=$E3D/bin:$PATH CUDA_HOME=$E3D CONDA_PREFIX=$E3D
$E3D/bin/python track_object_v4.py --init $INIT --out $OBJ_OUT --sdf $OBJ_OUT/sdf --episodes $1 "${@:2}"
