#!/bin/bash
# Object v5 for a list of episodes: support-calibrated joint fit (track_object_v5.py) on top of the v4
# tracks; the budgeted meshes + SDF grids of objects_v4/sdf are reused (same meshes).
#   sbatch objects_v5.sbatch "<episodes>" [extra track_object_v5.py args]
#SBATCH --job-name=objv5
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=25a-hgpn062
#SBATCH --output=/work/doreen071/v2d/jobs/objv5_%j.log
set -e
W=/work/doreen071/v2d/work
INIT=${INIT:-$W/objects_v4}
OBJ_OUT=${OBJ_OUT:-$W/objects_v5}
cd /work/doreen071/v2d/v2d-track1/scripts
E3D=/home/doreen071/.conda/envs/sam3do
export PATH=$E3D/bin:$PATH CUDA_HOME=$E3D CONDA_PREFIX=$E3D
$E3D/bin/python track_object_v5.py --init $INIT --out $OBJ_OUT --sdf $W/objects_v4/sdf --episodes $1 "${@:2}"
