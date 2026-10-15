#!/bin/bash
# SAM 3D Objects mesh candidates for the shared per-class meshes (5 array tasks, episodes sharded):
# MoGe-2 point maps in the sam3 env, then SAM 3D Objects in the sam3do env.
#SBATCH --job-name=objcand
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=/work/doreen071/v2d/jobs/objcand_%A_%a.log
S=/work/doreen071/v2d/v2d-track1/scripts/object_candidates.py
SH=$SLURM_ARRAY_TASK_ID/5
/home/doreen071/.conda/envs/sam3/bin/python $S --stage prep --shard $SH "$@" || exit 1
E3D=/home/doreen071/.conda/envs/sam3do
export PATH=$E3D/bin:$PATH CUDA_HOME=$E3D CONDA_PREFIX=$E3D LIDRA_SKIP_INIT=1
$E3D/bin/python $S --stage recon --shard $SH "$@"
