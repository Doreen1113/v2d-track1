#!/bin/bash
# Run one object-v2 script in the sam3do env (PyTorch3D) on one GPU:
#   sbatch object_v2_run.sbatch <script.py> [args...]
#SBATCH --job-name=objv2
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/doreen071/v2d/jobs/objv2_%j.log
E3D=/home/doreen071/.conda/envs/sam3do
export PATH=$E3D/bin:$PATH CUDA_HOME=$E3D CONDA_PREFIX=$E3D
cd /work/doreen071/v2d/v2d-track1/scripts
$E3D/bin/python "$@"
