#!/bin/bash
# Run one script in the v2d env on one GPU:  sbatch v2d_run.sbatch <script.py> [args...]
#SBATCH --job-name=v2drun
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/doreen071/v2d/jobs/v2drun_%j.log
cd /work/doreen071/v2d/v2d-track1/scripts
/home/doreen071/.conda/envs/v2d/bin/python "$@"
