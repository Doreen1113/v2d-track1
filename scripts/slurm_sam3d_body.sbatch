#!/bin/bash
# SAM 3D Body over all Track 1 episodes, 8 single-GPU array tasks (each takes every 8th episode).
#   sbatch /work/doreen071/v2d/v2d-track1/scripts/slurm_sam3d_body.sbatch
#SBATCH --job-name=s3db_body
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00
#SBATCH --array=0-7
#SBATCH --output=/work/doreen071/v2d/jobs/s3db_body_%A_%a.log
module load cuda/12.6 2>/dev/null || true
PY=/home/doreen071/.conda/envs/sam3db/bin/python
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
cd /work/doreen071/v2d/v2d-track1/scripts
$PY run_sam3d_body.py --all --shard ${SLURM_ARRAY_TASK_ID}/8 --skip-existing "$@"
