#!/bin/bash
# GVHMR -> MHR fallback body pipeline over all Track 1 episodes, 8 single-GPU array tasks.
# Requires person boxes from slurm_person_bbox.sbatch. Stages default to all three; e.g.
#   sbatch slurm_gvhmr.sbatch --stages preproc
#   sbatch slurm_gvhmr.sbatch --stages predict convert
#SBATCH --job-name=gvhmr
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00
#SBATCH --array=0-7
#SBATCH --output=/work/doreen071/v2d/jobs/gvhmr_%A_%a.log
module load cuda/12.6 2>/dev/null || true
PY=/home/doreen071/.conda/envs/gvhmr/bin/python
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
cd /work/doreen071/v2d/v2d-track1/scripts
$PY run_gvhmr.py --all --shard ${SLURM_ARRAY_TASK_ID}/8 "$@"
