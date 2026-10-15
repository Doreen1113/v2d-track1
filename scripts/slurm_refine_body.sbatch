#!/bin/bash
# Whole-sequence body refinement (refine_body_sequence.py), 6 single-GPU array tasks.
#   sbatch /work/doreen071/v2d/v2d-track1/scripts/slurm_refine_body.sbatch [extra args]
#SBATCH --job-name=refine_body
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/work/doreen071/v2d/jobs/refine_body_%A_%a.log
module load cuda/12.6 2>/dev/null || true
PY=/home/doreen071/.conda/envs/sam3db/bin/python
hostname
cd /work/doreen071/v2d/v2d-track1/scripts
$PY refine_body_sequence.py --all --shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} "$@"
