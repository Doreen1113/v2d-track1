#!/bin/bash
# Body refinement v3 (refine_body_v3.py) as single-GPU array tasks; each task takes shard i/N of the episodes.
#   sbatch --array=0-5 slurm_refine_body_v3.sbatch --all --out /work/doreen071/v2d/work/body_v3 [extra args]
#SBATCH --job-name=refine_v3
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=/work/doreen071/v2d/jobs/refine_v3_%A_%a.log
PY=/home/doreen071/.conda/envs/sam3do/bin/python
hostname
cd /work/doreen071/v2d/v2d-track1/scripts
$PY refine_body_v3.py --shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} "$@"
