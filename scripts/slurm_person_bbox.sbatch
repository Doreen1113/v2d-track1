#!/bin/bash
# Main-actor person boxes (ViTDet) for all Track 1 episodes, 6 single-GPU array tasks.
#   sbatch /work/doreen071/v2d/v2d-track1/scripts/slurm_person_bbox.sbatch
#SBATCH --job-name=person_bbox
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/work/doreen071/v2d/jobs/person_bbox_%A_%a.log
module load cuda/12.6 2>/dev/null || true
PY=/home/doreen071/.conda/envs/sam3db/bin/python
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
cd /work/doreen071/v2d/v2d-track1/scripts
$PY track_person_bbox.py --all --shard ${SLURM_ARRAY_TASK_ID}/6 --skip-existing "$@"
