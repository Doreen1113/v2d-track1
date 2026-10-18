#!/bin/bash
# Proxy metrics of a body directory (eval_body_proxies.py), 3 array tasks of 10 episodes.
#   sbatch slurm_eval_body.sbatch <body_dir> <out_dir> [--ref <dir>]
#SBATCH --job-name=eval_body
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/work/doreen071/v2d/jobs/eval_body_%A_%a.log
PY=/home/doreen071/.conda/envs/sam3do/bin/python
i=$SLURM_ARRAY_TASK_ID
EPS=$(seq $i 3 29 | tr '\n' ' ')
$PY /work/doreen071/v2d/v2d-track1/scripts/eval_body_proxies.py --body "$1" --out "$2" --episodes $EPS "${@:3}"
