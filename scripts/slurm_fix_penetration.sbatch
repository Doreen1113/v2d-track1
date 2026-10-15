#!/bin/bash
# Hand-object penetration fix over assembled episodes, 10 array tasks of 3 episodes each.
#   sbatch slurm_fix_penetration.sbatch <episodes_dir> [extra fix_penetration.py args]
#SBATCH --job-name=fixpen
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=03:55:00
#SBATCH --array=0-9
#SBATCH --output=/work/doreen071/v2d/jobs/fixpen_%A_%a.log
PY=/home/doreen071/.conda/envs/v2d/bin/python
i=$SLURM_ARRAY_TASK_ID
$PY /work/doreen071/v2d/v2d-track1/scripts/fix_penetration.py --episodes-dir "$1" --episodes $i $((i+10)) $((i+20)) "${@:2}"
