#!/bin/bash
# Run a list of training tasks on one node, round-robin over the visible GPUs with
# PER_GPU concurrent tasks per GPU.
# Usage: sbatch --gres=gpu:8 ... train_node.sh <tasks_file> <hmd_root> <run_root>
# tasks_file lines: <episode_index> <sequence_id> <seed> <total_env_steps>
set -uo pipefail
TASKS=$1; HMD=$2; RUN_ROOT=$3
PER_GPU=${PER_GPU:-1}
ulimit -u 65536 2>/dev/null || true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
SCRIPT_DIR=/work/doreen071/v2d/v2d-track1/scripts/t2
NGPU=$(nvidia-smi -L | grep -c '^GPU')
SLOTS=$((NGPU * PER_GPU))
mkdir -p "$RUN_ROOT"
i=0
mapfile -t LINES < <(grep -v '^\s*$' "$TASKS")
for line in "${LINES[@]}"; do
  read -r EP SEQ SEED STEPS <<< "$line"
  gpu=$((i % NGPU))
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    bash "$SCRIPT_DIR/train_task.sh" "$EP" "$SEQ" "$SEED" "$STEPS" "$HMD" "$RUN_ROOT"
  ) > "$RUN_ROOT/task_ep${EP}_s${SEED}.out" 2>&1 &
  i=$((i + 1))
  if [ $((i % SLOTS)) -eq 0 ]; then wait -n; fi
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 20; done
done
wait
echo NODE_DONE
