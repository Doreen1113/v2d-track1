#!/bin/bash
# Train one FlashSAC G1 recon-body policy for one retargeted Track 2 episode and evaluate it.
# Usage: train_task.sh <episode_index> <sequence_id> <seed> <total_env_steps> <hmd_root> <run_root>
# Writes <run_root>/ep<E>_s<SEED>/{train.log,policy_*.safetensors,results/episode_XXXXXX.{parquet,json}}
set -uo pipefail
EP=$1; SEQ=$2; SEED=$3; STEPS=$4; HMD=$5; RUN_ROOT=$6
KIT=${KIT:-/work/doreen071/v2d/kit/v2d_submission_kit}
FC=${FC:-/work/doreen071/v2d/t2/video_to_data/robotic_grounding/flash_chord}
export PATH=/home/doreen071/.conda/envs/t2cu12/bin:$PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
PARQUET="$HMD/whole_body/soma/sequence_id=$SEQ/robot_name=g1/data.parquet"
OUT="$RUN_ROOT/ep$(printf %03d $EP)_s$SEED"
mkdir -p "$OUT"
if [ ! -f "$PARQUET" ]; then echo "missing retargeted parquet $PARQUET" | tee -a "$OUT/train.log"; exit 1; fi
# Curriculum thresholds are absolute environment steps for the 350M recipe; keep the same fractions.
THR=$(python - "$STEPS" <<'PY'
import sys
total=int(sys.argv[1]); base=349999104
fr=[35000320,70000640,104998912,139999232,174999552,209999872,288749568,306249728,349999104]
print("[" + ",".join(str(int(round(t/base*total))) for t in fr) + "]")
PY
)
# Ground-only episodes ship an empty support stage, which the scene builder rejects.
SUPPORT_USDA="$HMD/whole_body/reconstructed_stage/${SEQ}_support.usda"
SUPPORT_OVERRIDE=""
if ! grep -q -E "def (Cube|Cylinder)" "$SUPPORT_USDA" 2>/dev/null; then SUPPORT_OVERRIDE="scene.support=false"; fi
echo "episode $EP seq $SEQ seed $SEED steps $STEPS thresholds $THR support=${SUPPORT_OVERRIDE:-usda}" | tee "$OUT/task.txt"
FINAL=$(ls "$OUT"/policy_*.safetensors 2>/dev/null | sort -t_ -k2 -n | tail -1)
if [ -z "$FINAL" ] || [ "$(basename "$FINAL" .safetensors | cut -d_ -f2)" -lt "$STEPS" ]; then
  cd "$FC" && python scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac \
    "task.parquet='${PARQUET}'" reset.seed=$SEED logging.mode=disabled \
    output_dir="$OUT" training.total_environment_steps=$STEPS "curriculum.thresholds=$THR" \
    training.checkpoint.save_interval_environment_steps=$STEPS $SUPPORT_OVERRIDE \
    "hydra.run.dir=$OUT/hydra" >> "$OUT/train.log" 2>&1
  echo "train exit $?" >> "$OUT/train.log"
fi
FINAL=$(ls "$OUT"/policy_*.safetensors 2>/dev/null | sort -t_ -k2 -n | tail -1)
if [ -z "$FINAL" ]; then echo "no checkpoint for ep $EP seed $SEED"; exit 1; fi
echo "{\"$EP\": \"$FINAL\"}" > "$OUT/checkpoints.json"
cd "$KIT" && python examples/run_policy_evaluation.py --checkpoints "$OUT/checkpoints.json" \
  --results "$OUT/results" --runtime-root "$FC" --world-count 1 > "$OUT/eval.log" 2>&1
echo "eval exit $? ep $EP seed $SEED"; grep -E "mppe|success|tracking" "$OUT/eval.log" | head -5
