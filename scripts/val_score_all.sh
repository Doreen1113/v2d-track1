#!/bin/bash
# Score every assembled run under /work/doreen071/v2d_val/runs (or the given run names) with both vertex-role
# choices: results/<run>.json (roles all) and results/<run>_nohand.json.
#   bash val_score_all.sh [run ...]
#   SCORED=scored_frames_framedrop.json bash val_score_all.sh [run ...]   # the frame-drop episodes -> *_framedrop*.json
VAL=/work/doreen071/v2d_val
PY=/home/doreen071/.conda/envs/v2d/bin/python
cd /work/doreen071/v2d/v2d-track1/scripts
mkdir -p $VAL/results
RUNS=${@:-$(ls $VAL/runs)}
X=${SCORED:+--scored $SCORED}; T=${SCORED:+_framedrop}
for r in $RUNS; do
  $PY val_score.py --pred $VAL/runs/$r/episodes --forward torch --quiet $X --out $VAL/results/$r$T.json 2>&1 | grep MEAN | sed "s/^/$r all    /"
  $PY val_score.py --pred $VAL/runs/$r/episodes --forward torch --quiet $X --roles nohand --out $VAL/results/${r}${T}_nohand.json 2>&1 | grep MEAN | sed "s/^/$r nohand /"
done
