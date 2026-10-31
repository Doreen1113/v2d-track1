#!/bin/bash
# Score runs under /work/doreen071/v2d_val/runs on both sets (scored + frame-drop) and both role choices into
# /work/doreen071/v2d_val/nlf/results/<run>[_framedrop][_nohand].json
#   val_nlf_score.sh RUN [RUN ...]
VAL=/work/doreen071/v2d_val; R=$VAL/nlf/results
PY=/home/doreen071/.conda/envs/v2d/bin/python
cd /work/doreen071/v2d/v2d-track1/scripts
mkdir -p $R
for r in "$@"; do
  for S in "" framedrop; do
    X=${S:+--scored scored_frames_framedrop.json}; T=${S:+_framedrop}
    $PY val_score.py --pred $VAL/runs/$r/episodes --forward torch --quiet $X --out $R/$r$T.json 2>&1 | grep MEAN | sed "s/^/$r$T all    /"
    $PY val_score.py --pred $VAL/runs/$r/episodes --forward torch --quiet $X --roles nohand --out $R/$r${T}_nohand.json 2>&1 | grep MEAN | sed "s/^/$r$T nohand /"
  done
done
