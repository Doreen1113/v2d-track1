#!/bin/bash
# Mirror of the Track 1 tree for the local validation set: the pipeline scripts are copied and every
# /work/doreen071/v2d path in them is rewritten to the mirror root, so the unmodified pipeline runs on the
# validation videos without touching the challenge work/ directory.
#   bash val_setup_root.sh            # (re)copy the scripts, keep everything else
# Mirror layout (ROOT=/work/doreen071/v2d_val/root):
#   data/track_1/videos/chunk-000/observation.images.exo_camera/episode_XXXXXX.mp4   (val_build_behave.py)
#   data/track_1/meta/episodes_metadata.jsonl                                         (val_build_behave.py)
#   work/                       pipeline outputs (masks, depth, intrinsics, body_gvhmr/preproc, body*, ...)
#   v2d-track1/scripts/         rewritten copy of the repo scripts (re-run this file after pulling new scripts)
#   assets, third_party, kit, envs -> symlinks into /work/doreen071/v2d (read-only use)
set -e
SRC=/work/doreen071/v2d
VAL=/work/doreen071/v2d_val
ROOT=$VAL/root
mkdir -p $ROOT/work/body_v2 $ROOT/work/sap_kp $ROOT/v2d-track1/scripts $ROOT/data/track_1/meta
for d in assets third_party kit envs; do
  [ -e $ROOT/$d ] || ln -s $SRC/$d $ROOT/$d
done
rsync -a --delete --exclude __pycache__ $SRC/v2d-track1/scripts/ $ROOT/v2d-track1/scripts/
# rewrite absolute paths (word boundary: /work/doreen071/v2d_val is left alone, so this is idempotent)
grep -rlE '/work/doreen071/v2d\b' $ROOT/v2d-track1/scripts | while read -r f; do
  sed -i -E 's#/work/doreen071/v2d\b#/work/doreen071/v2d_val/root#g' "$f"
done
# Slurm accounts in copied sbatch files are irrelevant: the val jobs live in $VAL/jobs
# resources the pipeline reads from work/ that are not per-episode outputs
cp -n $SRC/work/body_v2/kp70_mapping.npz $ROOT/work/body_v2/ 2>/dev/null || true
[ -f $SRC/work/sap_kp/kp308_mapping.npz ] && cp -n $SRC/work/sap_kp/kp308_mapping.npz $ROOT/work/sap_kp/ || true
# the scored frames of the validation set, where the pipeline expects the challenge's
# (scored set + frame-drop episodes, so every built episode has an entry)
if [ -f $VAL/gt/scored_frames.json ]; then
  /home/doreen071/.conda/envs/v2d/bin/python -c "
import json, os
g = '$VAL/gt'
d = json.load(open(g + '/scored_frames.json'))
if os.path.exists(g + '/scored_frames_framedrop.json'):
    d.update(json.load(open(g + '/scored_frames_framedrop.json')))
json.dump({k: d[k] for k in sorted(d, key=int)}, open('$ROOT/work/body_v2/scored_frames.json', 'w'))"
fi
echo "mirror ready: $ROOT ($(ls $ROOT/v2d-track1/scripts | wc -l) scripts)"
grep -rn '/work/doreen071/v2d\b' $ROOT/v2d-track1/scripts | head -3 || true
