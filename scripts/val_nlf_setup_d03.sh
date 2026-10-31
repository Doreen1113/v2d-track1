#!/bin/bash
# Pipeline mirror for the Date03 held-out set (val_nlf_build_d03.py), same as val_setup_root.sh but rooted at
# /work/doreen071/v2d_val/root_d03.
set -e
SRC=/work/doreen071/v2d
ROOT=/work/doreen071/v2d_val/root_d03
mkdir -p $ROOT/work/body_v2 $ROOT/work/sap_kp $ROOT/v2d-track1/scripts $ROOT/data/track_1/meta
for d in assets third_party kit envs; do
  [ -e $ROOT/$d ] || ln -s $SRC/$d $ROOT/$d
done
rsync -a --delete --exclude __pycache__ $SRC/v2d-track1/scripts/ $ROOT/v2d-track1/scripts/
grep -rlE '/work/doreen071/v2d\b' $ROOT/v2d-track1/scripts | while read -r f; do
  sed -i -E 's#/work/doreen071/v2d\b#/work/doreen071/v2d_val/root_d03#g' "$f"
done
cp -n $SRC/work/body_v2/kp70_mapping.npz $ROOT/work/body_v2/ 2>/dev/null || true
[ -f $SRC/work/sap_kp/kp308_mapping.npz ] && cp -n $SRC/work/sap_kp/kp308_mapping.npz $ROOT/work/sap_kp/ || true
echo "mirror ready: $ROOT"
