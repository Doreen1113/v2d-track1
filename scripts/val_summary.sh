#!/bin/bash
# Collect the validation-set tables into /work/doreen071/v2d_val/results/summary.md (after val_score_all.sh).
VAL=/work/doreen071/v2d_val; R=$VAL/results
PY=/home/doreen071/.conda/envs/v2d/bin/python
cd /work/doreen071/v2d/v2d-track1/scripts
have() { for r in "$@"; do [ -f $R/$r.json ] && echo $R/$r.json; done; }
{
echo "# Validation results ($(date '+%F %R'))"
echo
echo "Episodes in gt/scored_frames.json; CD-H in cm, roles 'all' unless marked nohand. Columns: focal setting."
echo
echo "## Focal settings, CD-H (roles all)"
$PY val_table.py $(have moge gt gt090 gt095 gt105 gt110) --metric cd_h_cm --diff
echo
echo "## Focal settings, CD-H (roles nohand)"
$PY val_table.py $(have moge_nohand gt_nohand gt090_nohand gt095_nohand gt105_nohand gt110_nohand) --metric cd_h_cm --diff
echo
echo "## Depth ratio (pred / GT pelvis depth, raw camera frame) and frame-0 Sim(3) scale"
$PY val_table.py $(have moge gt gt090 gt095 gt105 gt110) --metric z_ratio sim3_scale
echo
echo "## ACC-H (floored by GT jitter, see README)"
$PY val_table.py $(have moge gt gt095 gt105) --metric acc_h_cm
echo
echo "## Focal response (CD-H, roles all)"
$PY val_focal_response.py $(for s in moge:intrinsics gt:intrinsics_gt gt090:intrinsics_gt090 gt095:intrinsics_gt095 gt105:intrinsics_gt105 gt110:intrinsics_gt110; do n=${s%%:*}; [ -f $R/$n.json ] && echo "$n=$R/$n.json:${s#*:}"; done)
echo
echo "## Focal response (CD-H, roles nohand)"
$PY val_focal_response.py $(for s in moge:intrinsics gt:intrinsics_gt gt090:intrinsics_gt090 gt095:intrinsics_gt095 gt105:intrinsics_gt105 gt110:intrinsics_gt110; do n=${s%%:*}; [ -f $R/${n}_nohand.json ] && echo "$n=$R/${n}_nohand.json:${s#*:}"; done)
if [ -f $R/v6sap_moge.json ]; then
  echo
  echo "## Sapiens refine (v6) vs refine_v5, same SAM3DB input (MoGe focal)"
  $PY val_table.py $R/moge.json $R/v6sap_moge.json --metric cd_h_cm acc_h_cm sim3_scale --diff
  $PY val_table.py $R/moge_nohand.json $R/v6sap_moge_nohand.json --metric cd_h_cm --diff
fi
if [ -f $R/moge_framedrop.json ]; then
  echo
  echo "## Frame-drop episodes (not in the main set)"
  $PY val_table.py $(have moge_framedrop gt_framedrop gt095_framedrop gt105_framedrop) --metric cd_h_cm z_ratio --diff
fi
} > $R/summary.md 2>&1
echo "" >> $R/summary.md; echo "Scorer validation: results/scorer_validation.md" >> $R/summary.md; echo "wrote $R/summary.md"
