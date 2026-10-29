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
if [ -f $R/va_moge.json ]; then
  echo
  echo "## V-A (dinov3 + vith body-only fusion) vs MoGe baseline"
  $PY val_table.py $R/moge.json $R/va_moge.json --metric cd_h_cm --diff
  $PY val_table.py $R/moge_nohand.json $R/va_moge_nohand.json --metric cd_h_cm --diff
fi
if ls $R/oracle_a_*.json > /dev/null 2>&1; then
  echo
  echo "## Oracle error budget (val_oracle.py): mean CD-H over the scored episodes, one component replaced by the GT"
  $PY - <<'PYEOF'
import json, os
import numpy as np
R = "/work/doreen071/v2d_val/results"
names = [("", "prediction as is"), ("a", "pelvis trajectory since f0"), ("b", "  depth (camera z) only"),
         ("c", "  x / y only"), ("d", "torso orientation since f0"), ("ad", "trajectory + orientation since f0"),
         ("dabs", "torso orientation, absolute (f0 too)"), ("e", "articulation (torso-anchored)"),
         ("f", "shape + scales"), ("g", "per-frame rigid placement of the mesh (rot + trans)"),
         ("grot", "  rotation part of g only (about the pelvis)"),
         ("h", "articulation + identity (only the predicted rigid placement left)"), ("all", "everything (sanity)"),
         ("droot", "[ill-defined] root rotation param since f0"), ("drootabs", "[ill-defined] root rotation param"),
         ("eparam", "[ill-defined] articulation params")]
def mean(run, nohand):
    p = f"{R}/{run}{'_nohand' if nohand else ''}.json"
    if not os.path.exists(p):
        return None
    e = json.load(open(p))["episodes"]
    return np.mean([v["cd_h_cm"] for v in e.values()]), len(e)
runs = [("moge", "moge"), ("va", "va_moge")]
print("| oracle | " + " | ".join(f"{n} all | {n} nohand" for n, _ in runs) + " |")
print("|---|" + "---|---|" * len(runs))
for x, label in names:
    cells = []
    for n, base in runs:
        for nh in (False, True):
            m = mean(base if not x else f"oracle_{x}_{n}", nh)
            cells.append("-" if m is None else f"{m[0]:.2f}")
    print(f"| {x or 'base'}: {label} | " + " | ".join(cells) + " |")
PYEOF
fi
if [ -f $R/moge_framedrop.json ]; then
  echo
  echo "## Frame-drop episodes (not in the main set)"
  $PY val_table.py $(have moge_framedrop gt_framedrop gt095_framedrop gt105_framedrop) --metric cd_h_cm z_ratio --diff
fi
} > $R/summary.md 2>&1
echo "" >> $R/summary.md; echo "Scorer validation: results/scorer_validation.md" >> $R/summary.md; echo "wrote $R/summary.md"
