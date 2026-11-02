#!/bin/bash
# End-to-end reproduction of the Track 1 submission N1pw03 (repo commit f2f17f7 at submission time; this
# recipe is written against ae87473, which differs from f2f17f7 only by the Track 2 scripts/t2 folder):
#
#   N1pw03 = body "V-A" (SAM 3D Body dinov3 + vith fusion, sequence refinement v5 without the upright prior,
#          sigma_h 3, no finger penetration fix)
#        + objects_v6 (recipe composition of the v4 / v5 render-and-compare trajectories, sigma_o 12)
#        + NLF pelvis-depth fusion geolp0.3_5 (weight 0.3, low-pass 5 frames) with the object moved along with the body
#        + object meshes carved at the hand contacts (CSG), PEN exactly 0
#
# Usage:  bash reproduce.sh [stage ...]       (default: every stage, in order)
# Stages: data intrinsics masks person body depth vitpose refine ratio objects_v1 objects_v2 objects_v3
#         objects_v4 objects_v5 objects_v6 assemble carve nlf compose pack check
#
# Layout: everything lives under $ROOT (= /work/doreen071/v2d in the container; common.py reads V2D_ROOT
# but many scripts hard-code this path, so bind the data tree there). The repository is at $REPO.
# Episodes are processed with the same sharding as the original runs where it matters for memory, not for
# the result; shards are independent and can be spread over several GPUs / jobs.
#
# Environments (interpreters). The container ships PY_V2D (/opt/venv). The model stages need the other
# environments (container/README.md "Model environments"; pinned lists in container/env_freeze/):
#   PY_V2D    assembly, fusion, refinement, composition, NLF trajectory fusion, carving, checks, packing
#   PY_SAM3DB SAM 3D Body + ViTDet (sam-3d-body checkout, detectron2), NLF inference (torch >= 2.4)
#   PY_SAM3   SAM 3 / SAM 2.1 / Grounding DINO masks, MoGe-2 intrinsics / depth / point maps
#   PY_SAM3DO SAM 3D Objects, PyTorch3D object tracking (track_object_v2 / v4 / v5)
#   PY_GVHMR  GVHMR preprocessing (ViTPose-H keypoints)
# Every stage records what it ran into $ROOT/jobs/reproduce/<stage>.log.
set -euo pipefail

ROOT=${V2D_ROOT:-/work/doreen071/v2d}
REPO=${REPO:-/opt/v2d-track1}
S=$REPO/scripts
W=$ROOT/work
KIT=$ROOT/kit/v2d_submission_kit
SUB=$ROOT/submissions_v7
NLF_RAW=${NLF_RAW:-$ROOT/v2d_val/nlf/challenge/raw}   # where the original run kept the NLF output
PY_V2D=${PY_V2D:-/opt/venv/bin/python}
PY_SAM3DB=${PY_SAM3DB:-$HOME/.conda/envs/sam3db/bin/python}
PY_SAM3=${PY_SAM3:-$HOME/.conda/envs/sam3/bin/python}
PY_SAM3DO=${PY_SAM3DO:-$HOME/.conda/envs/sam3do/bin/python}
PY_GVHMR=${PY_GVHMR:-$HOME/.conda/envs/gvhmr/bin/python}
E3D=$(dirname $(dirname $PY_SAM3DO))
COMMIT_URL=${COMMIT_URL:-https://github.com/Doreen1113/v2d-track1/commit/f2f17f73277536d783627fe15953666bfb0b7e79}
LOG=$ROOT/jobs/reproduce
mkdir -p $LOG $W
cd $S

# SAM 3D Objects and PyTorch3D need the environment's own CUDA toolkit on the path (as in the original jobs).
env3d() { PATH=$E3D/bin:$PATH CUDA_HOME=$E3D CONDA_PREFIX=$E3D LIDRA_SKIP_INIT=1 "$@"; }
shards() { local n=$1; shift; for ((i = 0; i < n; i++)); do "$@" --shard $i/$n; done; }

stage_data() {
  # Track 1 dataset (public), submission kit, official MHR assets (Apache-2.0 release, not the SAM 3D Body copy).
  mkdir -p $ROOT/data $ROOT/assets
  $PY_V2D -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/video_to_data_challenge', repo_type='dataset', allow_patterns=['track_1/*', 'README.md'], local_dir='$ROOT/data')"
  curl -L -o $ROOT/kit.zip https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip
  $PY_V2D -c "import zipfile; zipfile.ZipFile('$ROOT/kit.zip').extractall('$ROOT/kit')"
  curl -L -o $ROOT/assets/mhr_assets.zip https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip
  $PY_V2D -c "import zipfile; zipfile.ZipFile('$ROOT/assets/mhr_assets.zip').extractall('$ROOT/assets/mhr_release_zip')"
  cp $(find $ROOT/assets/mhr_release_zip -name mhr_model.pt | head -1) $ROOT/assets/mhr_model.pt
  echo "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc  $ROOT/assets/mhr_model.pt" | sha256sum -c -
  # ViTDet-H person detector used by SAM 3D Body's tracker (detectron2 model zoo, Apache-2.0).
  mkdir -p $ROOT/assets/vitdet
  curl -L -o $ROOT/assets/vitdet/model_final_f05665.pkl https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl
  # Gated / licensed weights (user's own Hugging Face account, after accepting each model's licence; README):
  #   facebook/sam3, facebook/sam-3d-body-dinov3, facebook/sam-3d-body-vith, facebook/sam-3d-objects
  #   (the scripts download them with huggingface_hub at first use; SAM 3D Objects also needs its
  #   checkpoints under $ROOT/third_party/sam-3d-objects/checkpoints/hf, see README),
  #   NLF TorchScript model -> $ROOT/third_party_new/nlf_models/nlf_l_multi_0.3.2.patch6.torchscript,
  #   GVHMR inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth (+ hmr2, loaded by the preprocessing).
}

stage_intrinsics() {
  # MoGe-2 focal on 12 frames per episode; the scripts then use the global median focal (807.66 px) for all
  # four cameras with the principal point at the image centre -> work/intrinsics/episode_XXXXXX.json
  $PY_SAM3 estimate_intrinsics.py --frames 12 --out $W/intrinsics
  # The per-episode medians were replaced by their global median (807.66 px on the original run, the
  # fallback in common.load_intrinsics): the four cameras share one lens. This step was done once by hand
  # on the original run (the stored JSONs all carry fx = fy = 807.6573); it is written out here.
  $PY_V2D - <<EOF
import json, glob, numpy as np
fs = sorted(glob.glob("$W/intrinsics/episode_*.json")); recs = [json.load(open(f)) for f in fs]
f = float(np.median([r["fx"] for r in recs])); print("global median focal", f)
for p, r in zip(fs, recs):
    r["fx"] = r["fy"] = f; r["focal_source"] = "moge2_global_median"; json.dump(r, open(p, "w"), indent=1)
EOF
}

stage_masks() {
  # Object + person masks: Grounding DINO + SAM 2.1 (complete coverage) and SAM 3 (precise), merged.
  shards 10 $PY_SAM3 segment_objects.py --skip-existing --out-dir $W/masks_sam2
  shards 10 $PY_SAM3 segment_objects_sam3.py --skip-existing --out-dir $W/masks_sam3
  $PY_SAM3 merge_masks.py --sam2 $W/masks_sam2 --sam3 $W/masks_sam3 --out $W/masks
}

stage_person() {
  # Main-actor boxes (ViTDet + tracker) for the GVHMR preprocessing (ViTPose keypoints).
  $PY_SAM3DB track_person_bbox.py --all
}

stage_body() {
  # Member A: SAM 3D Body dinov3, full mode (own ViTDet tracker, hand decoder prompts) -> work/body
  shards 4 $PY_SAM3DB run_sam3d_body.py --all --skip-existing --hf-repo facebook/sam-3d-body-dinov3 --intr-dir $W/intrinsics --out $W/body
  # Member B: SAM 3D Body vith, body decoder only, on A's boxes, scored frames +-30 -> work/body_vith_b
  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} $PY_SAM3DB run_sam3d_body.py --all --skip-existing --hf-repo facebook/sam-3d-body-vith \
    --intr-dir $W/intrinsics --boxes-from $W/body --window 30 --body-only --batch 16 --n-vis 0 --out $W/body_vith_b
  # Scored frames (from the sample parquet) and the 70-keypoint regressor rows used by the refinement.
  mkdir -p $W/body_v2
  [ -f $W/body_v2/scored_frames.json ] || $PY_V2D - <<EOF
import sys; sys.path.insert(0, "$S")
import json, refine_body_sequence as r
json.dump({str(k): v for k, v in r.scored_frames().items()}, open("$W/body_v2/scored_frames.json", "w"))
EOF
  [ -f $W/body_v2/kp70_mapping.npz ] || $PY_SAM3DB refine_body_sequence.py --export-kp-map
  # Fusion A + B -> work/body_ens. body_ens was produced on 2026-10-29 with the script at commit 553e66c
  # (options --w-b 0.5, --keep-a 68:122). At HEAD --member-weight 1 is the same average; --keep-a must be
  # given explicitly because its default changed to 30:50,68:122,132:134 in commit b534320.
  # TODO(ambiguous): the HEAD script also differs from 553e66c in its member checks; for a bit-exact
  # body_ens use `git show 553e66c:scripts/fuse_body_ensemble.py`.
  $PY_V2D fuse_body_ensemble.py --a $W/body --b $W/body_vith_b --out $W/body_ens --member-weight 1.0 --keep-a 68:122 --window 30
}

stage_depth() {
  # MoGe-2 metric depth at the fixed focal (quarter resolution) -> work/depth; used by the body refinement
  # (floor plane) and by every object stage.
  shards 6 $PY_SAM3 compute_depth.py
}

stage_vitpose() {
  # ViTPose-H COCO-17 keypoints (GVHMR preprocessing; only the preproc stage is needed) -> work/body_gvhmr/preproc
  shards 8 $PY_GVHMR run_gvhmr.py --all --stages preproc
}

stage_refine() {
  # body_v2: sequence refinement (per-episode identity), needed by the object stages (hands, floor, ratio).
  shards 6 $PY_V2D refine_body_sequence.py --all --out $W/body_v2
  # body_v3: pooled identity, sig_acc_rel 0.016 (defaults of refine_body_v3.py), input of objects_v5.
  shards 6 $PY_V2D refine_body_v3.py --all --out $W/body_v3
  # body_vA ("V-A"): body_v2 setting on the fused input, upright prior off, no depth target.
  shards 2 $PY_V2D refine_body_v5.py --all --no-upright --body-in $W/body_ens --out $W/body_vA
}

stage_ratio() {
  # MoGe / body scale from the floor plane, one JSON per refined body (objects: floor.json and floor_b3.json;
  # final assembly: floor_bA.json).
  $PY_V2D make_floor_ratio.py --body $W/body_v2 --out $W/moge_to_body_ratio_floor.json
  $PY_V2D make_floor_ratio.py --body $W/body_v3 --out $W/moge_to_body_ratio_floor_b3.json
  $PY_V2D make_floor_ratio.py --body $W/body_vA --out $W/moge_to_body_ratio_floor_bA.json
}

stage_objects_v1() {
  # v1: SAM 3D Objects mesh on the cleanest frames + per-frame render-and-compare pose -> work/objects
  $PY_SAM3 prepare_object_frames.py --k 4
  shards 6 env3d $PY_SAM3DO reconstruct_object.py --k 4
  shards 10 env3d $PY_SAM3DO track_object_pose.py
}

stage_objects_v2() {
  # v2: one shared mesh per object class (8 SAM 3D Objects candidates per episode, multi-frame silhouette
  # selection), whole-trajectory fit with static segments, contact depth correction -> work/objects_v2
  for i in 0 1 2 3 4; do $PY_SAM3 object_candidates.py --stage prep --shard $i/5; env3d $PY_SAM3DO object_candidates.py --stage recon --shard $i/5; done
  env3d $PY_SAM3DO select_class_mesh.py
  env3d $PY_SAM3DO select_class_mesh.py --init-only --override black_pan=ep13_f000042
  shards 6 env3d $PY_SAM3DO track_object_v2.py
  env3d $PY_SAM3DO finalize_objects_v2.py
}

stage_objects_v3() {
  # v3: the v2 tracks re-finalised with the floor-plane ratio -> work/objects_v3 (as run_v3.sh)
  mkdir -p $W/objects_v3 && ln -sfn $W/objects_v2/track $W/objects_v3/track
  OBJ_V2=$W/objects_v3 env3d $PY_SAM3DO finalize_objects_v2.py --ratio $W/moge_to_body_ratio_floor.json
}

stage_objects_v4() {
  # v4: budgeted mesh + SDF grid, then the joint hand-aware fit from the v3 trajectory -> work/objects_v4
  # TODO(ambiguous): objects_v4 was last rebuilt on 2026-10-17/18 through run_v4.sh, whose defaults are
  # --body work/body_v3 --ratio floor_b3.json (the script's own defaults are body_v2 / floor.json).
  local items=""
  for e in $(seq 0 29); do items="$items $W/objects_v3/$(printf episode_%06d_object.ply $e):$e"; done
  $PY_V2D contactlib.py $W/objects_v4/sdf $items
  for g in "27 15 22 0 5 10 24 26" "16 17 4 1 6 12 23" "9 11 2 7 13 18 25" "3 8 14 19 20 21 28 29"; do
    env3d $PY_SAM3DO track_object_v4.py --init $W/objects_v3 --out $W/objects_v4 --sdf $W/objects_v4/sdf \
      --body $W/body_v3 --ratio $W/moge_to_body_ratio_floor_b3.json --episodes $g
  done
}

stage_objects_v5() {
  # v5: support-surface calibration + rest constraints on top of v4 (same meshes) -> work/objects_v5
  # (run_v5.sh defaults: body_v3, floor_b3)
  for g in "27 15 22 0 5 10 24 26" "16 17 4 1 6 12 23" "9 11 2 7 13 18 25" "3 8 14 19 20 21 28 29"; do
    env3d $PY_SAM3DO track_object_v5.py --init $W/objects_v4 --out $W/objects_v5 --sdf $W/objects_v4/sdf \
      --body $W/body_v3 --ratio $W/moge_to_body_ratio_floor_b3.json --episodes $g
  done
}

stage_objects_v6() {
  # v6: per-episode recipe composition of v4 / v5 (scripts/recipes_v6.json) -> work/objects_v6
  $PY_V2D compose_objects_v6.py --out $W/objects_v6 --recipes $S/recipes_v6.json --body $W/body_v2 --ratio $W/moge_to_body_ratio_floor.json
}

stage_assemble() {
  # A1/episodes_prefix: body_vA sigma_h 3 + objects_v6 sigma_o 12 + floor ratio bA (no finger fix: N1pw03 uses
  # the "prefix" episodes, the finger fix of A1 is not applied)
  $PY_V2D assemble_episode.py --body $W/body_vA --sigma-h 3 --objects $W/objects_v6 --sigma-o 12 \
    --ratio $W/moge_to_body_ratio_floor_bA.json --out $SUB/A1/episodes_prefix
}

stage_carve() {
  # Object meshes carved at the hand contacts (scorer-exact), one job per episode in the original run
  # (identical flags for all 30). Output A1pc/episodes = copy of A1/episodes_prefix with carved .ply files.
  for e in $(seq 0 29); do
    OMP_NUM_THREADS=8 $PY_V2D carve_object_contacts.py --src $SUB/A1/episodes_prefix --out $SUB/A1pc/episodes --episodes $e \
      --remesh 0 --iters 25 --csg-band 10 --csg-tries 24 --csg-rboost-max 20 --report $LOG/A1pc_e${e}_report.json
  done
}

stage_nlf() {
  # NLF (SMPL, camera frame) on the challenge videos, with the SAM 3D Body boxes and the pipeline intrinsics.
  # The original run used jobs/nlf/infer.sbatch (4 lanes, EPS 0..29, OUT=$NLF_RAW) with the per-run
  # overrides passed through its EXTRA variable, which the job log does not record.
  # TODO(ambiguous): --video-dir / --intr-dir / --boxes-from below are the challenge-tree equivalents of the
  # script's validation defaults (its default --boxes-from is the validation root's work/body_moge, i.e.
  # the SAM 3D Body run; for the challenge that is work/body). --batch / --num-aug / --box-scale defaults.
  $PY_SAM3DB val_nlf_infer.py --episodes $(seq -s ' ' 0 29) --out $NLF_RAW --skip-existing \
    --video-dir $ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera --intr-dir $W/intrinsics --boxes-from $W/body
  # Depth-path fusion geolp0.3_5 (z = z_VA^0.7 z_NLF^0.3 along the pelvis ray, correction low-passed
  # over 5 frames; weight 0.3 chosen on the leaderboard sweep 0.15-1.0), the object moved by the same per-frame
  # offset -> N1pw03/episodes (jobs/wprobe/N1pw03.sbatch; N1pc used geolp0.5_5)
  $PY_V2D val_nlf_traj.py --base $SUB/A1/episodes_prefix --nlf $NLF_RAW --src raw --mode geolp0.3_5 --with-object --out $SUB/N1pw03/episodes
}

stage_compose() {
  # N1pw03/episodes = the NLF-fused poses (object co-moved) + the A1pc carved meshes. The carve was
  # run on A1pc (same hand / object geometry: the body and the object are shifted by the same offset per
  # frame, so the contacts and the carved meshes stay valid); verified file-identical for the submission.
  mkdir -p $SUB/N1pw03/episodes
  # (val_nlf_traj.py already wrote the .npz files into N1pw03/episodes)
  cp $SUB/A1pc/episodes/episode_*_object.ply $SUB/N1pw03/episodes/
}

stage_pack() {
  # Kit packer -> parquet (reduces every mesh to the submission budget; the carve worked on that reduced mesh)
  (cd $KIT && $PY_V2D eval_reconstruction.py --episodes $SUB/N1pw03/episodes --sample data/track_1_sample_submission.parquet \
    --commit $COMMIT_URL --out $SUB/N1pw03/N1pw03.parquet)
}

stage_check() {
  # Scorer-exact penetration on all 30 episodes (expected: PEN 0, 0 points inside) and a format check.
  $PY_V2D check_pen_exact.py --episodes-dir $SUB/N1pw03/episodes --workers 3 --threads 2 --json $LOG/N1pw03_pen.json
  $PY_V2D - <<EOF
import pandas as pd, numpy as np
a = pd.read_parquet("$SUB/N1pw03/N1pw03.parquet"); b = pd.read_parquet("$KIT/data/track_1_sample_submission.parquet")
assert len(a) == len(b) and list(a.columns) == list(b.columns), (a.shape, b.shape)
num = a.select_dtypes("number"); assert np.isfinite(num.to_numpy(dtype=float)).all()
print("parquet ok:", a.shape)
EOF
}

STAGES=${@:-data intrinsics masks person body depth vitpose refine ratio objects_v1 objects_v2 objects_v3 objects_v4 objects_v5 objects_v6 assemble carve nlf compose pack check}
for st in $STAGES; do
  echo "== stage $st $(date -Is)" | tee -a $LOG/$st.log
  stage_$st 2>&1 | tee -a $LOG/$st.log
  echo "== done $st $(date -Is)" | tee -a $LOG/$st.log
done
