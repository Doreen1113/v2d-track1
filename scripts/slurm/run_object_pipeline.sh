#!/bin/bash
# Submit the object pipeline as dependent Slurm jobs (1 GPU each, sharded over episodes).
#   STAGES (default all): seg2 seg3 merge depth prep recon track
#   usage: bash run_object_pipeline.sh [stage ...]
# Environments: sam3 env (SAM 2.1, SAM 3, Grounding DINO, MoGe-2), sam3do env (SAM 3D Objects, PyTorch3D).
set -e
ROOT=/work/doreen071/v2d
S=$ROOT/v2d-track1/scripts
LOG=$ROOT/jobs
PY3=/home/doreen071/.conda/envs/sam3/bin/python
E3D=/home/doreen071/.conda/envs/sam3do
NSHARD=${NSHARD:-10}
STAGES=${@:-seg2 seg3 merge depth prep recon track}
HDR="#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00"
ENV3D="export PATH=$E3D/bin:\$PATH CUDA_HOME=$E3D CONDA_PREFIX=$E3D LIDRA_SKIP_INIT=1"
DEP=""
submit() {  # name, nshard, command-with-SHARD
  local name=$1 n=$2 cmd=$3 ids=""
  for ((i=0;i<n;i++)); do
    f=$LOG/${name}_$i.sbatch
    printf '#!/bin/bash\n#SBATCH --job-name=%s\n%s\n#SBATCH --output=%s/%s_%%j.log\n%s\n' \
      "${name}_$i" "$HDR" "$LOG" "${name}_$i" "${cmd//SHARD/$i\/$n}" > $f
    id=$(sbatch --parsable ${DEP:+--dependency=afterok:$DEP} $f); ids=${ids:+$ids:}$id
  done
  DEP=$ids; echo "$name -> $ids"
}
for st in $STAGES; do case $st in
  seg2)  submit seg2 $NSHARD "$PY3 $S/segment_objects.py --shard SHARD --out-dir $ROOT/work/masks_sam2" ;;
  seg3)  submit seg3 $NSHARD "$PY3 $S/segment_objects_sam3.py --shard SHARD --out-dir $ROOT/work/masks_sam3" ;;
  merge) submit merge 1 "$PY3 $S/merge_masks.py" ;;
  depth) submit depth 6 "$PY3 $S/compute_depth.py --shard SHARD" ;;
  prep)  submit prep 1 "$PY3 $S/prepare_object_frames.py --k 4" ;;
  recon) submit recon 6 "$ENV3D; $E3D/bin/python $S/reconstruct_object.py --shard SHARD --k 4" ;;
  track) submit track $NSHARD "$ENV3D; $E3D/bin/python $S/track_object_pose.py --shard SHARD" ;;
esac; done
