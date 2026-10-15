#!/bin/bash
# Object stage v2 as dependent Slurm jobs (outputs in work/objects_v2, v1 in work/objects untouched).
#   STAGES (default all): cand select init track final eval
#   usage: bash run_object_v2.sh [stage ...]
#   cand    object_candidates.py   SAM 3D Objects meshes from 8 frames per episode   (5 jobs)
#   select  select_class_mesh.py   one mesh per class by multi-frame silhouette score (5 jobs)
#   init    select_class_mesh.py --init-only: class-mesh inits (pan: the best candidate WITH handle)
#   track   track_object_v2.py     whole-trajectory fit with static segments          (6 jobs)
#   final   finalize_objects_v2.py contact depth correction + v1-layout files         (1 job)
#   eval    evaluate_object_tracks.py on work/objects_v2                               (1 job)
set -e
ROOT=/work/doreen071/v2d
S=$ROOT/v2d-track1/scripts
RUN=$S/slurm/object_v2_run.sbatch
DEP=""
CLASSES=(big_red_bowl black_pan foam_grass_block hula_hoop iron paint_roller pink_foam_roll short_wood_stool white_desk white_laptop_cart)
sub() {  # sbatch args...
  sbatch --parsable ${DEP:+--dependency=afterok:$DEP} "$@"
}
STAGES=${@:-cand select init track final eval}
cd $ROOT/jobs
for st in $STAGES; do case $st in
  cand)
    DEP=$(sub $S/slurm/object_v2_candidates.sbatch) ;;
  select)
    ids=""
    for i in 0 2 4 6 8; do
      id=$(sub -J sel$i $RUN select_class_mesh.py --classes ${CLASSES[$i]} ${CLASSES[$((i+1))]}); ids=${ids:+$ids:}$id
    done
    DEP=$ids ;;
  init)
    DEP=$(sub -J objinit $RUN select_class_mesh.py --init-only --override black_pan=ep13_f000042) ;;
  track)
    ids=""
    for i in 0 1 2 3 4 5; do
      id=$(sub -J trk$i $RUN track_object_v2.py --shard $i/6); ids=${ids:+$ids:}$id
    done
    DEP=$ids ;;
  final)
    DEP=$(sub -J objfin $RUN finalize_objects_v2.py) ;;
  eval)
    DEP=$(sub -J objeval $RUN evaluate_object_tracks.py --objects $ROOT/work/objects_v2 \
          --out $ROOT/work/objects_v2/eval_v2.json) ;;
esac; echo "$st -> $DEP"; done
