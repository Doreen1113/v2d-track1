"""Held-out check for NLF: a second, smaller validation set from BEHAVE's official TEST split (Date03, subjects Sub03 /
Sub04, which appear in no training sequence). NLF's training mix includes BEHAVE (its training config lists it), and
every clip of the main validation set (Date01/02/04/05/06) is in BEHAVE's train split, so NLF may have seen those
exact frames. This set is built with val_build_behave.py's stages, unchanged except:
  - BEHAVE root /work/doreen071/v2d_val/data_d03/behave (params / calibs / objects linked from the main download,
    Date03 colour videos and timestamps added)
  - no per-subject SMPL fit (the SMPL-H keyframe meshes of Date03 are a 45 GB download): the SMPL-H betas are used as
    SMPL betas, so the GT shape is off by a few cm; this is shared by every method scored against it
  - Kinect 2 throughout, the 600-frame window chosen automatically (stage plan)
  - outputs: GT /work/doreen071/v2d_val/gt_d03, pipeline mirror /work/doreen071/v2d_val/root_d03

  python val_nlf_build_d03.py --stage plan --gt /work/doreen071/v2d_val/gt_d03_cand --root /work/doreen071/v2d_val/root_d03 [--cand]
  python val_nlf_build_d03.py --stage plan intr video meta gt --gt /work/doreen071/v2d_val/gt_d03 --root /work/doreen071/v2d_val/root_d03
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import val_build_behave as vb  # noqa: E402

D03 = "/work/doreen071/v2d_val/data_d03/behave"
CANDIDATES = [
    ("Date03_Sub03_suitcase_lift", "suitcase"), ("Date03_Sub03_boxlarge", "cardboard box"),
    ("Date03_Sub03_monitor_move", "computer monitor"), ("Date03_Sub03_trashbin", "trash bin"),
    ("Date03_Sub03_toolbox", "toolbox"), ("Date03_Sub03_tablesmall_move", "small table"),
    ("Date03_Sub03_backpack_hand", "backpack"), ("Date03_Sub03_boxlong", "cardboard box"),
    ("Date03_Sub04_backpack_hand", "backpack"), ("Date03_Sub04_boxlong", "cardboard box"),
    ("Date03_Sub04_suitcase_lift", "suitcase"), ("Date03_Sub04_toolbox", "toolbox"),
    ("Date03_Sub04_trashbin", "trash bin"), ("Date03_Sub04_monitor_move", "computer monitor"),
    ("Date03_Sub04_boxlarge", "cardboard box"), ("Date03_Sub04_suitcase_ground", "suitcase"),
    ("Date03_Sub04_tablesmall_lift", "small table"), ("Date03_Sub03_boxmedium", "cardboard box"),
]
# chosen from the candidate plan: whole body in view on >= 99% of the window, >= 90% registered frames,
# <= 2 colour-stream time jumps (filled in after the candidate run)
SELECTED = os.environ.get("D03_SELECTED")

vb.BEHAVE = D03
vb.FRAME_DROP = set()
vb.subject_fit = lambda seq: None
if "--cand" in sys.argv:
    sys.argv.remove("--cand")
    vb.SEQUENCES = [(s, None, None, p) for s, p in CANDIDATES]
else:
    sel = [int(i) for i in SELECTED.split(",")]
    # Kinect 2 for all: it is the view the candidate plan picked (the others drop colour frames on Date03 or cut the
    # body); without this the plan's per-episode view rotation would move some episodes to another Kinect
    vb.SEQUENCES = [(CANDIDATES[i][0], 2, None, CANDIDATES[i][1]) for i in sel]

if __name__ == "__main__":
    if "--videos" not in sys.argv:
        sys.argv += ["--videos", f"{D03}/videos"]
    vb.main()
