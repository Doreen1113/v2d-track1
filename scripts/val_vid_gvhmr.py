"""GVHMR (video model, camera-space "incam" SMPL-X output) on the validation episodes, converted to MHR.

Thin wrapper around run_gvhmr.py (stages predict + convert) with the validation paths:
  - GVHMR code: /work/doreen071/v2d/third_party_new/GVHMR (copy of third_party/GVHMR, commit ee960bb, with
    inputs/checkpoints/body_models/{smplx,smpl} linked; checkpoints linked to the original checkout)
  - preprocessing (boxes, ViTPose-H, HMR2 features): root/work/body_gvhmr/preproc (run_gvhmr.py --stages preproc)
  - intrinsics: root/work/intrinsics (MoGe-2, as the baseline)
  - output: root/work/body_vid_gvhmr/{smplx/,episode_XXXXXX.npz}; the episode NPZs are in the refined-body format
    (pose [T,136], scales, shape, valid) and go straight into val_assemble.py.

Runs in the gvhmr env:
  /home/doreen071/.conda/envs/gvhmr/bin/python val_vid_gvhmr.py --episodes 0 7 8 --stages predict convert
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_gvhmr as rg  # noqa: E402

VAL = "/work/doreen071/v2d_val"
NEW_ROOT = "/work/doreen071/v2d/third_party_new/GVHMR"
rg.GVHMR_ROOT = NEW_ROOT
rg.SMPLX_FILE = os.path.join(NEW_ROOT, "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz")
rg.INTR_DIR = f"{VAL}/root/work/intrinsics"
rg.VIDEO_DIR = f"{VAL}/root/data/track_1/videos/chunk-000/observation.images.exo_camera"
rg.OUT_DIR = f"{VAL}/root/work/body_vid_gvhmr"
sys.path = [p for p in sys.path if p != "/work/doreen071/v2d/third_party/GVHMR"]
sys.path.insert(0, NEW_ROOT)


def main():
    os.makedirs(rg.OUT_DIR, exist_ok=True)
    pp = os.path.join(rg.OUT_DIR, "preproc")
    if not os.path.exists(pp):
        try:
            os.symlink(f"{VAL}/root/work/body_gvhmr/preproc", pp)
        except FileExistsError:  # another lane made it
            pass
    if "--episodes" not in sys.argv and "--all" not in sys.argv:
        sys.argv += ["--episodes"] + [str(e) for e in range(19)]
    if "--out" not in sys.argv:
        sys.argv += ["--out", rg.OUT_DIR]
    import hmr4d
    assert hmr4d.__file__.startswith(NEW_ROOT), hmr4d.__file__
    import hmr4d.model.gvhmr.gvhmr_pl_demo  # noqa: F401  registers model/gvhmr/gvhmr_pl_demo in the hydra store
    rg.main()


if __name__ == "__main__":
    main()
