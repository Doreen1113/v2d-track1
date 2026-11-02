# Reproduction container (Track 1, submission N1pw03)

This folder holds what the challenge asks for next to the technical report: a container that reproduces
the selected leaderboard submission, **N1pw03** (CD-H 14.537, CD-O 22.276, ACC-H 0.1375, ACC-O 0.0731,
PEN 0.00000; submitted 2026-11-02 with repo commit `f2f17f7`). N1pw03 is the earlier N1pc with the NLF
fusion weight lowered from 0.5 to 0.3 (`val_nlf_traj.py --mode geolp0.3_5`); everything else is identical.

| file | content |
|---|---|
| `Dockerfile` | CUDA 12.6 / Ubuntu 22.04 / Python 3.11 image with the pipeline's "v2d" environment and the scripts |
| `apptainer.def` | the same image as an Apptainer / Singularity definition (what we built and tested on the cluster) |
| `requirements.txt` | pinned pip list of that environment (trimmed from the `pip freeze` of the run environment) |
| `reproduce.sh` | every stage from the raw Track 1 dataset to the parquet, with the flags of N1pw03 |
| `env_freeze/` | full `pip freeze` of the five environments the submission was produced with (reference) |

## What N1pw03 is

```
body    SAM 3D Body (dinov3, full mode) + SAM 3D Body (vith, body decoder) fused   -> work/body_ens
        sequence refinement v5, upright prior off, no depth target ("V-A")           -> work/body_vA
        assembled with Gaussian smoothing sigma_h = 3 frames, no finger penetration fix
objects SAM 3 / SAM 2.1 masks, MoGe-2 depth, SAM 3D Objects class meshes,
        render-and-compare trajectories v4 (hand-aware) and v5 (support-calibrated),
        composed per episode by recipes_v6.json ("objects_v6"), sigma_o = 12 frames,
        MoGe-to-body scale from the floor plane of body_vA (moge_to_body_ratio_floor_bA.json)
NLF     pelvis-depth fusion geolp0.3_5 (z = z_VA^0.7 z_NLF^0.3 along the pelvis ray, correction low-passed
        over 5 frames), the object moved with the body
PEN     object meshes carved at the hand contacts (CSG), scorer-exact penetration 0 on all 30 episodes
pack    kit eval_reconstruction.py
```

`reproduce.sh` lists the exact commands; every stage is one function. Places where the original flags
could not be recovered from the job files are marked `TODO(ambiguous)` in the script instead of being
guessed silently (three such places: the `fuse_body_ensemble.py` version, the body/ratio inputs of
`objects_v4`, and the NLF inference overrides).

## Running the container

The scripts resolve data, weights, the submission kit and outputs under one root, `/work/doreen071/v2d`
(`common.py` reads `V2D_ROOT`, but many scripts also hard-code that path). Bind the data tree there.

```bash
# Apptainer (tested)
apptainer exec --nv -B /path/to/v2d_root:/work/doreen071/v2d v2d_track1.sif \
    bash /opt/v2d-track1/container/reproduce.sh data intrinsics ...

# Docker
docker run --gpus all -it -v /path/to/v2d_root:/work/doreen071/v2d v2d-track1:track1 \
    container/reproduce.sh
```

Layout of the root (created by `reproduce.sh data` where public):

```
data/track_1/            Track 1 dataset (HF nvidia/video_to_data_challenge, public)
kit/v2d_submission_kit/  official submission kit (packer, scorer code used by the checks)
assets/mhr_model.pt      MHR v1.0.1 model (official release)      assets/vitdet/  ViTDet-H detector
third_party/             source checkouts of the model repos (below)
third_party_new/nlf_models/nlf_l_multi_0.3.2.patch6.torchscript
work/                    intermediate results         submissions_v7/   assembled episodes + parquet
```

## Environments

The image contains the Python 3.11 **v2d** environment (`requirements.txt`): fusion, sequence refinement,
object composition, assembly, NLF trajectory fusion, contact carving, the scorer-exact penetration check
and the kit packer. It is enough to re-run everything from the stored per-frame model outputs
(`work/body`, `work/body_vith_b`, `work/masks`, `work/depth`, `work/objects_v4`, `work/objects_v5`,
NLF raw output) to the parquet.

Deviations from the run environment: torch is the CUDA 12.6 wheel (the cluster environment used the CUDA
13 wheel of the same torch 2.14.1); scikit-image is a normal pip install (the cluster added it through
`PYTHONPATH`); the refinement scripts (`refine_body_sequence.py`, `refine_body_v3.py`,
`refine_body_v5.py`) were run from the sam3db environment on the cluster, but they import only numpy /
scipy / torch / cv2 and run unchanged in the v2d environment.

The four **model stages** need their own environments, which are not in the image (they pin
incompatible torch / CUDA builds and compiled extensions: detectron2, PyTorch3D, kaolin, spconv,
gsplat). Their complete pinned lists are in `env_freeze/`; the source checkouts are:

| environment | checkout (commit) | used by |
|---|---|---|
| sam3db (py3.11, torch 2.7.1+cu126) | `facebookresearch/sam-3d-body` @ `b5c765a`, detectron2 @ `a1ce2f9` | `run_sam3d_body.py`, `track_person_bbox.py`, `val_nlf_infer.py` (needs torch >= 2.4) |
| sam3 (py3.12, torch 2.10.0+cu128) | `facebookresearch/sam3` @ `0570b3a`, `facebookresearch/sam2` @ `2b90b9f`, `microsoft/MoGe` @ `74fbce0` | masks, intrinsics, depth, object frame preparation |
| sam3do (py3.11, torch 2.5.1+cu121) | `facebookresearch/sam-3d-objects` @ `f91db41`, PyTorch3D @ `75ebeea`, MoGe @ `74fbce0` | `reconstruct_object.py`, `object_candidates.py --stage recon`, `select_class_mesh.py`, `track_object_v2/v4/v5.py`, `finalize_objects_v2.py` |
| gvhmr (py3.10, torch 2.3.0+cu121) | `zju3dv/GVHMR` @ `ee960bb` | `run_gvhmr.py --stages preproc` (ViTPose-H keypoints only) |

Each is created the way its upstream README describes (conda env with the listed Python, the listed
torch build, `pip install -e` of the checkout), then pinned with the matching `env_freeze/*.txt`. The
scripts find the checkouts at `third_party/<name>` (`SAM3DB_ROOT`, `SAM3DO_ROOT`, `GVHMR_ROOT`
constants / environment variables in the scripts). `reproduce.sh` takes the interpreters through
`PY_SAM3DB`, `PY_SAM3`, `PY_SAM3DO`, `PY_GVHMR`.

## External weights, licences and gates

Nothing below is in the image or the repository. The gated Meta models are downloaded by the scripts
with `huggingface_hub` on first use; this needs the user's **own Hugging Face account** that has
accepted each model's licence on its model page (`hf auth login`, no token is stored anywhere in this
repository or printed by any script).

| weights | source | licence | gate |
|---|---|---|---|
| SAM 3 (`sam3.pt`) | HF `facebook/sam3` | SAM License (Meta) | HF gated (accept licence) |
| SAM 2.1 hiera-large | HF `facebook/sam2.1-hiera-large` | Apache-2.0 | none |
| Grounding DINO base | HF `IDEA-Research/grounding-dino-base` | Apache-2.0 | none |
| SAM 3D Body, DINOv3 backbone (`model.ckpt`) | HF `facebook/sam-3d-body-dinov3` | SAM 3D Body licence (Meta); the DINOv3 backbone inside is under the DINOv3 licence | HF gated (accept licence) |
| SAM 3D Body, ViT-H backbone (`model.ckpt`) | HF `facebook/sam-3d-body-vith` | SAM 3D Body licence (Meta) | HF gated (accept licence) |
| SAM 3D Objects (`checkpoints/*.ckpt`, `pipeline.yaml`) | HF `facebook/sam-3d-objects` -> `third_party/sam-3d-objects/checkpoints/hf/` | SAM 3D Objects licence (Meta) | HF gated (accept licence) |
| ViTDet-H person detector (`model_final_f05665.pkl`) | detectron2 model zoo, `dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/` | Apache-2.0 (detectron2) | none |
| MHR model + assets (`mhr_model.pt`, sha256 `352e271a...7377bc`) | `github.com/facebookresearch/MHR` release v1.0.1 `assets.zip` | Apache-2.0 (the official release is used, not the copy inside the SAM 3D Body checkpoints) | none |
| MoGe-2 ViT-L (normal) | HF `Ruicheng/moge-2-vitl-normal` | MIT (per the model card / MoGe repository) | none |
| NLF `nlf_l_multi_0.3.2.patch6.torchscript` | `github.com/isarandi/nlf` releases | code MIT; **weights: non-commercial research use** | none (GitHub release) |
| ViTPose-H (`vitpose-h-multi-coco.pth`) and HMR2 checkpoint loaded by GVHMR's preprocessing | GVHMR `inputs/checkpoints/` (links in the GVHMR README) | ViTPose Apache-2.0; GVHMR's own licence for its bundle | GVHMR's download form / drive |
| Track 1 dataset | HF dataset `nvidia/video_to_data_challenge`, folder `track_1/` | challenge data licence | none |

Not used by N1pw03 (only by other candidates in the ledger): WiLoR hands (`rolpotamias/WiLoR`, needs the
user's MANO registration), Sapiens2, GVHMR as a body source (only its ViTPose preprocessing is used),
SMPL-X (GVHMR predict stage, not run). The DINOv3 and ViT-H backbones are not downloaded separately: they
are inside the two SAM 3D Body checkpoints.

Per-frame weights of the body stage come from gated models, so a third party reproducing N1pw03 must
obtain the same gated access; everything after the model stages is deterministic given the stored
intermediates.

## Building the image

Docker (any machine with Docker; no GPU needed for the build):

```bash
cd v2d-track1
docker build -f container/Dockerfile -t v2d-track1:track1 .
```

Apptainer, as done on the cluster (no root, no `/etc/subuid` entry): Apptainer 1.4 falls back to a
root-mapped user namespace plus the `fakeroot` command. The build directory must be on local disk
(the build failed with `stat .: permission denied` when started from the Lustre `/work` tree; staging
the repository under `/tmp` and giving absolute paths works), and the cache / tmp directories should be
local too:

```bash
export APPTAINER_CACHEDIR=/tmp/$USER/apptainer_cache APPTAINER_TMPDIR=/tmp/$USER/apptainer_tmp
cp -r v2d-track1 /tmp/$USER/build && cd /tmp/$USER/build
apptainer build --fakeroot /tmp/$USER/v2d_track1.sif container/apptainer.def
```

The build downloads the base image, the apt packages and about 3 GB of pip wheels (torch + CUDA libs).
On the cluster's compute nodes the pip downloads ran at 0.2-3 MB/s, so the whole build took more than
one hour (a first attempt with a 1 h job limit was cut off while the SIF was being written); allow 2-3 h.

## Verification done on the cluster

Job logs in `/work/doreen071/v2d/jobs/container/` (`ctn_build_672693.log`, `test_in_image.sh`,
`test_n1pw03_*/verify.log`): the image was built on an H200 node (Apptainer 1.4.3, 60 min including
the downloads) and, with the data tree bound, the following were run inside it:

1. `python -c "import torch, torchvision, trimesh, skimage, fast_simplification, scipy, cv2, pandas, pyarrow"`
   and `torch.cuda.is_available()` with `--nv`;
2. `check_pen_exact.py --episodes 0` on the stored `N1pc/episodes` and `N1pw03/episodes` (scorer-exact PEN 0,
   0 points inside; the scorer path: MHR TorchScript forward, kit `v2dlb.mesh_budget`, winding-number test);
3. `assemble_episode.py` on episode 19 compared with the stored `A1/episodes_prefix` (max difference < 1e-5, mesh identical);
4. `val_nlf_traj.py --with-object` on episode 19 with `--mode geolp0.5_5` against the stored `N1p/episodes`
   and with `--mode geolp0.3_5` against the stored `N1pw03/episodes` (max difference < 1e-5);
5. `carve_object_contacts.py` on two short episodes (19, 14): the direct-dent phase and the CSG grid are
   identical to the stored run, and the re-carved meshes pass the scorer-exact PEN check with 0. The
   CSG step is not bit-reproducible (float32 winding numbers on the GPU), so the re-carved meshes differ
   from the submitted ones at the mm level (same face budget, surface change versus the uncarved mesh
   1.45 vs 1.47 mm on episode 19);
6. the kit packer on the stored `N1pc/episodes` and `N1pw03/episodes`, compared with the submitted parquets
   (identical row ids, commit URLs and values).

The heavy model stages were not re-run for the verification. The image's internal labels still carry
the earlier working name of the selection (N1pc); the content is the same for both submissions.

## Publishing the container (link for the submission form)

Cheapest route: upload the `.sif` to a public Hugging Face **model** repository (free, files up to
50 GB over Xet/LFS, direct download URL). From any machine that has the file and `pip install
huggingface_hub`:

```bash
hf auth login                                   # the user's own account; interactive, nothing printed
hf repo create <hf-user>/v2d-track1-container --type model
hf upload <hf-user>/v2d-track1-container /path/to/v2d_track1.sif v2d_track1.sif --repo-type model
# link for the form:
#   https://huggingface.co/<hf-user>/v2d-track1-container
# organisers can pull it with:
#   apptainer pull https://huggingface.co/<hf-user>/v2d-track1-container/resolve/main/v2d_track1.sif
```

Alternative, if a Docker image is preferred (needs Docker on the user's machine; the build is CPU-only
and takes about 10 minutes plus the download of the wheels):

```bash
cd v2d-track1
docker build -f container/Dockerfile -t <dockerhub-user>/v2d-track1:track1 .
docker login
docker push <dockerhub-user>/v2d-track1:track1
#   link: https://hub.docker.com/r/<dockerhub-user>/v2d-track1
```

Both routes can be done after the repository is made public; the image contains only the repository's
scripts and public pip packages (no data, no weights, no credentials).
