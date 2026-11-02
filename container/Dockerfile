# Reproduction container for the V2D Challenge Track 1 submission N1pw03 (repo commit f2f17f7).
#
# Build from the repository root (the build context must contain scripts/ and container/):
#   docker build -f container/Dockerfile -t v2d-track1:track1 .
# Run with the data tree mounted at the path the scripts expect (see container/README.md):
#   docker run --gpus all -it -v /path/to/v2d_root:/work/doreen071/v2d v2d-track1:track1
#
# The image holds the Python 3.11 "v2d" environment (container/requirements.txt): fusion, sequence
# refinement, object composition, assembly, NLF trajectory fusion, contact carving, the scorer-exact
# penetration check and the kit packer. The four upstream model stages (SAM 3, SAM 3D Body, SAM 3D
# Objects + PyTorch3D, GVHMR/ViTPose) need their own environments and gated weights; their pinned
# package lists are in container/env_freeze/ and the setup is described in container/README.md.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git gnupg software-properties-common \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY container/requirements.txt /opt/requirements.txt
RUN pip install -r /opt/requirements.txt

# Pipeline scripts (snapshot of the repository at build time) and the reproduction recipe.
COPY scripts /opt/v2d-track1/scripts
COPY container /opt/v2d-track1/container

# The scripts resolve data, weights, the submission kit and outputs under this root (common.py:
# V2D_ROOT; several scripts also hard-code the same path). Mount the data tree there.
ENV V2D_ROOT=/work/doreen071/v2d
RUN mkdir -p /work/doreen071/v2d

WORKDIR /opt/v2d-track1/scripts
ENTRYPOINT ["/bin/bash"]
