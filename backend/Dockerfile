# LipScribe demo backend for Coolify on the Alienware (DEPLOYMENT.md).
#
# Build context is the REPOSITORY ROOT, not backend/, because the backend imports the lipscribe
# package and scripts/train.py:
#   docker build -f backend/Dockerfile -t lipscribe-backend .
# backend/docker-compose.yml does the same, and Coolify builds from that file.
#
# The image holds code only. Model weights, the language model, the face landmarker and the
# pretrained encoders are mounted at run time (docker-compose.yml, "volumes"), so a code change
# rebuilds in minutes and the weights never go into git or the image.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# Match these to the Alienware's working .venv (DEPLOYMENT.md, step 2) and to its NVIDIA driver:
# cu128 wheels need driver 570 or newer (nvidia-smi shows the driver version).
# (The repository's requirements.txt says torch 2.14.1, which has no CUDA wheel; torchaudio 2.11.0
# pairs with torch 2.11.0, the newest on the cu128 index as of 26 Nov 2026.)
ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# libgl1/libglib2.0-0: OpenCV and MediaPipe; ffmpeg: clip extraction (inference.py);
# libsndfile1: soundfile; build-essential/cmake/zlib: building kenlm.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsndfile1 ffmpeg \
        build-essential cmake zlib1g-dev libbz2-dev liblzma-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch first, from the CUDA index; the constraints file stops later packages (silero-vad) from
# replacing it with a different build. The pip cache is a BuildKit cache mount: it is not part of
# the image, but wheels downloaded by a failed build (about 3 GB for CUDA torch) are reused by the
# next one, so a dropped connection does not mean starting the download again.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==${TORCH_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url ${TORCH_INDEX} \
    && pip freeze | grep -iE "^(torch|torchaudio)==" > /tmp/torch-constraints.txt

COPY backend/requirements.txt backend/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r backend/requirements.txt -c /tmp/torch-constraints.txt \
    && pip install --no-deps pyctcdecode==0.5.0
# (--no-deps: pyctcdecode declares numpy<2 but runs on numpy 2; see backend/requirements.txt)

# Code only (.dockerignore at the repository root keeps everything else out).
COPY pyproject.toml ./
COPY lipscribe/ lipscribe/
COPY scripts/ scripts/
COPY configs/ configs/
COPY frontend/ frontend/
COPY backend/ backend/

# Run as an unprivileged user; the mounted assets only need to be readable.
RUN useradd --create-home --uid 1000 lipscribe && mkdir -p /app/logs && chown lipscribe /app/logs
USER lipscribe

EXPOSE 8800
# /health answers as soon as the server is up (the models then take about 30 s; "loading": true).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8800/health || exit 1

CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8800", "--ws-max-size", "16777216"]
