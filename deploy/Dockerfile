# verl-omni Qwen3-TTS GRPO image: rollout (vLLM-Omni) + FSDP trainer + TTS reward backends.
#
# Follows the official verl-omni install (docs/start/install.md) via uv, then bakes the
# reward backends (faster-whisper / modelscope ERes2Net / UTMOSv2) and pre-fetches the
# UTMOSv2 weights so a broken download fails the build, not a scheduled 8-GPU job.
#
# Build/push (docker login as williammtan):
#   docker build --platform linux/amd64 -f deploy/Dockerfile -t williammtan/verl-omni-tts:vN .
#   docker push williammtan/verl-omni-tts:vN   # then bump the tag in deploy/qwen3-tts-grpo-job.yaml
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

# wget: UTMOSv2 shells out to it for its weight download (missing -> silent retry loop).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget ca-certificates build-essential ffmpeg libsndfile1 sox \
        python3.12 python3.12-dev python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# uv (the verl-omni-sanctioned installer).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace
COPY . /workspace/verl-omni

# Engine stack first (vllm + FA3 kernels), then rollout + train (vllm-omni, verl, verl-omni).
RUN cd /workspace/verl-omni \
    && uv venv --python 3.12 --seed \
    && . .venv/bin/activate \
    && uv pip install -e ".[gpu]" --torch-backend=cu128 \
    && uv pip install -e ".[vllm-omni,train,dev]" \
    && uv pip install pydub

# TTS reward backends (numpy<2 kept to satisfy verl/verl-omni; vllm pulls 2.x — codepaths used
# here are 2.x-compatible, mirrors the GSPO note).
RUN cd /workspace/verl-omni && . .venv/bin/activate \
    && uv pip install \
        faster-whisper \
        modelscope \
        funasr \
        librosa \
        soundfile \
        "git+https://github.com/sarulab-speech/UTMOSv2.git" \
    && uv pip install "pyarrow<21"  # verl/datasets need pyarrow.PyExtensionType (removed in 21); keep numpy 2.x for vllm

# Pre-fetch UTMOSv2 weights at build time (exercise the real path, not a best-effort echo).
RUN cd /workspace/verl-omni && . .venv/bin/activate \
    && python -c "import utmosv2; utmosv2.create_model(pretrained=True, device='cpu'); print('utmosv2 weights OK')"

# Build-time import smoke-check: a broken image fails the build, not a scheduled job.
RUN cd /workspace/verl-omni && . .venv/bin/activate \
    && python -c "import torch, vllm, vllm_omni, verl, verl_omni; \
from verl_omni.utils.reward_score import tts_quality; \
print('smoke OK', torch.__version__)"

ENV PATH="/workspace/verl-omni/.venv/bin:${PATH}"
WORKDIR /workspace/verl-omni
