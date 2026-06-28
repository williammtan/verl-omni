# verl-omni Qwen3-TTS GRPO image: rollout (vLLM-Omni) + FSDP trainer + TTS reward backends.
#
# Follows the official verl-omni install (docs/start/install.md) via uv, then bakes the
# reward backends (faster-whisper / modelscope ERes2Net / UTMOSv2) and pre-fetches the
# UTMOSv2 weights so a broken download fails the build, not a scheduled 8-GPU job.
#
# Build/push (docker login as williammtan):
#   docker build --platform linux/amd64 -f deploy/Dockerfile -t williammtan/verl-omni-tts:vN .
#   docker push williammtan/verl-omni-tts:vN   # then bump the tag in deploy/qwen3-tts-grpo-job.yaml
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

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
# Dep-only image: install from PUBLIC upstream verl-omni so no private code is baked. The job
# clones the private feat/qwen-tts branch at runtime and PYTHONPATH-shadows this install (the
# new TTS modules live there). Mirrors the whiplash "deps baked, code cloned" pattern.
ARG VERL_OMNI_REF=main
RUN git clone --depth 1 --branch "${VERL_OMNI_REF}" https://github.com/verl-project/verl-omni.git /workspace/verl-omni

# Engine stack first (vllm + FA3 kernels), then rollout + train (vllm-omni, verl, verl-omni).
RUN cd /workspace/verl-omni \
    && uv venv --python 3.12 --seed \
    && . .venv/bin/activate \
    && uv pip install -e ".[gpu]" --torch-backend=cu130 \
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
    && uv pip install "pyarrow<21" "datasets>=2.16,<4"  # reward backends drag datasets down to 1.1.1
       # (no built-in parquet builder -> tries to fetch parquet.py from GitHub); pyarrow<21 keeps
       # PyExtensionType; numpy stays 2.x for vllm.
# NOTE: qwen-tts (the trainable actor) is intentionally NOT installed here — it requires
# transformers 4.57.x / huggingface_hub<1.0, which is incompatible with vLLM-Omni's hub>=1.0. The
# two run in SEPARATE images (disaggregated): this image = vLLM-Omni rollout; trainer = qwen-tts.

# Pre-fetch UTMOSv2 weights at build time (exercise the real path, not a best-effort echo).
RUN cd /workspace/verl-omni && . .venv/bin/activate \
    && python -c "import utmosv2; utmosv2.create_model(pretrained=True, device='cpu'); print('utmosv2 weights OK')"

# Build-time import smoke-check: a broken image fails the build, not a scheduled job.
# (Only public deps here; the branch's tts_quality module is imported at runtime from the clone.)
RUN cd /workspace/verl-omni && . .venv/bin/activate \
    && python -c "import torch, vllm, vllm_omni, verl, verl_omni, faster_whisper, modelscope, utmosv2, datasets; \
assert torch.version.cuda is not None, 'CPU torch built — need CUDA wheel (set --torch-backend=cuXXX)'; \
assert int(datasets.__version__.split('.')[0]) >= 2, f'datasets too old: {datasets.__version__}'; \
print('smoke OK', torch.__version__, 'datasets', datasets.__version__)"

ENV PATH="/workspace/verl-omni/.venv/bin:${PATH}"
WORKDIR /workspace/verl-omni
