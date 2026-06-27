#!/usr/bin/env bash
# Qwen3-TTS GRPO training (FSDP talker actor + vLLM-Omni AR talker→code2wav rollout,
# reward on decoded audio). Hardware: 1 node × 8 H200.
#
# Recipe config: config/qwen3_tts_grpo.yaml (inherits verl ppo_trainer). Only volatile values
# (paths, GPU/node counts) are set here.
set -x

export NCCL_IB_DISABLE=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Load verl_omni on the driver (registers the vllm_omni rollout adapter + TTSRewardManager) and
# the Qwen3-TTS model patch (automodel/processor + talker-only freeze hints). Workers also load
# the model patch via external_lib in the launch args.
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni.models.transformers.qwen3_tts

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-TTS-12Hz-1.7B-Base"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/tts_voice_synth/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/tts_voice_synth/test.parquet"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_CONFIG="${SCRIPT_DIR}/qwen3_tts_stages.yaml"

python3 -m verl.trainer.main_ppo \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name=qwen3_tts_grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_tts \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="${STAGE_CONFIG}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    "$@"
