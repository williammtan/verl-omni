#!/usr/bin/env bash
# Fast iteration: just the main_ppo step (branch already checked out, env + data already prepped).
# I cp the fixed verl_omni file(s) over /workspace/verl-omni before each run.
set -uo pipefail
cd /workspace/verl-omni
unset PYTHONPATH
export VERL_TTS_REF_AUDIO=/weka/whiplash-sft/data/maeneka/ref.wav
export VERL_TTS_SPK_EMBED=/weka/whiplash-grpo/tts/spk_embed.json
export VERL_TTS_MODEL_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-Base   # reward-side code2wav (speech_tokenizer) decode
export NCCL_IB_DISABLE=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export HYDRA_FULL_ERROR=1
python deploy/patch_vllm_omni_ar_rescue.py || exit 22  # single-stage talker AR decode rescue
python deploy/patch_verl_unpad_fallback.py || exit 23   # verl unpad->transformers fallback (no flash-attn)
python deploy/patch_verl_ref_offload.py || exit 25      # ref policy via verl manual offload (KL/ref)
python -c "import verl_omni.models.transformers.qwen3_tts as q; print('patch module loaded')" || exit 21
TRAIN_FILE=/weka/whiplash-grpo/tts/verl_smoke/train.parquet \
VAL_FILE=/weka/whiplash-grpo/tts/verl_smoke/test.parquet \
bash examples/qwen3_tts_grpo_trainer/run_qwen3_tts_grpo.sh \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path=/workspace/qwen3_tts_stages_smoke.yaml \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.use_kl_loss=true \
  data.train_batch_size=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.logger='[console]' trainer.save_freq=-1 trainer.test_freq=0 trainer.total_epochs=1
echo "===== STEP EXIT=$? ====="
