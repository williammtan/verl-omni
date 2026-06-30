#!/usr/bin/env bash
# LOCAL single-GPU smoke (A100-40GB, in the depwall docker container). Mirrors run_qwen3_tts_grpo.sh
# but: (1) /work paths instead of /weka, (2) FSDP param+optimizer offload ON for 40GB, (3) the
# single-stage talker stage config, (4) a small batch (4 prompts x n=2 = 8 seqs) that fits 40GB
# through the actor update, (5) caps at 1 step. KL/ref is ON (the ref runs via verl manual offload —
# see deploy/patch_verl_ref_offload.py). NOTE: on a single 40GB GPU a 2nd step OOMs in the actor
# update (actor peak ~21GB + resident vLLM/CUDA contexts ~16GB) — that is the deferred memory/scaling
# work (more GPUs / a dedicated reward GPU / lower gpu_memory_utilization), not a correctness issue.
#
#   sudo docker exec depwall bash /workspace/verl-omni/examples/qwen3_tts_grpo_trainer/run_smoke_local.sh
set -uo pipefail
cd /workspace/verl-omni
unset PYTHONPATH
export PATH=/workspace/verl-omni/.venv/bin:$PATH

export VERL_TTS_REF_AUDIO=/work/ref.wav
export VERL_TTS_SPK_EMBED=/work/spk_embed.json
export VERL_TTS_MODEL_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-Base
export NCCL_IB_DISABLE=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export HYDRA_FULL_ERROR=1

python deploy/patch_vllm_omni_ar_rescue.py   || exit 22   # single-stage talker AR decode rescue
python deploy/patch_verl_unpad_fallback.py   || exit 23   # verl unpad->transformers fallback (no flash-attn)
python deploy/patch_verl_ref_offload.py      || exit 25   # ref policy via verl manual offload (KL/ref)
python -c "import verl_omni.models.transformers.qwen3_tts as q; print('patch module loaded')" || exit 21

TRAIN_FILE=/work/data/train.parquet \
VAL_FILE=/work/data/test.parquet \
bash examples/qwen3_tts_grpo_trainer/run_qwen3_tts_grpo.sh \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.total_training_steps=1 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path=/workspace/verl-omni/examples/qwen3_tts_grpo_trainer/qwen3_tts_stages_smoke.yaml \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.use_kl_loss=true \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.logger='[console]' trainer.save_freq=-1 trainer.test_freq=0 trainer.total_epochs=1
echo "===== STEP EXIT=$? ====="
