# Qwen3-TTS GRPO Trainer (POC)

GRPO post-training of **`Qwen/Qwen3-TTS-12Hz-1.7B-Base`** to improve spoken customer-service-rep
lines, scored by a multi-metric TTS reward on decoded audio. FSDP actor (talker) + vLLM-Omni AR
rollout (talker → code2wav), `adv_estimator: grpo`. Ported from the standalone `whiplash` recipe.

> **Status: POC, work-in-progress.** The reward + data + recipe scaffolding is in place and the
> reward math is unit-tested on CPU. Two integration points are flagged **PHASE-0/2 VALIDATE** in
> the configs and must be confirmed on a GPU node before a full run (see *Open items*).

## What is trained / frozen
- **Trainable:** the talker (codec_0 main path) + codec head.
- **Frozen:** sub-talker / `code_predictor` (codebooks 1–15), `speaker_encoder`, `code2wav`.
- Reward is computed on the **fully decoded 24 kHz waveform**; gradient flows only to the talker.
- Inverts the GSPO recipe (`examples/gspo_trainer/`), which trains the Thinker and *excludes* the
  talker.

## Reward (`verl_omni/utils/reward_score/tts_quality.py`, `reward_loop/reward_manager/tts.py`)
Weighted fusion (verl's GRPO estimator then z-scores within each prompt group):

| dim    | metric | model | weight |
|--------|--------|-------|--------|
| `text` | intelligibility (−CER) | faster-whisper large-v3 (int8) | 1.0 |
| `sim`  | speaker similarity vs the clone ref | 3D-Speaker ERes2Net (modelscope) | 1.0 |
| `mos`  | naturalness MOS | UTMOSv2 (git; weights baked in image) | 1.0 |
| `stab` | truncation/repetition/CER-outlier/synth-fail penalty | pure-python | 1.0 |

Backends degrade gracefully (failed backend → metric `None` → worst-case value). One fixed clone
voice (the SFT `maeneka` ref clip on weka) is used for every line, since the data has no speaker.

## Data
`data_process/tts_content_synth.py` extracts **every `assistant` message** from
`voice_14k_synthetic_04_27_clean.jsonl` (14,738 convos → 30,200 lines; ~25.5k after dedup) into
verl parquet, cloning the fixed `--ref_audio` voice for each line:

```bash
python examples/qwen3_tts_grpo_trainer/data_process/tts_content_synth.py \
  --ref_audio /weka/whiplash-sft/data/maeneka/ref.wav \
  --output_dir ~/data/tts_voice_synth
```

## Install (stack pins)
vLLM 0.22 / vLLM-Omni 0.22 / transformers 4.57.x / torch 2.11+**cu128** (cu128 keeps the whole
stack — and faster-whisper's ctranslate2 — on CUDA 12; cu130 breaks ctranslate2). See
`deploy/Dockerfile` for the exact baked image (`williammtan/verl-omni-tts`), which also pre-fetches
UTMOSv2 weights and pins `pyarrow<21`.

## Run
Local (after install + dataset prep):
```bash
bash examples/qwen3_tts_grpo_trainer/run_qwen3_tts_grpo.sh
```
Cluster (1×8 H200, `trainers` ns; clones code, preps data, trains):
```bash
kubectl apply -f deploy/qwen3-tts-grpo-job.yaml -n trainers
kubectl logs -n trainers -l app=will-tts-grpo -f
```

## Files
```
examples/qwen3_tts_grpo_trainer/
├── config/qwen3_tts_grpo.yaml      ← recipe (inherits verl ppo_trainer; adv_estimator grpo)
├── qwen3_tts_stages.yaml           ← vLLM-Omni talker→code2wav stage config  [PHASE-0 VALIDATE]
├── run_qwen3_tts_grpo.sh           ← launch (volatile overrides only)
├── data_process/tts_content_synth.py
└── README.md
verl_omni/utils/reward_score/tts_quality.py     ← reward math + RewardScorer (ported)
verl_omni/reward_loop/reward_manager/tts.py     ← TTSRewardManager (scores decoded audio)
verl_omni/models/transformers/qwen3_tts.py      ← talker automodel/freeze patch  [PHASE-2 VALIDATE]
deploy/{Dockerfile,qwen3-tts-grpo-job.yaml}
tests/test_tts_reward.py            ← CPU unit tests for the reward math
```

## Open items (must validate on GPU)
1. **Talker logprobs through the async server** (`qwen3_tts_stages.yaml`,
   `rollout.calculate_log_probs`): confirm vLLM-Omni surfaces processed codec-token logprobs for
   the AR talker. If not → Branch B: a custom rollout + actor-side teacher-forced codec_0 logprobs
   (port whiplash `codec0_logprobs`).
2. **Stage/arch names** in `qwen3_tts_stages.yaml` and **HF class names** in the model patch —
   confirm against the installed `vllm_omni.model_executor.models.qwen3_tts` and `qwen_tts`.
3. **On-policy sampler**: stage defaults `top_k=50, repetition_penalty=1.05` are neutralized
   (`top_k=-1, repetition_penalty=1.0`) so the importance ratio matches the full-softmax logprob.
4. **bf16 generation stability** (whiplash): watch for NaN/out-of-range codes; carry the
   safe-multinomial + code-clamp guards into the rollout if they surface.
