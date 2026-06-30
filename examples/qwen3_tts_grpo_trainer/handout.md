# Handout — Native Qwen3-TTS Talker GRPO on verl-omni (continue here)

**Branch:** `fix/vllm-omni-qwen-tts-deps` (pushed to `origin`, base = `main`). Last commit `3b17eb0`.
**Pod:** `will-depwall` in namespace `trainers` (1× H200, `deploy/depwall-pod.yaml`). Image `williammtan/verl-omni-tts:v4`.

## TL;DR

The native colocated path WORKS end-to-end and produces a **real learning signal**: vLLM-Omni AR talker
rollout (NOT disagg sglang) + an FSDP actor that teacher-forces codec-0 log-probs. Last smoke (1 GPU, 2 steps):
`response_length≈120`, `critic/rewards/mean ≈ +2–3` (clean, matches the proven disagg run), `grad_norm 8–12`,
advantages have variance. **One thing is still open: the rollout↔actor on-policy match.**

## The ONE open problem: on-policy logprob match

`training/rollout_probs_diff_mean ≈ 0.64`, `rollout_actor_probs_pearson_corr ≈ 0.08` over the real utterance.
(The earlier "0.22" was a **garbage-tail averaging artifact** — before eos-stop the response was 2048 tokens of
mostly junk; over the real ~120 tokens the true gap is ~0.64.) This means the FSDP actor's recomputed codec-0
logprobs don't yet match how vLLM-Omni actually generated the tokens, so the policy gradient is off-policy.

**Four layout fixes are already applied** (all confirmed live), and they are correct as far as they go:
1. **non_streaming_mode** — rollout request sets `non_streaming_mode:[True]` so the talker lays full text in
   prefill (the layout the actor reconstructs), not streaming. (`vllm_omni_async_server.py` `_tts_voice_clone_request`)
2. **placeholder length** = `assistant_len+2` (no wedged `tts_pad` → RoPE matches). (same fn)
3. **text_projection** — the actor applies `talker.text_projection` (a learned ResizeMLP) on every text embed,
   because vLLM-Omni generation does (whiplash skips it; it never compares rollout-vs-actor since it uses
   `old_logp=detach()`). (`qwen3_tts_forward.py` `assemble_talker_embeddings`)
4. **codes +1 alignment** — the accumulated `(T,16)` codes lead `token_ids` by exactly one frame
   (`codes[k+1,0]==token_ids[k]` at **100%** across all dumped samples — placeholder + a codec_bos transition
   frame). Now aligned by best-offset matching in `_process_output`.

**What's likely left:** a residual vLLM-Omni-vs-whiplash *generation-layout* difference the actor reconstruction
doesn't capture (the actor forward is bit-exact vs whiplash — G7/G8 pass — but whiplash is NOT the oracle here;
the vLLM-Omni rollout is). Candidates to check with the per-token diagnostic below: bf16 precision; the
sub-codebook teacher-forcing offset (does the actor condition frame k on codes[k][1:16] or codes[k-1][1:16]?);
speaker x-vector path; RoPE/position derivation; the codec_bos/eos boundary handling.

## The per-token diagnostic (IN FLIGHT — use this first)

`deploy/patch_verl_probdump.py` patches verl's `utils/debug/metrics.py:calculate_debug_metrics` to dump, for
sample 0, the **already-aligned** per-token `actor_logp` vs `rollout_logp` (+ `response_mask`, `responses`) to
`/weka/whiplash-grpo/tts/dump/probdump_0.pt` when `VERL_TTS_PROBDUMP=1`. No batch reconstruction needed — verl
aligns them. The runner already wires the patch + flag.

Run it, then analyze `probdump_0.pt`:
```python
import torch
d = torch.load("/weka/whiplash-grpo/tts/dump/probdump_0.pt", weights_only=False)
a, r, m = d["actor_logp"], d["rollout_logp"], d["response_mask"].bool()
a, r = a[m], r[m]
print("pearson", torch.corrcoef(torch.stack([a, r]))[0,1].item())
print("mean|diff|", (a-r).abs().mean().item(), "max", (a-r).abs().max().item())
for k in range(min(20, len(a))): print(k, round(a[k].item(),3), round(r[k].item(),3))
```
**Read the shape of the divergence:** diverges from token 0 → prefix/context (text/speaker/control-prefix)
mismatch. Diverges gradually → accumulating per-frame error (sub-codebook offset / RoPE). Spikes at specific
tokens → those token types (eos/bos/special). That localizes the fix.

Also useful: the rich rollout dump `sample_{0..3}.pt` (token_ids, codes, rollout log_probs, text) written by
`_process_output` under `VERL_TTS_DUMP=1`. `examples/qwen3_tts_grpo_trainer/` and the scratchpad `analyze_dump.py`
show how to read it (the +1 offset + eos analysis came from there).

## How to run (READ THIS — pod setup is NOT in the image)

The qwen-tts ↔ vLLM-Omni env (transformers 4.57.3 + qwen-tts + the hub union shim) is **NOT baked into the
image**. A fresh pod has tf 5.8.1 / hub 1.21.0 / no qwen-tts. After any pod (re)create:

```bash
# 0. (re)create pod if needed:  kubectl apply -f deploy/depwall-pod.yaml
# 1. restore the dep-wall env (idempotent, ~3-5 min, needs PyPI):
kubectl exec will-depwall -n trainers -- bash /workspace/verl-omni/deploy/setup_depwall_env.sh
# 2. SYNC THE WHOLE verl_omni PACKAGE in place — the image's verl_omni is OLDER than HEAD, and
#    `kubectl cp <dir>` NESTS into an existing dir. Use tar-pipe:
tar -C . -czf - verl_omni | kubectl exec -i will-depwall -n trainers -- tar -C /workspace/verl-omni -xzf -
#    also sync the recipe + deploy if missing (image lacks them):
#    kubectl cp examples/qwen3_tts_grpo_trainer will-depwall:/workspace/verl-omni/examples/qwen3_tts_grpo_trainer
#    kubectl cp deploy will-depwall:/workspace/verl-omni/deploy   (only if /workspace/verl-omni/deploy absent)
# 3. put the smoke runner + single-stage config where the runner expects them:
kubectl cp examples/qwen3_tts_grpo_trainer/run_smoke_step.sh        will-depwall:/workspace/run_smoke_step.sh
kubectl cp examples/qwen3_tts_grpo_trainer/qwen3_tts_stages_smoke.yaml will-depwall:/workspace/qwen3_tts_stages_smoke.yaml
# 4. run the 2-step smoke (it runs the 3 patches, exports the VERL_TTS_* flags):
kubectl exec will-depwall -n trainers -- bash /workspace/run_smoke_step.sh
```
The pod is **`restartPolicy: Never`** and `/workspace` is **ephemeral** (only `/weka` + `/dev/shm` mount), so an
eviction wipes the env + synced files — redo steps 1–3. Watch for eviction (`kubectl get events -n trainers`).

Smoke config: 1 GPU, `tp=1`, `adv_estimator=grpo`, `use_kl_loss=false` (ref/KL disabled — see below),
`train_batch_size=8`, `rollout.n=4`, `gpu_memory_utilization 0.28` (talker stage). Env: `VERL_TTS_REF_AUDIO`,
`VERL_TTS_SPK_EMBED=/weka/whiplash-grpo/tts/spk_embed.json`, `VERL_TTS_MODEL_PATH`, `VERL_TTS_DEBUG/DUMP/PROBDUMP`.

## Key files

- `verl_omni/models/transformers/qwen3_tts_forward.py` — the actor forward (the codec-0 math). `tts_actor_logits`
  is the pure heart; `assemble_talker_embeddings` builds the talker input (text_projection applied here).
- `verl_omni/models/transformers/qwen3_tts.py` — installs the forward; AutoConfig.register; talker-only freeze;
  agent-loop `multi_modal_inputs` patch that surfaces sub-codebooks + text to the actor; speaker x-vector load.
- `verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py` — `_tts_voice_clone_request` (non_streaming,
  placeholder, eos-stop), `_run_generation` (codes accumulation), `_process_output` (codes↔token_ids alignment,
  the VERL_TTS_DUMP dump).
- `verl_omni/reward_loop/reward_manager/tts.py` — reward-side code2wav decode + eos-trim + CER/sim/MOS/stability.
- `deploy/setup_depwall_env.sh`, `deploy/hf_hub_unionfix.{py,pth}`, `deploy/patch_*.py` — env + runtime patches.
- `examples/qwen3_tts_grpo_trainer/{run_smoke_step.sh,qwen3_tts_stages_smoke.yaml,depwall_g7_alignment.py,
  depwall_g8_actor.py,SOLVED_DEPENDENCY_WALL.md}` — runner, smoke stage, offline gates, dep-wall writeup.
- Reference (read-only): `whiplash/train/grpo_trainer.py:250-279 codec0_logprobs` (the ground-truth forward).
  vLLM-Omni source for the generation layout: `/tmp/vllm-omni-main/vllm_omni/model_executor/models/qwen3_tts/`
  (`prompt_embeds_builder.py` build_prompt_embeds + estimate_prompt_len; `qwen3_tts_talker.py` decode).

## Deferred (after on-policy is closed)

- **ref/KL disabled** (`use_kl_loss=false`): the all-frozen ref model isn't summoned to GPU by FSDP (device
  mismatch at `codec_embedding`). Fix the frozen-param summon to re-enable the KL term.
- **Reward GPU OOM**: ~27 caught CUDA-OOM warnings on the 0.28-util single GPU (rewards still land clean). Give
  the reward its own GPU when scaling.
- **Scale to 8 GPUs** + full recipe (`train_batch_size=32`, `n=8`, `adv_estimator=gdpo`, full data). The reward
  GDPO per-dim keys (`rw_text/rw_sim/rw_mos/rw_stab`) + `gdpo_reward_keys` are already in the config.
- Remove the `VERL_TTS_DEBUG/DUMP/PROBDUMP` instrumentation once on-policy is verified.

## Validation gates (from the plan)

G2 is THE gate: `actor/ppo_kl ≲ 1e-3` and `rollout_probs_diff_mean → ~0` at step 1. Currently `ppo_kl=0` (old==new,
single forward) but `rollout_probs_diff=0.64` — the on-policy work above is exactly closing this. G7/G8 (offline,
actor==whiplash) already PASS, so the actor math is correct; the gap is rollout-layout, not actor-math.
