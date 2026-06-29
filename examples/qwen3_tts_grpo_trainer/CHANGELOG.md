# Qwen3-TTS GRPO on verl-omni — Changelog (issues & fixes)

Goal: GRPO post-train `Qwen/Qwen3-TTS-12Hz-1.7B-Base` on the assistant-traces data, scored by a
multi-metric TTS reward, integrated with verl-omni.

> **TL;DR / read this first.** The integration has **one hard constraint** that dictates everything:
> Qwen3-TTS only produces correct audio on **transformers 4.x**, while verl-omni's rollout
> (`vllm_omni`) requires the **transformers-5 / huggingface_hub≥1.0** world. They **cannot share one
> Python process**. So verl-omni's *native colocated* actor+rollout is impossible, and the only
> correct architecture is **disaggregation** (qwen-tts trainer on 4.57.3 + a separate rollout
> server). Most of the work below was (a) discovering that constraint and (b) environment plumbing.
> The actual training is ~150 lines of recipe + a reward port; everything else was yak-shaving.

---

## 0. The core finding (the only thing that really matters)

| # | Issue | Resolution |
|---|---|---|
| C1 | verl-omni's actor loads the policy as a transformers-native `AutoModelForCausalLM`; **Qwen3-TTS has no such model** (`KeyError: 'qwen3_tts'`). | Only `qwen-tts` (PyPI) is a trainable impl; `vllm_omni`'s talker is **inference-only** (vLLM `Qwen3Model`/`ParallelLMHead`, not FSDP-trainable). |
| C2 | `qwen-tts==0.1.1` pins `transformers==4.57.3`, which **hard-fails import** on `huggingface_hub≥1.0`. `vllm_omni 0.22` **requires** `hub≥1.0`. | **Mutually exclusive in one env** (proven in-container). |
| C3 | Tried porting qwen-tts to transformers 5 (delete `check_model_inputs`, fix `pad_token_id`, patch RoPE `'default'`). | **Dead end** — QwenLM/Qwen3-TTS **issue #237 + PR #201 (closed, unmerged)** confirm transformers 5.x yields **85–99% silent audio** ("attention/logit incompatibility, cannot be fixed with API patches, requires retraining"). Ecosystem pins `<5`. |
| C4 | Therefore: native colocated verl-omni + qwen-tts is **impossible**. | **Disaggregate**: qwen-tts trainer (4.57.3) + sglang rollout server (own env) + optional reward server. This works and produces correct audio (CER≈0, MOS≈3.0, SIM≈0.68). |

Everything else below is plumbing to make that disaggregated path run.

---

## 1. Image / dependency stack (building `williammtan/verl-omni-tts`)

| # | Issue | Fix |
|---|---|---|
| 1 | `docker push` blocked (auto-mode: image bundled the private repo). | Rebuilt **dep-only** image (install from public upstream; runtime `git clone` the branch). |
| 2 | `vllm._C ImportError: libcudart.so.13` at runtime. | vLLM 0.22 needs **CUDA 13** → base `nvidia/cuda:12.8.1` → **`13.0.2`**. (Build-time smoke check missed it: vLLM only loads `_C` when a GPU is visible, absent at build.) |
| 3 | Built image shipped **CPU torch** (`2.11.0+cpu`). | `--torch-backend=auto` picks CPU on a GPU-less builder → pin **`--torch-backend=cu130`** + smoke-assert `torch.version.cuda`. |
| 4 | `datasets` downgraded to **1.1.1** (tried to fetch `parquet.py` from GitHub). | Reward backends dragged it down → pin **`datasets>=2.16,<4`**. |
| 5 | `pyarrow` too new (`PyExtensionType` removed). | Pin **`pyarrow<21`**. |
| 6 | faster-whisper/`ctranslate2` is a **CUDA-12** binary; clashes with the CUDA-13 stack. | Run faster-whisper on **CPU** in the (verl-omni) image (`whisper_device="cpu"`); on the reward server use the CUDA-12 whiplash image so whisper runs on GPU. |

> Note: this whole image turned out **not to be needed for training** — the trainer runs on the
> existing `williammtan/whiplash-grpo:v3` (CUDA-12) image. The verl-omni image is only used for the
> reward server. (Overcomplication: I built a full verl-omni image before realizing disagg meant the
> trainer should just reuse the proven whiplash image.)

## 2. Recipe / runtime (the cluster job)

| # | Issue | Fix |
|---|---|---|
| 7 | Job `rm -rf`'d its own working dir (`CODE_DIR` == the baked venv path). | Separate `CODE_DIR=/workspace/code`; `PYTHONPATH` shadow. |
| 8 | `tokenizer.apply_chat_template` crash — Qwen3-TTS has **no chat template**. | Inject a **passthrough chat template** (emits the raw text) via the model patch. |
| 9 | verl dataset filter **silently skipped every sample** ("Error processing… skipping"). | Set `filter_overlong_prompts: false` (CX lines are short; the filter hid the real error). |
| 10 | Then the `KeyError: 'qwen3_tts'` model-load failure → see **C1–C4** above. | Pivot to disaggregation. |

## 3. Disaggregated run (qwen-tts trainer + sglang rollout)

| # | Issue | Fix |
|---|---|---|
| 11 | `reward_sim = 0` — `No module named 'addict'` (modelscope ERes2Net dep missing). | Runtime `pip install addict`. |
| 12 | `reward_sim` still 0 — `No module named 'datasets'` in the speaker pipeline. | Runtime `pip install datasets simplejson sortedcontainers`. |
| 13 | ✅ **1-rank disagg works** — all 4 reward dims live, weight-sync each step, ~85–130s/step. | (validated) |

## 4. Multi-GPU attempts (push the training rate)

| # | Issue | Fix / outcome |
|---|---|---|
| 14 | **In-process 8-GPU DDP** (no sglang): HF `generate` ~50–100× slower than sglang → 105 min, **0 steps**. | Dead end — sglang rollout is required. |
| 15 | **2-rank DDP disagg**: **deadlock** — both ranks racing to load reward models in-process. | Root cause = in-process reward → build a **reward server** (below). |
| 16 | Reward server crashed: `python -m verl_omni…` triggers the heavy `verl_omni/__init__` (imports `vllm_omni`, absent on whiplash image). | Run the server **by file path** (it loads `tts_quality` by path, no package init). |
| 17 | ✅ Reward offload **fixes the DDP deadlock** — 2-rank DDP now lands steps, reward scored remotely. | (validated) |
| 18 | But DDP is **slower** than 1-rank (0.52 vs 0.64 rollouts/s). Profiling: all tiers idle (sglang 0–35%, trainer 0%, reward 3%) → step is **latency-bound, not compute-bound**. | DDP parallelizes trainer compute, which wasn't the bottleneck. |
| 19 | Bigger batch (128 rollouts/step) → **CUDA OOM** (140 GB). | whiplash runs the **whole rollout group in one forward (no microbatching)** → batch capped by worst-case seq len. Reverted to 64. |
| 20 | `sglang weight-sync every step` re-copies model+tokenizer to disk ("Fetching 13 files"). | Env-driven `SGLANG_SYNC_EVERY=4` → modest gain (0.47→0.52). |
| 21 | **Conclusion:** multi-GPU can't help this latency-bound workload **until** the trainer can hold a large per-rank batch — which needs **microbatching the logprob forward + grad accumulation** (verl-omni's `ppo_micro_batch` pattern). **Not implemented.** | Open. |

---

## What was actually delivered (committed on `feat/qwen-tts`)

- `verl_omni/utils/reward_score/tts_quality.py` — TTS reward math + `RewardScorer` (ported), 12 CPU tests.
- `verl_omni/utils/reward_score/tts_reward_server.py` — HTTP reward server (verl-omni async-scorer style).
- `verl_omni/reward_loop/reward_manager/tts.py` — verl-omni reward manager (for the *native* path, unused since native is blocked).
- `verl_omni/models/transformers/qwen3_tts.py` — model patch (passthrough chat template + automodel hints).
- `examples/qwen3_tts_grpo_trainer/` — config, data prep (assistant traces → prompts), run script, README.
- `deploy/` — Dockerfile, disagg sglang + trainer manifests, reward-server + DDP-trainer manifests, patchers.

## Minimal path in hindsight (where it got overcomplicated)

If doing this again, the **short path** is:

1. **Accept up front:** Qwen3-TTS = transformers 4.x only (issue #237); verl-omni rollout = transformers 5 → **disaggregate, don't fight it.** (This was discovered late, after building a native recipe + a verl-omni image that the trainer doesn't even use.)
2. **Trainer = the existing `whiplash-grpo:v3` image** + `--sglang-url`. No new image, no transformers/hub/datasets/pyarrow/cuda yak-shaving (items 1–6 were for an image only the reward server needs).
3. **Bake the 4 missing pip deps** (`addict datasets simplejson sortedcontainers`) into the trainer image once, instead of runtime-installing (items 11–12).
4. **Stay 1-rank.** For this latency-bound workload, single-GPU is fastest; multi-GPU (items 14–21) needs microbatching to pay off — skip it unless throughput is the explicit goal.
5. verl-omni's contribution is then just: the **reward port** + **reward server** + **recipe/data scaffolding** — the rest of verl-omni (FSDP actor, vLLM-Omni rollout, native reward manager) **cannot be used** because of the transformers 4-vs-5 wall.

Net: the genuinely necessary work was ~1 reward module + 1 data script + the disagg manifests + 4 pip deps. The image build, tf5 port, and multi-GPU/DDP exploration were detours driven by trying to use verl-omni *natively* before confirming that's impossible.
