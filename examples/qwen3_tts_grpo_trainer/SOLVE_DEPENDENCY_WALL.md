# Task for an LLM agent: make qwen-tts and vLLM-Omni coexist in one Python env (Qwen3-TTS + verl-omni)

## Mission (solve this no matter what)
Get a **single Python environment** in which **all of these are true at once**:
1. `import vllm_omni` succeeds (vLLM-Omni 0.22 — verl-omni's rollout backend).
2. `from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel` succeeds AND the model **generates
   correct, non-silent audio** for `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.
3. A teacher-forced **codec-0 log-prob forward + backward** runs on the loaded model (gradient flows
   to `talker.model`) — i.e. the model is trainable, not just inference.

If a single env is truly impossible, prove it rigorously and deliver the **least-disaggregated**
working alternative. "No matter what" = exhaust the version matrix, patch libraries, vendor/fork
code, or rebuild the model's transformers integration if needed. Do not stop at "use transformers 4.x".

## The exact conflict (verified empirically — reproduce before trusting)
- `qwen-tts==0.1.1` is the **only** PyPI release and **hard-pins `transformers==4.57.3`**.
- `transformers 4.57.x` **hard-fails at import** when `huggingface_hub>=1.0`:
  `dependency_versions_check.py` → `require_version_core` raises
  `ImportError: huggingface-hub>=0.34.0,<1.0 is required ... found 1.21.0`.
- `vllm_omni 0.22` **fails to import on `huggingface_hub<1.0`**: somewhere in its import chain a
  `huggingface_hub` `@strict` dataclass declares a field `import_name: str | None` (PEP-604 union),
  and the `<1.0` `@strict` validator rejects it:
  `StrictDataclassFieldValidationError: Unsupported type for field 'import_name': str | None`.
- ⇒ `hub>=1.0` kills transformers-4.57 (kills qwen-tts); `hub<1.0` kills vllm_omni. **No hub version
  satisfies both as-is.**
- On `transformers 5.x`, qwen-tts has import breaks (all patchable): delete the dead
  `from transformers.utils.generic import check_model_inputs` + its `@check_model_inputs()` decorator
  (tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py); `config.pad_token_id` → `getattr(...)`;
  `ROPE_INIT_FUNCTIONS['default']` KeyError → register a canonical default-RoPE init.
  **BUT** QwenLM/Qwen3-TTS **issue #237** + **PR #201 (CLOSED, unmerged)** report that on
  transformers 5.x the model emits **85–99% silent audio** ("fundamental incompatible changes in
  attention/logit computation … cannot be fixed with API-level patches, requires retraining").
  So merely making qwen-tts *import* on tf5 yields a **broken model**. Verify this silence claim
  yourself (generate + measure silence ratio) — do not assume.

## Attack angles, most promising first
**A. Make vLLM-Omni import on `hub<1.0` (KEEP qwen-tts on its correct 4.57.3).** ← try this first.
   The vllm_omni failure is a *single* validator bug: `huggingface_hub`'s `@strict` dataclass on
   `<1.0` doesn't support PEP-604 unions (`str | None`). If you **backport/monkeypatch that one
   validator** (or the specific dataclass) so `hub==0.36.2` accepts union types, vllm_omni may import
   on `transformers 4.57.3 + hub 0.36.2` — and then qwen-tts (correct audio) and vllm_omni coexist
   **with no model changes**. Investigate: install `transformers==4.57.3 huggingface_hub==0.36.2`,
   then `import vllm_omni`, find the exact `@strict` class/field, and patch
   `huggingface_hub.dataclasses` (or `sitecustomize` shim) to handle `types.UnionType`. Then check
   vllm_omni doesn't *also* need hub≥1.0 APIs at runtime (it may not — the strict validator may be
   the only blocker). This is the cleanest win.

**B. Make Qwen3-TTS produce correct audio on transformers 5.x** (then everything is hub≥1.0).
   This is the hard path PR #201 abandoned. Root-cause the attention/logit divergence vs 4.57.3
   (suspects from PR #201: causal-mask construction, `position_ids`, attn-impl dispatch,
   `_update_model_kwargs_for_generation`, RoPE int64-vs-float32 arange, `dynamic_rope_update`
   mutating `inv_freq`). Diff intermediate activations layer-by-layer between 4.57.3 and 5.x on the
   same input until silence is gone (target silence_ratio < 15%). Only pursue if A fails.

**C. Two isolated envs in one process** (last resort): run vllm_omni and qwen-tts in separate
   interpreters/subprocesses with an IPC bridge, or a custom transformers-5-native Qwen3-TTS modeling
   file. Heavy; document why A and B failed first.

## Constraints / context
- Target model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (12 Hz codec, talker_config: text_vocab 151936,
  codec vocab 3072, head_dim 128, rope_theta 1e6, mRoPE section [24,20,20]).
- Trainable scope: `talker.model` + `talker.codec_head` (codec-0); freeze `code_predictor`,
  `speaker_encoder`, `code2wav`. Reference: whiplash `codec0_logprobs` (teacher-forces codes with
  speaker-embed injected at position 6 + sub-codebook embeds 1–15).
- vLLM-Omni and verl-omni pins: `vllm==0.22.0`, `vllm-omni==0.22.0`, `verl==0.8.0`.
- Test on a CUDA GPU (an H200 is available); audio is 24 kHz.

## Acceptance test (must pass, single env, single process)
```python
import torch, numpy as np, soundfile as sf
import vllm_omni  # MUST import
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16)
# 1) generate audio, assert NOT silent
wavs, sr = m.generate_voice_clone(text="Hello, how can I help you today?", language="english",
                                  ref_audio="<ref.wav>", ref_text="", x_vector_only_mode=True)
w = np.asarray(wavs[0], np.float32); silence = float((np.abs(w) < 1e-3).mean())
assert silence < 0.15, f"SILENT AUDIO ({silence:.0%}) — tf-version model break not solved"
# 2) trainable codec-0 logprob backward (gradient flows)
#    (use whiplash codec0_logprobs or equivalent teacher-forced forward; assert grad norm > 0)
print("PASS: vllm_omni + correct trainable Qwen3-TTS in one env, silence", f"{silence:.1%}")
```
Deliver: the exact dependency set (versions), any patch/shim files (with a `sitecustomize.py` or pip
post-install), the passing test transcript, and a short writeup of which angle worked and why.
```
```
