# SOLVED: qwen-tts + vLLM-Omni coexist in ONE Python env (angle A)

**Result: Angle A works.** `vllm-omni 0.22` imports AND **runs** alongside `qwen-tts 0.1.1`
(transformers 4.57.3) in a **single env** — with **zero model changes** and a **one-line shim**.
Proven end-to-end:
- `import vllm_omni` succeeds on transformers 4.57.3 + hub 0.36.2 (the import wall, broken by the shim);
- the **full vllm_omni offline rollout** (2-stage talker→code2wav `Omni`) serves Qwen3-TTS and emits
  **correct audio** (whisper CER = 0.000) — no hub≥1.0/tf5 *runtime* API gap, shim active in vllm's
  spawned workers too;
- the native `qwen-tts` model generates **correct, intelligible audio** (whisper CER = 0.000); and
- a teacher-forced **codec-0 log-prob backward** flows gradient into `talker.model` + `talker.codec_head`.

This overturns the earlier "native colocation is impossible → disaggregation only" conclusion, which
had been reached **without actually trying the validator shim**.

Verified on an H200 in `williammtan/verl-omni-tts:v4` (vllm 0.22 + vllm-omni 0.22 + torch
2.11+cu130), downgrading only transformers + huggingface_hub and adding the shim + qwen-tts.

---

## Root cause (reproduced empirically, not assumed)

The conflict is **not** the transformers version. `vllm-omni 0.22`'s own metadata is
`transformers !=5.0.*..!=5.5.0, <5.9.0, >=4.56.0` and it declares **no** `huggingface_hub` pin;
`vllm 0.22` is `transformers >=4.56.0` (same 5.x exclusions). **`transformers==4.57.3` (qwen-tts's
hard pin) satisfies both.** vllm 0.22 even keeps an explicit "Transformers v4" code path
(prints a deprecation note, then works).

The actual blocker is a **single validator bug**, one hop away:

```
import vllm_omni
  → vllm → transformers 4.57.3 modeling_utils
    → transformers/integrations/hub_kernels.py → `from kernels import …`
      → kernels/deps.py  (module-level: _DEPENDENCY_DATA = DependencyData.from_dict(...))
        → instantiates  @strict @dataclass PythonPackage(pkg: str, import_name: str | None)
          → huggingface_hub 0.36.2 strict validator:
             StrictDataclassFieldValidationError: Unsupported type for field 'import_name': str | None
```

`huggingface_hub/dataclasses.py`'s `type_validator` dispatches union types via
`_BASIC_TYPE_VALIDATORS`, which on hub `<1.3.0` registers only **`typing.Union`** — not
**`types.UnionType`**, the origin of PEP-604 `X | Y`. So `str | None` falls through to
`raise TypeError("Unsupported type …")`. (`Optional[str]` would have worked; `str | None` does
not.) huggingface_hub fixed this in **v1.3.0** with exactly:
`_BASIC_TYPE_VALIDATORS[types.UnionType] = _validate_union`.

Because transformers 4.57.3 forces hub `<1.0`, and `kernels 0.14.1` (pulled in by the vllm stack)
ships the PEP-604 field, the two collide. **No model code is involved** — it's purely a
type-annotation parser gap in an old hub.

## The fix — backport the hub v1.3.0 one-liner (angle A)

`deploy/hf_hub_unionfix.py` registers `types.UnionType` against hub's existing `_validate_union`
(which already handles `get_args(str | None) == (str, NoneType)` correctly). `type_validator`
reads the module-global dict by name on every call, so a single `setdefault` is sufficient — no
function is replaced:

```python
import types
from huggingface_hub import dataclasses as _hf_dc
if hasattr(types, "UnionType"):                                   # Python 3.10+
    _hf_dc._BASIC_TYPE_VALIDATORS.setdefault(types.UnionType, _hf_dc._validate_union)
```

**Delivery mechanism = `.pth`, not sitecustomize.** Validation fires at the *first instantiation*
of a strict dataclass (here, at `import kernels`), so the patch must run before any import. A
`sitecustomize.py` is shadowed in this image by the system `/usr/lib/python3.12/sitecustomize.py`
(Python loads only the first one on `sys.path`). `deploy/hf_hub_unionfix.pth` sidesteps that:
`site.py` executes `import`-prefixed lines in **every** `.pth` it processes, in the main process
**and every vllm worker subprocess**, at interpreter startup. Both files are dropped into the
venv's `site-packages/`.

### Install (post-install step, or bake into the image)

```bash
SP=$(python -c "import site;print(site.getsitepackages()[0])")
cp deploy/hf_hub_unionfix.py  "$SP"/hf_hub_unionfix.py
cp deploy/hf_hub_unionfix.pth "$SP"/hf_hub_unionfix.pth   # -> `import hf_hub_unionfix; hf_hub_unionfix.apply()`
```

`deploy/sitecustomize.py` is also provided for envs whose `sitecustomize` is *not* shadowed, but
the `.pth` is the load-bearing one here.

## Exact dependency set (the solved env)

Start from the vllm-omni stack, downgrade transformers (which pulls hub<1.0), add qwen-tts:

```
# unchanged from williammtan/verl-omni-tts:v4
torch==2.11.0+cu130   torchaudio==2.11.0+cu130   vllm==0.22.0   vllm-omni==0.22.0
x-transformers==2.23.1   numpy==1.26.4   tokenizers==0.22.2   kernels==0.14.1   accelerate==1.12.0

# changed / added for coexistence
transformers==4.57.3            # was 5.8.1  (qwen-tts hard pin; inside vllm-omni's >=4.56,<5.9)
huggingface-hub==0.36.2         # was 1.21.0 (pulled by transformers 4.57.3; <1.0)
qwen-tts==0.1.1   (--no-deps)   # + sox onnxruntime einops librosa soundfile (most already in image)
+ hf_hub_unionfix.{py,pth} in site-packages
```

Concretely, on top of `verl-omni-tts:v4`:
```bash
pip install "transformers==4.57.3"          # downgrades transformers + huggingface_hub
pip install --no-deps "qwen-tts==0.1.1" sox # qwen-tts modeling + its only missing import
# (librosa/soundfile/onnxruntime/einops/torchaudio already present from the reward backends)
# then drop hf_hub_unionfix.{py,pth} into site-packages (above)
```

## Acceptance transcript (single env, single process, H200)

```
[1/3 env] transformers=4.57.3 huggingface_hub=0.36.2 vllm_omni=0.22.0 torch=2.11.0+cu130
[1/3 env] PASS: vllm_omni imports on transformers 4.57.3 + hub<1.0
[2/3 gen] loading Qwen3-TTS ...
[2/3 gen] generate_voice_clone (x_vector_only_mode=True) ...
[2/3 gen] samples=53760 sr=24000 dur=2.24s peak=0.7969 rms=0.0990 raw_silence=0.134 frame_silence=0.054 longest_silent_run=0.026
[2/3 gen] wrote /weka/whiplash-grpo/depwall_clone.wav
[2/3 gen] whisper hypo = 'Hello, how can I help you today?'
[2/3 gen] whisper CER=0.000 (target reproduced from audio)
[2/3 gen] PASS: correct non-silent audio (raw_silence 13.4%, intelligible)
[3/3 train] codec-0 teacher-forced backward grad_norm=7.6034e+02 params_with_grad=312 (talker.model + codec_head)
[3/3 train] PASS: gradient flows into talker.model + talker.codec_head
PASS: vllm_omni + correct trainable Qwen3-TTS coexist in ONE env/process (raw_silence 13.4%, CER 0).
```

## On the silence metric (important)

The task's draft asserts `(|w|<1e-3).mean() < 0.15`. **That threshold is flaky for this model**
(sampling stochasticity straddles it) and is *not* what distinguishes correct audio from the tf5
break. Three sampled generations, all perfectly intelligible:

| clip | dur | peak | rms | raw `<1e-3` | frame `<-45dBFS` | longest silent run | whisper CER |
|------|-----|------|-----|-------------|------------------|--------------------|-------------|
| "Hello…today?" (sample A) | 2.00s | 0.887 | 0.076 | 0.276 | 0.212 | 0.119 | **0.000** |
| "Hello…today?" (sample B, final transcript) | 2.24s | 0.797 | 0.099 | **0.134** | 0.054 | 0.026 | **0.000** |
| 27-word support line | 9.36s | 0.727 | 0.059 | 0.236 | 0.176 | 0.036 | **0.000** |

The raw `<1e-3` ratio ranges **0.13–0.28 across samples of the same prompt** (the model samples by
default), so the 0.15 line is *inside* the natural variance of correct output — a correct model
fails it on roughly half its samples. This 12Hz code2wav decoder emits **clean digital silence**
(true sub-(-60 dBFS) samples) in natural pauses, so the raw metric stays elevated **regardless of
utterance length**, with the longest contiguous silent run only 3–12% (one natural pause). The
transformers-5.x break (QwenLM/Qwen3-TTS #237 / #201) instead produces **85–99% silence and
unintelligible output**. We therefore gate on the **discriminating** signals — whisper CER
(intelligibility) and a generous raw-silence sanity bound the break fails by a wide margin — and
report the raw ratio transparently. CER = 0.000 (exact transcription on both clips) is decisive
proof the audio is correct.

## Alternative (fix B) and why we didn't use it

`transformers 4.57.3` itself pins `kernels >=0.6.1,<=0.9`, and `kernels <=0.9` has **no**
`deps.py` / `import_name` strict dataclass and pins `huggingface_hub <1.0` — so
`transformers 4.57.3 + kernels<=0.9 + hub 0.36.2` has **no strict-union crash, no shim**. We kept
the shim instead because it leaves the vllm stack's `kernels 0.14.1` untouched (vllm may rely on
the newer kernel-hub), and the shim is a faithful backport of hub's own v1.3.0 fix — lower risk
than downgrading a package vllm pulled in.

## Notes / follow-ups

- **faster-whisper on CUDA-13:** the reward whisper (ctranslate2) wants `libcublas.so.12`; this is
  a CUDA-13 image, so whisper runs on **CPU** here (one short clip — fine). For GPU whisper in the
  reward path, use a cu12 image for that backend (cf. the cu128 whiplash reward image) — an image
  detail, orthogonal to the dependency wall.
- **Full vllm_omni rollout — VERIFIED (not just import).** Ran vllm-omni's canonical offline example
  (`examples/offline_inference/text_to_speech/qwen3_tts/end2end.py`, `Omni(model=...).generate(...)`)
  for a Base / `xvec_only` voice clone of `Qwen/Qwen3-TTS-12Hz-1.7B-Base` **on the coexistence env**
  (transformers 4.57.3 + hub 0.36.2 + shim). The 2-stage **talker → code2wav** pipeline initialized
  (158s: weight load, torch.compile, cudagraph capture for both stages), generated **6.32s** of audio,
  and shut down cleanly — **no hub≥1.0 / tf5 runtime API gap**. whisper **CER = 0.000** on the served
  wav (exact transcription), i.e. correct speech, not silence. Transcript:
  ```
  [Omni] AsyncOmniEngine initialized in 158.36 seconds
  [Omni] Initialized with 2 stages for model Qwen/Qwen3-TTS-12Hz-1.7B-Base
  Loaded 621 weights for Qwen3TTSTalkerForConditionalGeneration ... Code2Wav decoder CUDA Graph enabled
  stage-0/stage-1 add request 0_... -> output_0_*.wav   (dur 6.32s, peak 0.561, whisper CER 0.000)
  ```
  Crucially, vllm runs workers with `VLLM_WORKER_MULTIPROC_METHOD=spawn`, and the spawned
  `StageEngineCoreProc` workers loaded transformers (`configuration_qwen3_tts`) **without** the
  StrictDataclass error — so the `.pth` shim applies in worker subprocesses too (the production case).
  So both halves of a native colocated recipe are proven on one env: vllm_omni rollout (correct audio)
  **and** qwen-tts trainable codec-0 backward.
- **Memory correction:** the prior "disaggregation is the only path" was premature — angle A
  (the validator shim) was never tried. Native colocation is viable.
```
