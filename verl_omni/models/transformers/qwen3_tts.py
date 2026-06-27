# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3-TTS model-specific patches for verl's FSDP actor.

We train the **talker** (codec_0 main path) and freeze the rest (sub-talker / code_predictor
codebooks 1-15, speaker_encoder); the code2wav decoder lives rollout-side (vLLM-Omni), so the
FSDP actor strips it. Mirrors ``qwen3_omni_thinker.py`` but for the TTS talker.

Loaded on demand via verl's
``actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_tts`` so it only takes
effect when a Qwen3-TTS model is trained.

⚠️ PHASE-2 VALIDATE ON-CLUSTER: the exact HF class names below
(``Qwen3TTSForConditionalGeneration`` / ``Qwen3TTSConfig`` and the talker submodule attribute)
must be confirmed against the installed ``qwen_tts`` / ``transformers`` Qwen3-TTS classes — adjust
the names once the image is up. Trainable scope is enforced via LoRA ``target_modules`` /
``exclude_modules`` in the recipe config; the strip/no-split hints here keep FSDP init lean.
"""

import logging

logger = logging.getLogger(__name__)

# Modules the FSDP actor does not need for the codec_0 log-prob forward (decode is rollout-side).
_STRIP_MODULES = ["code2wav", "speech_tokenizer"]


def _register_qwen3_tts_automodel() -> None:
    """Register the Qwen3-TTS talker model with AutoModelForCausalLM + verl's lookup."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        return

    # Resolve the Qwen3-TTS model + config classes from whichever package provides them.
    model_cls = config_cls = None
    arch_name = "Qwen3TTSForConditionalGeneration"
    for mod_path in ("transformers", "qwen_tts"):
        try:
            mod = __import__(mod_path, fromlist=["*"])
        except ImportError:
            continue
        model_cls = getattr(mod, arch_name, None) or getattr(mod, "Qwen3TTSModel", None)
        config_cls = getattr(mod, "Qwen3TTSConfig", None)
        if model_cls is not None:
            break
    if model_cls is None:
        logger.warning("verl_omni.qwen3_tts: Qwen3-TTS model class not found; patch is a no-op "
                       "until the real class name is wired in (Phase 2).")
        return

    try:
        from verl.utils.model import _architecture_to_auto_class

        _architecture_to_auto_class.setdefault(model_cls.__name__, AutoModelForCausalLM)
    except Exception as e:  # noqa: BLE001
        logger.warning("verl_omni.qwen3_tts: could not register architecture lookup (%s).", e)

    # FSDP engine reads _verl_strip_modules to delete sub-modules not needed for training.
    model_cls._verl_strip_modules = _STRIP_MODULES
    if config_cls is not None:
        try:
            AutoModelForCausalLM.register(config_cls, model_cls)
        except Exception:  # noqa: BLE001 — already registered
            pass
    logger.info("verl_omni.qwen3_tts: registered %s (strip=%s).", model_cls.__name__, _STRIP_MODULES)


# Passthrough chat template: Qwen3-TTS-Base is a TTS model with no chat_template, but verl's
# RLDataset calls tokenizer.apply_chat_template on the prompt. This template just emits each
# message's content (the line to speak) — so apply_chat_template returns the raw TTS text.
_TTS_PASSTHROUGH_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['content'] }}{% endfor %}"
)


def patch_hf_tokenizer_for_qwen3_tts() -> None:
    """Wrap ``verl.utils.tokenizer.hf_tokenizer`` to give chat-template-less tokenizers (Qwen3-TTS)
    a passthrough template, so verl's dataset tokenization doesn't crash."""
    try:
        import verl.utils.tokenizer as _vt
    except ImportError:
        return

    _original_hf_tokenizer = _vt.hf_tokenizer

    def _patched_hf_tokenizer(name_or_path, **kwargs):
        tok = _original_hf_tokenizer(name_or_path, **kwargs)
        if tok is not None and not getattr(tok, "chat_template", None):
            tok.chat_template = _TTS_PASSTHROUGH_CHAT_TEMPLATE
            logger.info("verl_omni.qwen3_tts: installed passthrough chat_template on %s.",
                        type(tok).__name__)
        return tok

    _vt.hf_tokenizer = _patched_hf_tokenizer


def apply_qwen3_tts_patches() -> None:
    """Apply all Qwen3-TTS patches (idempotent)."""
    _register_qwen3_tts_automodel()
    patch_hf_tokenizer_for_qwen3_tts()


# Apply on import so this module works as a verl ``external_lib`` target.
apply_qwen3_tts_patches()
