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
"""Qwen3-TTS talker integration for verl's FSDP actor (Path A — external_lib monkeypatch).

Trains the **talker** (codec_0 main path: ``talker.model`` + ``talker.codec_head``) via GRPO and
freezes everything else (sub-codebook tables, speaker_encoder, code2wav). Mirrors
``qwen3_omni_thinker.py`` but the talker is NOT a plain causal LM, so instead of delegating to a
decoder we install a custom ``forward`` that rebuilds the talker's 2-channel teacher-forcing input
(text + codec-0 + speaker@6 + sub-codebooks 1..15) and returns codec-0 logits verl can gather. The
embedding/forward/realignment math lives in :mod:`qwen3_tts_forward` and is validated bit-for-bit
against whiplash (the G7 gate).

Loaded via ``actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_tts``.

Data contract (the forward reads these, injected per-sample via ``multi_modal_inputs`` — see
:func:`patch_agent_loop_multi_modal_inputs`):
  - ``tts_text_ids``    (B, T_text) long  — whiplash-tokenized spoken text, left-aligned
  - ``tts_audio_codes`` (B, R, 16)  long  — the rollout's full sampled codes, left-aligned
  - ``response_len``    (B,)        long  — real codec length per sample
  - ``text_len``        (B,)        long  — real text-id length per sample
verl supplies ``input_ids/attention_mask`` (dense, right-padded, packed at start); the response
codec-0 occupies the suffix ``[L_i - response_len_i, L_i)`` of the real region ``L_i = attn.sum()``.
"""

import logging
import os

import torch

from verl_omni.models.transformers.qwen3_tts_forward import tts_actor_logits

logger = logging.getLogger(__name__)

# Trainable scope: only these submodule prefixes (under the ForConditionalGeneration root) get grad.
_TRAINABLE_PREFIXES = ("talker.model.", "talker.codec_head.")


# --------------------------------------------------------------------------------------------
# Speaker embedding cache — one fixed clone voice for the whole run (see recipe README), so the
# ref mel + speaker_encoder forward is computed once and reused.
# --------------------------------------------------------------------------------------------
def _load_ref_mel(ref_audio_path: str) -> torch.Tensor:
    """Mel for the speaker_encoder, matching whiplash TTSDataset._extract_mels (24 kHz / n_fft 1024
    / 128 mels / hop 256 / win 1024 / fmin 0 / fmax 12000)."""
    import librosa
    import numpy as np
    from qwen_tts.utils.mel import mel_spectrogram  # qwen-tts ships the same mel as whiplash

    wav, sr = librosa.load(ref_audio_path, sr=None, mono=True)
    assert int(sr) == 24000, f"ref audio must be 24 kHz, got {sr}"
    mels = mel_spectrogram(
        torch.from_numpy(wav.astype(np.float32)).unsqueeze(0),
        n_fft=1024, num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000,
    ).transpose(1, 2)
    return mels  # (1, mel_t, 128)


def _speaker_embedding(model, batch_size: int, device, dtype) -> torch.Tensor:
    """Cached (B, H) speaker embedding from the fixed VERL_TTS_REF_AUDIO clip."""
    cache = getattr(model, "_verl_tts_spk_cache", None)
    if cache is None:
        ref = os.environ.get("VERL_TTS_REF_AUDIO")
        if not ref:
            raise RuntimeError("VERL_TTS_REF_AUDIO must point to the fixed clone ref wav (24 kHz).")
        mel = _load_ref_mel(ref).to(device=device, dtype=dtype)
        with torch.no_grad():
            cache = model.speaker_encoder(mel).detach()  # (1, H)
        model._verl_tts_spk_cache = cache
        logger.info("verl_omni.qwen3_tts: cached speaker embedding %s from %s", tuple(cache.shape), ref)
    return cache.to(device=device, dtype=dtype).expand(batch_size, -1)


# --------------------------------------------------------------------------------------------
# The custom forward — verl flat batch -> talker codec-0 logits (realigned to verl positions).
# --------------------------------------------------------------------------------------------
def _qwen3_tts_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    tts_text_ids=None,
    tts_audio_codes=None,
    response_len=None,
    text_len=None,
    **kwargs,
):
    """Return ``CausalLMOutputWithPast`` with ``.logits (B, T, codec_vocab)`` aligned to verl's flat
    ``input_ids`` so verl's ``roll(input_ids,-1)`` gather over the response slice recovers codec-0
    log-probs. The actual talker math is the G7-validated :mod:`qwen3_tts_forward` core."""
    from transformers.modeling_outputs import CausalLMOutputWithPast

    if tts_audio_codes is None or response_len is None or text_len is None or tts_text_ids is None:
        raise RuntimeError(
            "qwen3_tts forward requires tts_text_ids/tts_audio_codes/response_len/text_len via "
            "multi_modal_inputs (set rollout.engine_kwargs.vllm_omni.surface_sub_codes and load this "
            "module as external_lib). Missing keys mean the agent-loop MM patch did not run."
        )

    B = input_ids.shape[0]
    spk = _speaker_embedding(self, B, input_ids.device, next(self.talker.parameters()).dtype)
    out_logits = tts_actor_logits(
        self, input_ids, attention_mask, tts_text_ids, tts_audio_codes, response_len, text_len, spk
    )
    return CausalLMOutputWithPast(logits=out_logits)


def _qwen3_tts_get_input_embeddings(self):
    return self.talker.model.codec_embedding


def _qwen3_tts_set_input_embeddings(self, value):
    self.talker.model.codec_embedding = value


def _apply_freeze(model) -> None:
    """Train only ``talker.model`` + ``talker.codec_head`` (matches whiplash). Everything else —
    sub-codebook tables, speaker_encoder, code2wav — is used in the forward but frozen."""
    n_train = 0
    for name, p in model.named_parameters():
        train = name.startswith(_TRAINABLE_PREFIXES)
        p.requires_grad_(train)
        n_train += int(train)
    logger.info("verl_omni.qwen3_tts: %d trainable param tensors (talker.model + codec_head).", n_train)


def _register_qwen3_tts_automodel() -> None:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        return

    # The trainable class is in the qwen-tts package, not transformers, and not at top level —
    # it is qwen_tts.core.models.modeling_qwen3_tts.Qwen3TTSForConditionalGeneration (verified in-image).
    model_cls = config_cls = None
    for mod_path, attr in (
        ("transformers", "Qwen3TTSForConditionalGeneration"),
        ("qwen_tts.core.models.modeling_qwen3_tts", "Qwen3TTSForConditionalGeneration"),
    ):
        try:
            mod = __import__(mod_path, fromlist=[attr])
        except Exception:  # noqa: BLE001
            continue
        model_cls = getattr(mod, attr, None)
        if model_cls is not None:
            break
    for mod_path in ("transformers", "qwen_tts.core.models.configuration_qwen3_tts"):
        try:
            mod = __import__(mod_path, fromlist=["Qwen3TTSConfig"])
        except Exception:  # noqa: BLE001
            continue
        config_cls = getattr(mod, "Qwen3TTSConfig", None)
        if config_cls is not None:
            break
    if model_cls is None:
        logger.warning("verl_omni.qwen3_tts: Qwen3TTSForConditionalGeneration not found; patch is a no-op.")
        return

    try:
        from verl.utils.model import _architecture_to_auto_class

        _architecture_to_auto_class.setdefault(model_cls.__name__, AutoModelForCausalLM)
    except Exception as e:  # noqa: BLE001
        logger.warning("verl_omni.qwen3_tts: could not register architecture lookup (%s).", e)

    model_cls.forward = _qwen3_tts_forward
    model_cls.get_input_embeddings = _qwen3_tts_get_input_embeddings
    model_cls.set_input_embeddings = _qwen3_tts_set_input_embeddings
    # FSDP wrap hints: talker decoder layers + code_predictor layers (real class names from modeling).
    model_cls._no_split_modules = ["Qwen3TTSTalkerDecoderLayer", "Qwen3TTSDecoderLayer"]

    # Freeze to the talker on post_init so verl's optimizer/FSDP see the right requires_grad.
    _orig_post_init = getattr(model_cls, "post_init", None)

    def _post_init_with_freeze(self):
        if _orig_post_init is not None:
            _orig_post_init(self)
        _apply_freeze(self)

    model_cls.post_init = _post_init_with_freeze

    if config_cls is not None:
        # tie_word_embeddings=False keeps the FSDP meta-tensor init path (cf. thinker).
        class _FalseTie:
            def __get__(self, obj, objtype=None):
                return False

            def __set__(self, obj, value):
                pass

        try:
            config_cls.tie_word_embeddings = _FalseTie()
        except Exception:  # noqa: BLE001
            pass
        try:
            AutoModelForCausalLM.register(config_cls, model_cls)
        except Exception:  # noqa: BLE001 — already registered
            pass
    logger.info("verl_omni.qwen3_tts: installed talker codec-0 forward + freeze on %s.", model_cls.__name__)


# --------------------------------------------------------------------------------------------
# Agent-loop hook: inject the per-sample talker inputs into multi_modal_inputs so they arrive as
# forward kwargs. Mirrors the routed_experts scatter precedent but LEFT-ALIGNS response data (the
# FSDP engine re-packs input_ids right-padded-at-start, so MM data just needs the real prefix).
# The rollout surfaces the sampled (T,16) codes into extra_fields["tts_audio_codes"]; the text ids
# are tokenized here the whiplash way (we hold the processor + the prompt text via raw_prompt).
# --------------------------------------------------------------------------------------------
def _build_assistant_text(text: str) -> str:
    """whiplash TTSDataset._build_assistant_text (sft_trainer.py:73-74) — the talker text prompt."""
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def _unwrap_non_tensor(v):
    """raw_prompt rides as a 0-d object array; unwrap to the underlying messages list."""
    if hasattr(v, "item") and not isinstance(v, (list, tuple)):
        try:
            return v.item()
        except Exception:  # noqa: BLE001
            return v
    return v


def patch_agent_loop_multi_modal_inputs() -> None:
    try:
        from verl.experimental.agent_loop import agent_loop as _al
    except Exception as e:  # noqa: BLE001 — verl import may break under the pinned tf; degrade
        logger.warning("verl_omni.qwen3_tts: agent_loop MM patch skipped (%s).", e)
        return

    _orig = _al.AgentLoopWorker._compute_multi_modal_inputs

    def _patched(self, output, input_ids):
        mmi = _orig(self, output, input_ids)  # {} when processor is None (TTS) — we add to it
        ef = getattr(output, "extra_fields", None) or {}
        codes = ef.get("tts_audio_codes")  # (T,16) surfaced by the rollout _process_output
        if codes is None:
            return mmi  # surface_sub_codes off, or non-TTS rollout

        # text ids the whiplash way, from the prompt text (we hold the processor/tokenizer)
        raw_prompt = _unwrap_non_tensor(ef.get("raw_prompt"))
        text = raw_prompt[0]["content"] if raw_prompt else ""
        tok = self.processor or self.tokenizer
        text_ids = tok(text=_build_assistant_text(text), return_tensors="pt", padding=True)["input_ids"]
        text_ids = text_ids[:, :-5].reshape(-1)  # drop the trailing assistant header (sft_trainer.py:106)

        R = int(self.rollout_config.response_length)
        Ttext = int(self.rollout_config.prompt_length)
        codes = torch.as_tensor(codes, dtype=torch.long)  # (rl, 16)
        rl = min(codes.shape[0], R)
        tl = min(text_ids.numel(), Ttext)

        audio_buf = torch.zeros(1, R, codes.shape[-1], dtype=torch.long)
        audio_buf[0, :rl] = codes[:rl]
        text_buf = torch.zeros(1, Ttext, dtype=torch.long)
        text_buf[0, :tl] = text_ids[:tl]

        mmi["tts_audio_codes"] = audio_buf  # -> (B, R, 16)
        mmi["tts_text_ids"] = text_buf  # -> (B, Ttext)
        mmi["response_len"] = torch.tensor([rl], dtype=torch.long)  # -> (B,)
        mmi["text_len"] = torch.tensor([tl], dtype=torch.long)  # -> (B,)
        return mmi

    _al.AgentLoopWorker._compute_multi_modal_inputs = _patched
    logger.info("verl_omni.qwen3_tts: patched AgentLoopWorker._compute_multi_modal_inputs (TTS MM inject).")


# --------------------------------------------------------------------------------------------
# Passthrough chat template — Qwen3-TTS-Base has no chat_template but verl's RLDataset calls
# tokenizer.apply_chat_template on the prompt.
# --------------------------------------------------------------------------------------------
_TTS_PASSTHROUGH_CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


def patch_hf_tokenizer_for_qwen3_tts() -> None:
    try:
        import verl.utils.tokenizer as _vt
    except ImportError:
        return

    _original_hf_tokenizer = _vt.hf_tokenizer

    def _patched_hf_tokenizer(name_or_path, **kwargs):
        tok = _original_hf_tokenizer(name_or_path, **kwargs)
        if tok is not None and not getattr(tok, "chat_template", None):
            tok.chat_template = _TTS_PASSTHROUGH_CHAT_TEMPLATE
            logger.info("verl_omni.qwen3_tts: installed passthrough chat_template on %s.", type(tok).__name__)
        return tok

    _vt.hf_tokenizer = _patched_hf_tokenizer


def apply_qwen3_tts_patches() -> None:
    _register_qwen3_tts_automodel()
    patch_agent_loop_multi_modal_inputs()
    patch_hf_tokenizer_for_qwen3_tts()


apply_qwen3_tts_patches()
