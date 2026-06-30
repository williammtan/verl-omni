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
"""Qwen3-TTS talker codec-0 forward — the trainable signal for verl's FSDP actor.

This is a faithful port of whiplash's teacher-forced codec-0 log-prob path
(``whiplash/train/sft_trainer.py::TTSDataset.collate_fn`` + ``grpo_trainer.py::codec0_logprobs``)
into verl-omni, so the native recipe can train the talker through verl's GRPO actor instead of
the standalone whiplash trainer.

The talker is NOT a plain causal LM: its input is a **2-channel, position-co-located** sequence
with an 8-slot control prefix (think/nothink tokens, a speaker-embedding slot at codec position 6,
codec pad), and the codec-0 logits at each frame depend on the *previous* frame's full 16-codebook
set (speaker + residual codebooks 1..15 are summed into the input embedding). verl, by contrast,
hands the actor a flat single-channel ``input_ids`` and expects ``.logits`` it can gather with
``roll(input_ids, -1)``. This module bridges the two:

  1. :func:`build_talker_batch` — rebuilds whiplash's 2-channel batch from per-sample
     ``(text_ids, audio_codes)`` (pure tensor ops; unit-testable on CPU).
  2. :func:`codec0_logits` — assembles the talker input embedding (text + codec-0 + speaker@6 +
     sub-codebooks 1..15) and runs the talker, returning per-position codec-0 logits over the 3072
     codec vocab (needs the model).
  3. :func:`realign_to_verl` — copies the codec-0 logit rows onto verl's flat response positions so
     verl's own ``roll``/gather/response-slice recovers the codec-0 log-probs (pure).

Layout reference (whiplash ``collate_fn``), per sample with text length ``tl`` / codec length ``cl``,
total ``t = max(tl + cl) + 8``:
  ch0 (text):  [0:3]=text[:3]  [3:7]=tts_pad  [7]=tts_bos  [8:8+tl-3]=text[3:]
               [8+tl-3]=tts_eos  [8+tl-2:8+tl+cl]=tts_pad
  ch1 (codec): [3]=nothink [4]=think_bos [5]=think_eos [6]=0(speaker slot) [7]=codec_pad
               [8:8+tl-2]=codec_pad  [8+tl-2]=codec_bos  [8+tl-1:8+tl-1+cl]=codec0  [.+cl]=codec_eos
  codec_0_labels: [8+tl-1 : 8+tl-1+cl]=codec0 ; [8+tl-1+cl]=codec_eos ; else -100
  speaker @ codec position 6 ; sub-codebooks 1..15 over the codec span only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

# Number of residual sub-codebooks summed into the talker input (columns 1..15 of the 16-wide
# code tensor). Column 0 is codec-0 (the trained next-token target).
NUM_SUB_CODEBOOKS = 15
NUM_CODEBOOKS = NUM_SUB_CODEBOOKS + 1  # 16
SPEAKER_SLOT = 6  # codec position reserved for the injected speaker embedding


@dataclass
class TalkerTokens:
    """The control/special token ids the talker layout needs, read from the model config so we
    never hardcode (they live on ``config`` / ``config.talker_config``)."""

    tts_pad: int
    tts_bos: int
    tts_eos: int
    codec_pad: int
    codec_bos: int
    codec_eos: int
    codec_nothink: int
    codec_think_bos: int
    codec_think_eos: int

    @classmethod
    def from_config(cls, config) -> "TalkerTokens":
        tcfg = config.talker_config
        return cls(
            tts_pad=int(config.tts_pad_token_id),
            tts_bos=int(config.tts_bos_token_id),
            tts_eos=int(config.tts_eos_token_id),
            codec_pad=int(tcfg.codec_pad_id),
            codec_bos=int(tcfg.codec_bos_id),
            codec_eos=int(tcfg.codec_eos_token_id),
            codec_nothink=int(tcfg.codec_nothink_id),
            codec_think_bos=int(tcfg.codec_think_bos_id),
            codec_think_eos=int(tcfg.codec_think_eos_id),
        )


@dataclass
class TalkerBatch:
    """The teacher-forcing batch the talker forward consumes, plus per-sample offsets needed to
    realign the codec-0 logits back onto verl's flat sequence."""

    input_ids: torch.Tensor  # (B, t, 2)  channel 0 = text, channel 1 = codec
    codec_ids: torch.Tensor  # (B, t, 16) full codebooks (col 0 = codec-0)
    text_embedding_mask: torch.Tensor  # (B, t, 1) bool
    codec_embedding_mask: torch.Tensor  # (B, t, 1) bool (False at the speaker slot)
    codec_mask: torch.Tensor  # (B, t) bool — the acoustic-frame span (sub-codebooks added here)
    attention_mask: torch.Tensor  # (B, t) long
    codec_0_labels: torch.Tensor  # (B, t) long, -100 outside the codec span
    text_lens: list[int]  # tl per sample
    codec_lens: list[int]  # cl per sample
    # codec-0 logit at whiplash index (8 + tl - 2 + k) predicts codec-0 token k (k in [0, cl)).
    logit_start: list[int]  # 8 + tl - 2 per sample


def build_talker_batch(
    text_ids: list[torch.Tensor],
    audio_codes: list[torch.Tensor],
    tokens: TalkerTokens,
    *,
    device=None,
    sub_codebook_vocab: int | None = None,
) -> TalkerBatch:
    """Port of whiplash ``TTSDataset.collate_fn`` (sft_trainer.py:111-180).

    Args:
        text_ids: per-sample text token ids, each shape ``(1, tl)`` or ``(tl,)`` — already the
            whiplash ``processor(_build_assistant_text(text))[:, :-5]`` tokens.
        audio_codes: per-sample full codebooks, each shape ``(cl, 16)`` long (col 0 = codec-0).
        tokens: control token ids from :meth:`TalkerTokens.from_config`.
        sub_codebook_vocab: if given, sub-codebook columns 1..15 are clamped to ``[0, vocab)``
            (matches whiplash's rollout clamp, guards against device-side asserts on bf16-sampled
            out-of-range codes).
    """
    b = len(text_ids)
    tls = [int(t.reshape(-1).shape[0]) for t in text_ids]
    cls = [int(c.shape[0]) for c in audio_codes]
    t = max(tl + cl for tl, cl in zip(tls, cls)) + 8

    input_ids = torch.zeros((b, t, 2), dtype=torch.long, device=device)
    codec_ids = torch.zeros((b, t, NUM_CODEBOOKS), dtype=torch.long, device=device)
    text_embedding_mask = torch.zeros((b, t), dtype=torch.bool, device=device)
    codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool, device=device)
    codec_mask = torch.zeros((b, t), dtype=torch.bool, device=device)
    attention_mask = torch.zeros((b, t), dtype=torch.long, device=device)
    codec_0_labels = torch.full((b, t), -100, dtype=torch.long, device=device)

    for i in range(b):
        tid = text_ids[i].reshape(-1).to(device=device, dtype=torch.long)
        codes = audio_codes[i].to(device=device, dtype=torch.long)
        if sub_codebook_vocab is not None:
            codes = codes.clone()
            codes[:, 1:NUM_CODEBOOKS] = codes[:, 1:NUM_CODEBOOKS].clamp_(0, sub_codebook_vocab - 1)
        codec0 = codes[:, 0]
        tl, cl = tls[i], cls[i]

        # text channel (ch0)
        input_ids[i, :3, 0] = tid[:3]
        input_ids[i, 3:7, 0] = tokens.tts_pad
        input_ids[i, 7, 0] = tokens.tts_bos
        input_ids[i, 8 : 8 + tl - 3, 0] = tid[3:]
        input_ids[i, 8 + tl - 3, 0] = tokens.tts_eos
        input_ids[i, 8 + tl - 2 : 8 + tl + cl, 0] = tokens.tts_pad
        text_embedding_mask[i, : 8 + tl + cl] = True

        # codec channel (ch1) — positions 3..7 are the think/speaker/pad control block
        input_ids[i, 3:8, 1] = torch.tensor(
            [tokens.codec_nothink, tokens.codec_think_bos, tokens.codec_think_eos, 0, tokens.codec_pad],
            dtype=torch.long, device=device,
        )
        input_ids[i, 8 : 8 + tl - 3, 1] = tokens.codec_pad
        input_ids[i, 8 + tl - 3, 1] = tokens.codec_pad
        input_ids[i, 8 + tl - 2, 1] = tokens.codec_bos
        input_ids[i, 8 + tl - 1 : 8 + tl - 1 + cl, 1] = codec0
        input_ids[i, 8 + tl - 1 + cl, 1] = tokens.codec_eos

        codec_0_labels[i, 8 + tl - 1 : 8 + tl - 1 + cl] = codec0
        codec_0_labels[i, 8 + tl - 1 + cl] = tokens.codec_eos

        codec_ids[i, 8 + tl - 1 : 8 + tl - 1 + cl, :] = codes

        codec_embedding_mask[i, 3 : 8 + tl + cl] = True
        codec_embedding_mask[i, SPEAKER_SLOT] = False
        codec_mask[i, 8 + tl - 1 : 8 + tl - 1 + cl] = True
        attention_mask[i, : 8 + tl + cl] = True

    return TalkerBatch(
        input_ids=input_ids,
        codec_ids=codec_ids,
        text_embedding_mask=text_embedding_mask.unsqueeze(-1),
        codec_embedding_mask=codec_embedding_mask.unsqueeze(-1),
        codec_mask=codec_mask,
        attention_mask=attention_mask,
        codec_0_labels=codec_0_labels,
        text_lens=tls,
        codec_lens=cls,
        logit_start=[8 + tl - 2 for tl in tls],
    )


def assemble_talker_embeddings(talker, batch: TalkerBatch, speaker_emb: torch.Tensor) -> torch.Tensor:
    """Build the talker input embedding (whiplash codec0_logprobs:261-267), returning ``emb (B,t,H)``.

    ``speaker_emb`` is ``(B, H)`` injected at codec position 6; ``talker`` is the
    ``Qwen3TTSTalkerForConditionalGeneration`` (``.model.text_embedding`` / ``.model.codec_embedding``
    / ``.code_predictor.get_input_embeddings()``).
    """
    ids = batch.input_ids
    if os.environ.get("VERL_TTS_DEBUG"):
        sub0 = talker.code_predictor.get_input_embeddings()[0]
        print(
            f"[tts-dev] text_emb={talker.model.text_embedding.weight.device} "
            f"codec_emb={talker.model.codec_embedding.weight.device} "
            f"sub0={sub0.weight.device} ids={ids.device} spk={speaker_emb.device}",
            flush=True,
        )
    # vLLM-Omni generation runs every text-side embedding (incl. the tts_pad at the speaker slot)
    # through talker.text_projection — a learned ResizeMLP, NOT identity (prompt_embeds_builder.py:
    # 1280, 1236). whiplash's codec0_logprobs skips it (it never compares rollout-vs-actor, using
    # old_logp=logp.detach()), but verl does, so the rollout is the on-policy oracle: apply it here.
    te_raw = talker.model.text_embedding(ids[:, :, 0])
    text_proj = getattr(talker, "text_projection", None)
    if text_proj is not None:
        te_raw = text_proj(te_raw)
    te = te_raw * batch.text_embedding_mask
    ce = talker.model.codec_embedding(ids[:, :, 1]) * batch.codec_embedding_mask
    ce = ce.clone()
    ce[:, SPEAKER_SLOT, :] = speaker_emb.to(ce.dtype)
    emb = te + ce
    sub_tables = talker.code_predictor.get_input_embeddings()
    cmask = batch.codec_mask.unsqueeze(-1)
    for i in range(1, NUM_CODEBOOKS):
        emb = emb + sub_tables[i - 1](batch.codec_ids[:, :, i]) * cmask
    return emb


def codec0_logits(talker, batch: TalkerBatch, speaker_emb: torch.Tensor) -> torch.Tensor:
    """Run the talker over the teacher-forcing window and return codec-0 logits ``(B, t-1, 3072)``.

    Mirrors whiplash ``codec0_logprobs`` (grpo_trainer.py:269-273): the talker is fed
    ``inputs_embeds[:, :-1]`` with ``attention_mask[:, :-1]`` (so it recomputes RoPE from the mask,
    matching generation), and ``out.logits[j]`` predicts the codec-0 token at position ``j+1``.
    """
    emb = assemble_talker_embeddings(talker, batch, speaker_emb)
    out = talker(
        inputs_embeds=emb[:, :-1, :],
        attention_mask=batch.attention_mask[:, :-1],
        use_cache=False,
        output_hidden_states=False,
    )
    return out.logits


def tts_actor_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tts_text_ids: torch.Tensor,
    tts_audio_codes: torch.Tensor,
    response_len: torch.Tensor,
    text_len: torch.Tensor,
    speaker_emb: torch.Tensor,
) -> torch.Tensor:
    """verl flat batch -> codec-0 logits ``(B, T, codec_vocab)`` realigned to verl's positions.

    The pure heart of the actor forward (no verl/transformers-output deps), so a G8 harness can
    validate it against whiplash on a hand-built verl batch. ``model`` is the
    ``Qwen3TTSForConditionalGeneration``; ``input_ids/attention_mask`` are verl's dense
    right-padded-at-start tensors (real region ``[0, L_i)``, ``L_i = attention_mask[i].sum()``,
    codec-0 response = suffix ``[L_i-response_len_i, L_i)``). ``tts_text_ids`` ``(B, *)`` and
    ``tts_audio_codes`` ``(B, *, 16)`` are left-aligned per-sample; ``speaker_emb`` is ``(B, H)``.
    """
    talker = model.talker
    device = input_ids.device
    b = input_ids.shape[0]
    t_out = input_ids.shape[1]
    real_len = attention_mask.sum(dim=1).to(torch.long)

    text_ids_list, audio_codes_list, response_starts = [], [], []
    for i in range(b):
        rl, tl, li = int(response_len[i]), int(text_len[i]), int(real_len[i])
        text_ids_list.append(tts_text_ids[i, :tl].to(torch.long))
        audio_codes_list.append(tts_audio_codes[i, :rl].to(torch.long))  # (rl, 16)
        response_starts.append(li - rl)  # dense index of codec-0 token 0

    tokens = TalkerTokens.from_config(model.config)
    sub_vocab = int(talker.code_predictor.get_input_embeddings()[0].num_embeddings)

    if os.environ.get("VERL_TTS_DEBUG"):
        te_v = int(talker.model.text_embedding.num_embeddings)
        ce_v = int(talker.model.codec_embedding.num_embeddings)
        for i in range(b):
            rl, tl, li = int(response_len[i]), int(text_len[i]), int(real_len[i])
            ac, ti = audio_codes_list[i], text_ids_list[i]
            print(
                f"[tts-dbg] i={i} L={li} resp_len={rl} text_len={tl} rs={li - rl} "
                f"codec0[min={int(ac[:, 0].min())},max={int(ac[:, 0].max())}] "
                f"sub[min={int(ac[:, 1:].min())},max={int(ac[:, 1:].max())}] "
                f"text[min={int(ti.min())},max={int(ti.max())}]",
                flush=True,
            )
        print(f"[tts-dbg] vocab text_emb={te_v} codec_emb={ce_v} sub_emb={sub_vocab}", flush=True)

    batch = build_talker_batch(text_ids_list, audio_codes_list, tokens, device=device, sub_codebook_vocab=sub_vocab)
    whip_logits = codec0_logits(talker, batch, speaker_emb)
    vocab = whip_logits.shape[-1]
    # verl gathers log-probs over the FULL flat sequence (text prompt + codec response) before
    # slicing the response, so out_logits must be wide enough for the text-prompt labels too (they
    # are discarded downstream). Widen to cover max(codec vocab, any input id).
    out_vocab = max(vocab, int(input_ids.max().item()) + 1)
    if os.environ.get("VERL_TTS_DEBUG"):
        for i in range(b):
            rl, li = int(response_len[i]), int(real_len[i])
            rs = li - rl
            resp = input_ids[i, rs : rs + rl]
            print(
                f"[tts-dbg fwd] i={i} codec_head_vocab={vocab} "
                f"verl_resp_codec0[min={int(resp.min())},max={int(resp.max())}] "
                f"mm_codec0[max={int(audio_codes_list[i][:, 0].max())}] rs={rs} rl={rl}",
                flush=True,
            )
    return realign_to_verl(whip_logits, batch, response_starts, (b, t_out, out_vocab))


def realign_to_verl(
    whip_logits: torch.Tensor,
    batch: TalkerBatch,
    response_starts: list[int],
    out_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Scatter codec-0 logits onto verl's flat ``(B, T, vocab)`` so verl's ``roll(input_ids,-1)``
    gather over the response slice recovers codec-0 log-probs.

    verl computes ``logp[p] = logprob(input_ids[p+1] | logits[p])``; the response codec-0 token ``k``
    sits at flat position ``response_start + k`` and is predicted by ``logits[response_start+k-1]``.
    whiplash's logit for that same token is ``whip_logits[8+tl-2+k]`` (``batch.logit_start[i]+k``).
    So for each sample we copy ``cl`` contiguous rows:
        out[i, rs-1 : rs-1+cl] = whip_logits[i, ls : ls+cl]
    Prompt/pad rows are left zero — verl drops them via ``response_mask``.
    """
    b, T, out_vocab = out_shape
    codec_vocab = whip_logits.shape[-1]
    out = whip_logits.new_zeros((b, T, out_vocab))  # prompt/pad rows: finite 0 (their logprobs are discarded)
    for i in range(b):
        cl = batch.codec_lens[i]
        ls = batch.logit_start[i]
        rs = response_starts[i]
        sl = slice(rs - 1, rs - 1 + cl)
        if out_vocab > codec_vocab:
            # response rows: mask the non-codec columns so codec-0 log_softmax is over the codec vocab only
            out[i, sl, codec_vocab:] = -1e4
        out[i, sl, :codec_vocab] = whip_logits[i, ls : ls + cl, :]
    return out
