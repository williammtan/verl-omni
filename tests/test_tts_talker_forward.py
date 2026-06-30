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
"""Pure (no-model) tests for the Qwen3-TTS talker codec-0 forward layout + realignment.

These validate the index math that the G7 on-GPU harness can't isolate: that
:func:`build_talker_batch` reproduces whiplash's collate layout, and — the load-bearing claim —
that verl's own ``roll(input_ids,-1)`` gather over :func:`realign_to_verl`'s output recovers the
*same* per-token codec-0 log-probs that whiplash's ``codec0_logprobs`` gather produces. Needs only
torch (CPU); no qwen-tts model.

Run (in any image with torch): ``pytest tests/test_tts_talker_forward.py -q``
"""

import torch

from verl_omni.models.transformers.qwen3_tts_forward import (
    SPEAKER_SLOT,
    TalkerTokens,
    build_talker_batch,
    realign_to_verl,
)

# Arbitrary but distinct control-token ids (real values come from the model config at runtime).
TOKENS = TalkerTokens(
    tts_pad=900, tts_bos=901, tts_eos=902,
    codec_pad=4196, codec_bos=4197, codec_eos=4198,
    codec_nothink=4203, codec_think_bos=4204, codec_think_eos=4205,
)
# Codec-head output width used only for the synthetic logits in these pure tests. It must exceed
# every id used as a label below (incl. the codec specials ~4198), so the gather is in-bounds. The
# real codec-head width comes from the model at runtime; the forward never hardcodes it.
VOCAB = 4300
SUB_VOCAB = 2048


def _sample(tl, cl, seed):
    g = torch.Generator().manual_seed(seed)
    text_ids = torch.randint(0, 800, (1, tl), generator=g)
    codes = torch.randint(0, SUB_VOCAB, (cl, 16), generator=g)  # col 0 = codec-0
    return text_ids, codes


def test_layout_matches_whiplash_collate():
    """build_talker_batch places codec-0, labels, masks, and the speaker slot exactly where
    whiplash's collate_fn does."""
    samples = [_sample(7, 5, 0), _sample(11, 9, 1)]
    text_ids = [s[0] for s in samples]
    audio_codes = [s[1] for s in samples]
    batch = build_talker_batch(text_ids, audio_codes, TOKENS, sub_codebook_vocab=SUB_VOCAB)

    for i, (tid, codes) in enumerate(samples):
        tl, cl = tid.shape[1], codes.shape[0]
        codec0 = codes[:, 0]
        s = 8 + tl - 1  # codec-0 span start
        # codec-0 lives in channel 1 and in the labels over [s, s+cl)
        assert torch.equal(batch.input_ids[i, s : s + cl, 1], codec0)
        assert torch.equal(batch.codec_0_labels[i, s : s + cl], codec0)
        assert batch.codec_0_labels[i, s + cl].item() == TOKENS.codec_eos
        # outside the codec span (and the eos) labels are masked
        assert (batch.codec_0_labels[i, :s] == -100).all()
        # full 16 codebooks retained at the codec span
        assert torch.equal(batch.codec_ids[i, s : s + cl, :], codes)
        # control block + speaker slot
        assert batch.input_ids[i, 7, 0].item() == TOKENS.tts_bos
        assert batch.input_ids[i, 8 + tl - 2, 1].item() == TOKENS.codec_bos
        assert batch.codec_embedding_mask[i, SPEAKER_SLOT, 0].item() is False or \
            batch.codec_embedding_mask[i, SPEAKER_SLOT, 0].item() == 0
        # the logit that predicts codec-0 token k is at index logit_start + k
        assert batch.logit_start[i] == 8 + tl - 2
        assert batch.codec_lens[i] == cl and batch.text_lens[i] == tl


def test_realign_makes_verl_gather_equal_whiplash_gather():
    """THE correctness claim: with random talker logits, verl's response-slice gather over the
    realigned (B,T,V) tensor yields the identical codec-0 log-probs as whiplash's teacher-forced
    gather (logits[j] vs codec_0_labels[j+1])."""
    samples = [_sample(7, 5, 2), _sample(11, 9, 3)]
    text_ids = [s[0] for s in samples]
    audio_codes = [s[1] for s in samples]
    batch = build_talker_batch(text_ids, audio_codes, TOKENS, sub_codebook_vocab=SUB_VOCAB)

    b = len(samples)
    t = batch.input_ids.shape[1]
    torch.manual_seed(7)
    whip_logits = torch.randn(b, t - 1, VOCAB)  # talker(emb[:,:-1]) → length t-1

    # whiplash gather: tok_logp[j] = log_softmax(whip_logits[j])[codec_0_labels[j+1]]
    labels = batch.codec_0_labels[:, 1:]  # (b, t-1)
    logp = torch.log_softmax(whip_logits.float(), dim=-1)
    whip_tok_logp = logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    whip_mask = labels != -100

    # Build a verl-style flat batch: response region holds codec-0 (+eos) at a chosen start.
    T = t + 4  # extra room to prove positions are not assumed
    response_starts = []
    flat_input_ids = torch.zeros(b, T, dtype=torch.long)
    for i, (_, codes) in enumerate(samples):
        cl = codes.shape[0]
        rs = 3 + i  # arbitrary, per-sample-distinct response start
        response_starts.append(rs)
        flat_input_ids[i, rs : rs + cl] = codes[:, 0]
        flat_input_ids[i, rs + cl] = TOKENS.codec_eos  # include eos as the last response token

    # realign cl codec-0 rows; extend by 1 to also carry the eos-predicting row.
    out = realign_to_verl(whip_logits, batch, response_starts, (b, T, VOCAB))
    # also copy the eos-predicting row (whiplash index logit_start+cl) so verl can score eos too
    for i, (_, codes) in enumerate(samples):
        cl = codes.shape[0]
        out[i, response_starts[i] - 1 + cl, :] = whip_logits[i, batch.logit_start[i] + cl, :]

    # verl gather: logp_verl[p] = log_softmax(out[p])[flat_input_ids[p+1]] over the response slice
    out_logp = torch.log_softmax(out.float(), dim=-1)
    for i, (_, codes) in enumerate(samples):
        tl = text_ids[i].shape[1]
        cl = codes.shape[0]
        rs = response_starts[i]
        for k in range(cl + 1):  # cl codec-0 tokens + eos
            label = flat_input_ids[i, rs + k]
            verl_lp = out_logp[i, rs - 1 + k, label]
            whip_j = batch.logit_start[i] + k  # 8+tl-2+k
            whip_lp = whip_tok_logp[i, whip_j]
            assert whip_mask[i, whip_j], f"sample {i} k {k}: whiplash position not a label"
            assert torch.allclose(verl_lp, whip_lp, atol=1e-6), (
                f"sample {i} token {k}: verl {verl_lp.item():.6f} != whiplash {whip_lp.item():.6f}"
            )
