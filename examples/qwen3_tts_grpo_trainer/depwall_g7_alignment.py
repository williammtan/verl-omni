"""G7 — offline alignment gate: verl-omni's ported codec-0 forward == whiplash's, per token.

Proves the actor port (`verl_omni/models/transformers/qwen3_tts_forward.py`) reproduces whiplash's
teacher-forced codec-0 log-probs *before* any cluster training job. Two checks on one generated
rollout sample:

  A. ``build_talker_batch`` tensor-equals whiplash ``TTSDataset.collate_fn`` (the 2-channel layout,
     masks, speaker slot, codec span).
  B. ``codec0_logits`` (+ whiplash-style gather) matches whiplash ``codec0_logprobs`` per codec-0
     token (same speaker embedding, same talker forward).

Run on a 1-GPU pod (needs the qwen-tts model + a whiplash checkout):
    WHIPLASH_DIR=/workspace/whiplash python examples/qwen3_tts_grpo_trainer/depwall_g7_alignment.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --ref /weka/whiplash-sft/data/maeneka/ref.wav
"""

import argparse
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--ref", default="/weka/whiplash-sft/data/maeneka/ref.wav")
    ap.add_argument("--text", default="Hello, how can I help you today?")
    ap.add_argument("--language", default="english")
    ap.add_argument("--whiplash-dir", default=os.environ.get("WHIPLASH_DIR", "/workspace/whiplash"))
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    if args.whiplash_dir and args.whiplash_dir not in sys.path:
        sys.path.insert(0, args.whiplash_dir)

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    from verl_omni.models.transformers.qwen3_tts_forward import (
        TalkerTokens,
        build_talker_batch,
        codec0_logits,
    )

    dev = "cuda:0"
    print(f"[G7] loading {args.model} ...", flush=True)
    m = Qwen3TTSModel.from_pretrained(args.model, device_map=dev, dtype=torch.bfloat16)
    inner = m.model  # Qwen3TTSForConditionalGeneration
    talker = inner.talker

    # ---- 1) generate one rollout sample (codes (T,16)) — mirrors whiplash LocalRolloutEngine.sample
    items = m.create_voice_clone_prompt(ref_audio=args.ref, ref_text=None, x_vector_only_mode=True)
    vcp = m._prompt_items_to_voice_clone_prompt(items)
    gen_input_ids = m._tokenize_texts([m._build_assistant_text(args.text)])
    gen_kwargs = m._merge_generate_kwargs(do_sample=True, temperature=1.0, max_new_tokens=args.max_new_tokens)
    codes_list, _ = inner.generate(
        input_ids=gen_input_ids, ref_ids=None, voice_clone_prompt=vcp, languages=[args.language], **gen_kwargs
    )
    n_codebook = inner.talker.code_predictor.get_input_embeddings()[0].num_embeddings
    codes = codes_list[0].detach().cpu().clamp_(0, n_codebook - 1)  # (T,16)
    print(f"[G7] generated codes {tuple(codes.shape)} codec_vocab={n_codebook}", flush=True)

    # ---- 2) whiplash gold path: TTSDataset.collate_fn + codec0_logprobs
    from whiplash.train.grpo_trainer import codec0_logprobs
    from whiplash.train.sft_trainer import TTSDataset

    row = {"text": args.text, "audio_codes": codes, "ref_audio": args.ref}
    ds = TTSDataset([row], m.processor, inner.config)
    whip_batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ds.collate_fn([ds[0]]).items()}
    with torch.no_grad():
        gold_tok, gold_mask = codec0_logprobs(inner, whip_batch)

    # ---- 3) my port: build_talker_batch (A) + codec0_logits (B)
    text_ids = ds[0]["text_ids"]  # whiplash's processor(_build_assistant_text(text))[:, :-5]
    tokens = TalkerTokens.from_config(inner.config)
    my_batch = build_talker_batch([text_ids], [codes], tokens, device=dev, sub_codebook_vocab=n_codebook)

    # (A) layout equality vs whiplash collate
    a_ok = True
    for key, mine in [
        ("input_ids", my_batch.input_ids),
        ("codec_ids", my_batch.codec_ids),
        ("attention_mask", my_batch.attention_mask),
        ("codec_0_labels", my_batch.codec_0_labels),
        ("codec_mask", my_batch.codec_mask),
        ("text_embedding_mask", my_batch.text_embedding_mask),
        ("codec_embedding_mask", my_batch.codec_embedding_mask),
    ]:
        ref = whip_batch[key]
        same = mine.shape == ref.shape and torch.equal(mine.to(ref.dtype), ref)
        a_ok = a_ok and same
        print(f"[G7-A] {key:20s} match={same} shape={tuple(mine.shape)}", flush=True)
    assert a_ok, "[G7-A] FAIL: build_talker_batch != whiplash collate_fn"
    print("[G7-A] PASS: layout identical to whiplash collate_fn", flush=True)

    # (B) logits/logprob equality — same speaker embedding, same talker forward
    spk = inner.speaker_encoder(whip_batch["ref_mels"].to(inner.dtype)).detach()
    with torch.no_grad():
        my_logits = codec0_logits(talker, my_batch, spk)
        labels = my_batch.codec_0_labels[:, 1:]
        my_logp = torch.log_softmax(my_logits.float(), dim=-1)
        my_tok = my_logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)

    sel_gold = gold_tok[gold_mask]
    sel_mine = my_tok[my_batch.codec_0_labels[:, 1:] != -100]
    assert sel_gold.shape == sel_mine.shape, f"token count mismatch {sel_gold.shape} vs {sel_mine.shape}"
    max_abs = (sel_gold.float() - sel_mine.float()).abs().max().item()
    print(f"[G7-B] codec-0 tokens={sel_mine.numel()} max|Δlogp|={max_abs:.3e} "
          f"mean logp mine={sel_mine.mean().item():.4f} gold={sel_gold.mean().item():.4f}", flush=True)
    assert max_abs < args.atol, f"[G7-B] FAIL: max|Δlogp|={max_abs:.3e} >= {args.atol}"
    print("[G7-B] PASS: codec-0 logprobs match whiplash codec0_logprobs", flush=True)
    print("\nG7 PASS: verl-omni ported codec-0 forward == whiplash, per token.", flush=True)


if __name__ == "__main__":
    main()
