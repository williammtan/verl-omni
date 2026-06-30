"""G8 — offline actor-glue gate: the verl forward (`tts_actor_logits`) reproduces whiplash's codec-0
log-probs through verl's exact gather, on a hand-built verl dense batch.

Where G7 validated the talker math in whiplash's own layout, G8 validates the *verl integration*:
the dense right-padded-at-start `input_ids`, the response-region recovery
(`response_start = L_i - response_len`), the multi_modal_inputs unpacking, the realignment, and the
teacher-forcing `-1` shift — i.e. that `out_logits[response_start-1+j]` gathered against
`input_ids[response_start+j]` (codec-0 token j) equals whiplash's per-token codec-0 logprob.

Run on the 1-GPU pod (same env as G7):
    WHIPLASH_DIR=/workspace/whiplash python examples/qwen3_tts_grpo_trainer/depwall_g8_actor.py
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
    ap.add_argument("--prompt-pad", type=int, default=9, help="fake verl prompt length (tests dynamic response_start)")
    ap.add_argument("--right-pad", type=int, default=5, help="extra right padding (tests pad handling)")
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    if args.whiplash_dir and args.whiplash_dir not in sys.path:
        sys.path.insert(0, args.whiplash_dir)

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    from verl_omni.models.transformers.qwen3_tts_forward import tts_actor_logits

    dev = "cuda:0"
    print(f"[G8] loading {args.model} ...", flush=True)
    m = Qwen3TTSModel.from_pretrained(args.model, device_map=dev, dtype=torch.bfloat16)
    inner = m.model

    # ---- generate one rollout sample (codes (T,16)) ----
    items = m.create_voice_clone_prompt(ref_audio=args.ref, ref_text=None, x_vector_only_mode=True)
    vcp = m._prompt_items_to_voice_clone_prompt(items)
    gen_input_ids = m._tokenize_texts([m._build_assistant_text(args.text)])
    gen_kwargs = m._merge_generate_kwargs(do_sample=True, temperature=1.0, max_new_tokens=args.max_new_tokens)
    codes_list, _ = inner.generate(
        input_ids=gen_input_ids, ref_ids=None, voice_clone_prompt=vcp, languages=[args.language], **gen_kwargs
    )
    n_codebook = inner.talker.code_predictor.get_input_embeddings()[0].num_embeddings
    codes = codes_list[0].detach().cpu().clamp_(0, n_codebook - 1)  # (cl,16)
    cl = codes.shape[0]
    codec0 = codes[:, 0]
    print(f"[G8] generated codes {tuple(codes.shape)}", flush=True)

    # ---- whiplash gold (same as G7) ----
    from whiplash.train.grpo_trainer import codec0_logprobs
    from whiplash.train.sft_trainer import TTSDataset

    ds = TTSDataset([{"text": args.text, "audio_codes": codes, "ref_audio": args.ref}], m.processor, inner.config)
    whip_batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ds.collate_fn([ds[0]]).items()}
    with torch.no_grad():
        gold_tok, gold_mask = codec0_logprobs(inner, whip_batch)
    gold = gold_tok[gold_mask].float().cpu()  # length cl (+1 eos); compare the first cl
    spk = inner.speaker_encoder(whip_batch["ref_mels"].to(inner.dtype)).detach()
    text_ids = ds[0]["text_ids"].view(-1)  # whiplash-tokenized text
    tl = text_ids.numel()

    # ---- build a verl-style DENSE batch: [prompt(Pp) || response(codec0, cl)] right-padded ----
    Pp = args.prompt_pad
    T = Pp + cl + args.right_pad
    input_ids = torch.zeros(1, T, dtype=torch.long, device=dev)
    input_ids[0, :Pp] = torch.arange(1, Pp + 1)  # arbitrary prompt tokens (forward ignores them)
    input_ids[0, Pp : Pp + cl] = codec0.to(dev)  # response region == rollout codec-0
    attention_mask = torch.zeros(1, T, dtype=torch.long, device=dev)
    attention_mask[0, : Pp + cl] = 1  # real region [0, L), L = Pp+cl; pad on the right

    # multi_modal_inputs the agent-loop patch would inject (left-aligned)
    R = cl + args.right_pad
    tts_audio_codes = torch.zeros(1, R, 16, dtype=torch.long, device=dev)
    tts_audio_codes[0, :cl] = codes.to(dev)
    tts_text_ids = torch.zeros(1, max(tl, 4), dtype=torch.long, device=dev)
    tts_text_ids[0, :tl] = text_ids.to(dev)
    response_len = torch.tensor([cl], device=dev)
    text_len = torch.tensor([tl], device=dev)

    # ---- run the actor glue forward ----
    with torch.no_grad():
        out_logits = tts_actor_logits(
            inner, input_ids, attention_mask, tts_text_ids, tts_audio_codes, response_len, text_len, spk
        )
    assert out_logits.shape[0] == 1 and out_logits.shape[1] == T, f"bad shape {tuple(out_logits.shape)}"

    # ---- verl-style gather: logp[j] = log_softmax(out[response_start-1+j])[input_ids[response_start+j]] ----
    response_start = Pp  # = L - response_len
    lp = torch.log_softmax(out_logits[0].float(), dim=-1)
    mine = torch.stack([lp[response_start - 1 + j, int(input_ids[0, response_start + j])] for j in range(cl)]).cpu()

    max_abs = (mine - gold[:cl]).abs().max().item()
    print(f"[G8] T={T} Pp={Pp} cl={cl} response_start={response_start} "
          f"max|Δlogp|={max_abs:.3e} mean mine={mine.mean():.4f} gold={gold[:cl].mean():.4f}", flush=True)
    # consistency: the codec-0 the forward scored must equal verl's response labels
    assert torch.equal(input_ids[0, Pp : Pp + cl].cpu(), codec0), "response region != rollout codec-0"
    assert max_abs < args.atol, f"[G8] FAIL: max|Δlogp|={max_abs:.3e} >= {args.atol}"
    print("\nG8 PASS: verl actor glue (tts_actor_logits) == whiplash codec-0 logprobs through verl's gather.", flush=True)


if __name__ == "__main__":
    main()
