"""Precompute the fixed clone voice's speaker x-vector ONCE, via the model's canonical
``extract_speaker_embedding`` (ECAPA over a 24 kHz mel). The same vector feeds both the vLLM-Omni
talker rollout (``voice_clone_prompt.ref_spk_embedding``, ``x_vector_only_mode=True``) and the verl
actor (speaker @ pos 6), so generation and the teacher-forced recompute condition on an identical
speaker. Saved as a JSON float list.

    python deploy/precompute_spk_embed.py --ref /weka/.../ref.wav --out /weka/.../spk_embed.json
"""

import argparse
import json

import librosa
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--ref", required=True, help="fixed clone reference wav")
    ap.add_argument("--out", required=True, help="output JSON path for the x-vector")
    args = ap.parse_args()

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    m = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0", dtype=torch.bfloat16)
    inner = m.model
    wav, sr = librosa.load(args.ref, sr=None, mono=True)
    with torch.no_grad():
        xvec = inner.extract_speaker_embedding(np.asarray(wav, dtype=np.float32), int(sr))
    xvec = xvec.detach().reshape(-1).float().cpu().tolist()
    with open(args.out, "w") as f:
        json.dump(xvec, f)
    print(f"saved {len(xvec)}-dim x-vector -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
