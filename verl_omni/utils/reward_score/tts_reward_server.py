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
"""Async HTTP TTS reward server (verl-omni-style scorer) for Qwen3-TTS GRPO.

Offloads reward scoring (faster-whisper CER, ERes2Net speaker-sim, UTMOSv2 MOS, stability) off the
trainer GPU onto its own GPU pod(s). The disaggregated trainer POSTs decoded rollout audio here and
gets back per-clip metrics, so (a) each trainer step is faster and (b) the trainer no longer loads
reward models in-process — which removes the multi-rank reward-model-load deadlock and unlocks DDP.

Run:  uvicorn-style via `python -m verl_omni.utils.reward_score.tts_reward_server` (reads env
PORT, REWARD_DEVICE, WHISPER_DEVICE). Wire model knobs via env.

Wire format (POST /score_batch):
  {"clips": [{"id","text","ref_audio","sr","wav_b64"}], "weights": {...optional...}}
  wav_b64 = base64(np.float32 mono PCM bytes). Returns {"results": [{id, cer, spk_similarity,
  utmos, truncated, repeated, cer_outlier, synth_ok}]} — the fields whiplash/verl reward composition
  needs. The client reconstructs its UttResult from these.
"""

from __future__ import annotations

import base64
import importlib.util
import logging
import os
import sys
import threading

import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tts_reward_server")

# Load the pure reward module by path so we don't trigger verl_omni's heavy package __init__.
_MOD = os.path.join(os.path.dirname(__file__), "tts_quality.py")
_spec = importlib.util.spec_from_file_location("tts_quality", _MOD)
tts_quality = importlib.util.module_from_spec(_spec)
sys.modules["tts_quality"] = tts_quality
_spec.loader.exec_module(tts_quality)


def _b64_to_wav(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).copy()


def _load_ref(path: str):
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.asarray(wav, dtype=np.float32), int(sr)


def build_app():
    from fastapi import FastAPI

    device = os.environ.get("REWARD_DEVICE", "cuda:0")
    whisper_device = os.environ.get("WHISPER_DEVICE", device)  # own GPU -> whisper on GPU
    scorer = tts_quality.RewardScorer(
        device=device,
        whisper_device=whisper_device,
        whisper_model=os.environ.get("WHISPER_MODEL", "large-v3"),
    )
    lock = threading.Lock()  # backends aren't thread-safe; serialize scoring on this replica
    ref_cache: dict = {}

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/score_batch")
    def score_batch(payload: dict):
        clips_in = payload.get("clips", [])
        if not clips_in:
            return {"results": []}
        sr = int(clips_in[0].get("sr", 24000))
        ref_path = clips_in[0].get("ref_audio")
        if ref_path and ref_path not in ref_cache:
            try:
                ref_cache[ref_path] = _load_ref(ref_path)
            except Exception as e:  # noqa: BLE001
                log.warning("ref load failed (%s) — SIM disabled.", e)
                ref_cache[ref_path] = None
        ref_wav = ref_cache.get(ref_path)
        clips = [
            {
                "id": c.get("id", str(i)),
                "text": c.get("text", ""),
                "wav": _b64_to_wav(c["wav_b64"]),
                "sr": int(c.get("sr", sr)),
                "ref_audio_wav": ref_wav,
                "group_key": c.get("group_key"),
            }
            for i, c in enumerate(clips_in)
        ]
        with lock:
            results = scorer.score_batch(clips)
        return {
            "results": [
                {
                    "id": r.id,
                    "cer": r.cer,
                    "spk_similarity": r.spk_similarity,
                    "emotion_similarity": r.emotion_similarity,
                    "utmos": r.utmos,
                    "duration_s": r.duration_s,
                    "truncated": r.truncated,
                    "repeated": r.repeated,
                    "cer_outlier": r.cer_outlier,
                    "synth_ok": r.synth_ok,
                    "hyp_text": r.hyp_text,
                }
                for r in results
            ]
        }

    return app


def main():
    import uvicorn

    app = build_app()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8100")), workers=1)


if __name__ == "__main__":
    main()
