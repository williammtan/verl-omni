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
"""Reward manager for Qwen3-TTS GRPO — scores decoded audio with the multi-metric TTS reward.

The vLLM-Omni Qwen3-TTS rollout decodes codec tokens to a 24kHz waveform internally
(Talker → Code2Wav) and surfaces it on the rollout output's multimodal payload. This manager
reads that waveform per sample, scores it with :class:`RewardScorer` (intelligibility / speaker-sim
/ MOS / stability), and returns the weighted fused scalar; verl's ``adv_estimator: grpo`` then
does the per-group advantage normalization.

One ``RewardScorer`` (lazy, in-process metric models) is held per manager and pinned to a GPU.
Mirrors the structure of :class:`VisualRewardManager`.
"""

import os
import threading

import numpy as np
import torch
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.experimental.reward_loop.reward_manager.registry import register

from verl_omni.utils.reward_score.tts_quality import RewardConfig, RewardScorer, fused_reward, raw_rewards


def _load_ref_wav(path):
    """Load the fixed clone reference clip once (cached). Returns (wav, sr) or None."""
    if not path:
        return None
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.asarray(wav, dtype=np.float32), int(sr)


def _extract_audio(data_item):
    """Pull the decoded waveform (np.float32) + sample rate from a rollout output.

    The vLLM-Omni TTS rollout exposes audio under the multimodal output; we accept a few
    shapes so the manager is robust to how the rollout adapter packs it:
      - non_tensor_batch["audio"] = np.ndarray|Tensor, ["sr"] = int
      - non_tensor_batch["multimodal_output"] = {"audio": ..., "sr": ...}
    """
    nb = data_item.non_tensor_batch
    audio = nb.get("audio")
    sr = nb.get("sr")
    if audio is None and isinstance(nb.get("multimodal_output"), dict):
        mm = nb["multimodal_output"]
        audio, sr = mm.get("audio"), mm.get("sr")
    if audio is None:
        return None, None
    if isinstance(audio, (list, tuple)):
        audio = torch.cat([a if torch.is_tensor(a) else torch.as_tensor(a) for a in audio], dim=-1)
    if torch.is_tensor(audio):
        audio = audio.float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if isinstance(sr, (list, tuple)):
        sr = sr[-1]
    sr = int(sr.item()) if hasattr(sr, "item") else int(sr or 24000)
    return audio, sr


@register("TTSRewardManager")
class TTSRewardManager(RewardManagerBase):
    """Scores decoded Qwen3-TTS rollouts on the multi-metric TTS reward."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        rc = (getattr(config, "reward", None) and getattr(config.reward, "tts", None)) or {}
        self.reward_cfg = RewardConfig(
            w_text=float(rc.get("w_text", 1.0)) if isinstance(rc, dict) else 1.0,
            w_sim=float(rc.get("w_sim", 1.0)) if isinstance(rc, dict) else 1.0,
            w_mos=float(rc.get("w_mos", 1.0)) if isinstance(rc, dict) else 1.0,
            w_stab=float(rc.get("w_stab", 1.0)) if isinstance(rc, dict) else 1.0,
        )
        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        # Whisper ASR (intelligibility/CER reward) defaults to CPU faster-whisper (the ctranslate2
        # cu12-vs-cu13 punt). Opt into GPU ASR via env: VERL_TTS_WHISPER_DEVICE=cuda runs whisper on
        # the reward worker's GPU using the torch-native transformers backend (distil-whisper by
        # default) -- much faster and far lighter on host RAM than CPU large-v3.
        whisper_kw: dict = {}
        _wd = os.environ.get("VERL_TTS_WHISPER_DEVICE")
        if _wd:
            whisper_kw["whisper_device"] = device if _wd == "cuda" else _wd
            whisper_kw["whisper_model"] = os.environ.get(
                "VERL_TTS_WHISPER_MODEL", "distil-whisper/distil-large-v3"
            )
            _wb = os.environ.get("VERL_TTS_WHISPER_BACKEND")
            if _wb:
                whisper_kw["whisper_backend"] = _wb
        self.scorer = RewardScorer(device=device, **whisper_kw)
        self._ref_cache: dict = {}
        # Reward-side code2wav: when the rollout surfaces codec tokens but no waveform
        # (single-stage talker), decode (T,16) codes -> 24kHz wav here and score that.
        # Lazy-loads only the qwen-tts speech_tokenizer (the small conv vocoder, not the
        # 1.7B talker), pinned to the reward GPU. Path source: VERL_TTS_MODEL_PATH env,
        # else the actor model path, else the public Base checkpoint.
        self._device = device
        self._decoder = None
        self._decoder_lock = threading.Lock()
        self._codebook_max = 2047
        mp = os.environ.get("VERL_TTS_MODEL_PATH")
        if not mp:
            try:
                mp = config.actor_rollout_ref.model.path
            except Exception:
                mp = None
        self._model_path = mp or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    @classmethod
    def assemble_rm_scores(cls, data: DataProto, scores: list[float]) -> torch.Tensor:
        """Per-sample audio rewards: ``rm_scores`` has shape ``(batch_size, 1)``."""
        return torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

    def _get_decoder(self):
        """Lazy-load the qwen-tts speech_tokenizer (code2wav vocoder) on the reward GPU.

        Resolves the ``speech_tokenizer/`` subfolder the way ``Qwen3TTSModel`` does
        (``cached_file`` -> ``dirname``), so an HF repo id or a local checkpoint dir both
        work. Guarded by a lock — reward samples score in concurrent executor threads.
        """
        if self._decoder is not None:
            return self._decoder
        with self._decoder_lock:
            if self._decoder is None:
                from qwen_tts import Qwen3TTSTokenizer
                from transformers.utils import cached_file

                cfg = cached_file(self._model_path, "speech_tokenizer/config.json")
                st_dir = os.path.dirname(cfg)
                dec = Qwen3TTSTokenizer.from_pretrained(st_dir, device_map=self._device, dtype=torch.bfloat16)
                try:
                    self._codebook_max = int(dec.model.config.decoder_config.codebook_size) - 1
                except Exception:
                    pass
                self._decoder = dec
        return self._decoder

    def _decode_codes(self, data_item):
        """Decode surfaced rollout codec tokens ``(T,16)`` -> ``(wav float32, sr)``.

        Returns ``(None, None)`` when no codes are present so the caller falls through to
        the synth-fail sentinel. Clamps to ``[0, codebook_size)`` (the tokenizer only
        clamps the lower bound; an out-of-range index device-asserts).
        """
        # The rollout's extra_fields (incl. tts_audio_codes) are packed by verl's agent loop
        # into non_tensor_batch["tool_extra_fields"] as one object-array dict (agent_loop.py:976;
        # same path visual.py:52 reads). After data[0] it's the dict itself.
        tef = data_item.non_tensor_batch.get("tool_extra_fields")
        if isinstance(tef, np.ndarray) and tef.dtype == object:  # defensive: unwrap if not yet indexed
            tef = tef.item() if tef.ndim == 0 else (tef[0] if len(tef) else None)
        codes = tef.get("tts_audio_codes") if isinstance(tef, dict) else None
        if codes is None:
            return None, None
        codes = torch.as_tensor(codes, dtype=torch.long)
        if codes.ndim != 2 or codes.shape[-1] != 16:
            return None, None
        # Trim post-eos garbage: the talker emits codec_eos (2150) ~100-120 frames in; with rollout
        # eos-stop the codes are already short, but trim defensively so we never decode the ~170s tail.
        eos = (codes[:, 0] == 2150).nonzero().flatten()
        if len(eos):
            codes = codes[: int(eos[0])]
        if codes.shape[0] == 0:
            return None, None
        dec = self._get_decoder()
        codes = codes.clamp_(0, self._codebook_max)
        wavs, sr = dec.decode([{"audio_codes": codes}])
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1), int(sr)

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
        text = extra_info.get("text") or data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
        ref_audio = extra_info.get("ref_audio")
        uid = extra_info.get("id") or data_item.non_tensor_batch.get("uid")

        wav, sr = _extract_audio(data_item)

        def _score():
            w, s = wav, sr
            if w is None:
                w, s = self._decode_codes(data_item)  # decode codes->wav in the executor thread
            if w is None or w.size == 0:
                # Hard synth failure: worst-case per-dim rewards (sign-correct for GDPO).
                # Key set MUST match the success branch — the reward loop reads keys from
                # sample 0 only and hard-indexes every sample, so a ragged dict drops
                # dimensions or KeyErrors (reward_loop.py:371-374).
                return {
                    "score": -10.0,
                    "rw_text": -1.0, "rw_sim": 0.0, "rw_mos": 1.0,
                    "rw_stab": -float(self.reward_cfg.p_fail),
                    "cer": -1.0, "sim": -1.0, "mos": -1.0,
                    "synth_ok": 0.0, "truncated": 0.0, "repeated": 0.0,
                }
            if ref_audio not in self._ref_cache:
                self._ref_cache[ref_audio] = _load_ref_wav(ref_audio)
            res = self.scorer.score(
                w, s, id=str(uid), text=text, ref_audio_wav=self._ref_cache[ref_audio], group_key=str(uid)
            )
            score = fused_reward(res, self.reward_cfg)
            raw = raw_rewards(res, self.reward_cfg)  # per-dim rewards for GDPO (higher = better)
            return {
                "score": float(score),
                "rw_text": float(raw["text"]), "rw_sim": float(raw["sim"]),
                "rw_mos": float(raw["mos"]), "rw_stab": float(raw.get("stab", 0.0)),
                "cer": res.cer if res.cer is not None else -1.0,
                "sim": res.spk_similarity if res.spk_similarity is not None else -1.0,
                "mos": res.utmos if res.utmos is not None else -1.0,
                "synth_ok": float(res.synth_ok),
                "truncated": float(res.truncated),
                "repeated": float(res.repeated),
            }

        result = await self.loop.run_in_executor(None, _score)
        score = result.pop("score")
        return {"reward_score": score, "reward_extra_info": result}
