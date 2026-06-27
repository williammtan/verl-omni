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

import numpy as np
import torch
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

from verl_omni.utils.reward_score.tts_quality import RewardConfig, RewardScorer, fused_reward


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
        self.scorer = RewardScorer(device=device)
        self._ref_cache: dict = {}

    @classmethod
    def assemble_rm_scores(cls, data: DataProto, scores: list[float]) -> torch.Tensor:
        """Per-sample audio rewards: ``rm_scores`` has shape ``(batch_size, 1)``."""
        return torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
        text = extra_info.get("text") or data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
        ref_audio = extra_info.get("ref_audio")
        uid = extra_info.get("id") or data_item.non_tensor_batch.get("uid")

        wav, sr = _extract_audio(data_item)

        def _score():
            if wav is None or wav.size == 0:
                return {"score": -10.0, "synth_ok": 0.0}  # hard synth failure
            if ref_audio not in self._ref_cache:
                self._ref_cache[ref_audio] = _load_ref_wav(ref_audio)
            res = self.scorer.score(
                wav, sr, id=str(uid), text=text, ref_audio_wav=self._ref_cache[ref_audio], group_key=str(uid)
            )
            score = fused_reward(res, self.reward_cfg)
            return {
                "score": float(score),
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
