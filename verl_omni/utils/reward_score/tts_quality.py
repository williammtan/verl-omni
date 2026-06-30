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
"""Multi-metric TTS reward for Qwen3-TTS GRPO (ported from the whiplash recipe).

A rollout is a decoded waveform; we score it on four dimensions and fuse them into a single
per-utterance reward (verl's ``adv_estimator: grpo`` then does the per-group advantage
normalization, so this module returns a per-sample scalar, not a group z-score):

    text  intelligibility  = -CER (faster-whisper transcribe vs the prompt text)
    sim   speaker-sim      = cosine of 3D-Speaker ERes2Net embeddings vs the clone reference
    mos   naturalness MOS  = UTMOSv2 reference-free MOS
    stab  stability        = graded penalty (truncation / repetition / CER-outlier / synth-fail)

Heavy backends (faster-whisper, modelscope ERes2Net, UTMOSv2) are imported lazily and degrade
gracefully: a backend that cannot load leaves its metric ``None``, which ``raw_rewards`` maps to
the worst plausible value. The reward math (``UttResult`` → ``raw_rewards`` → fused scalar) is
pure and unit-testable on CPU with no models.

Differs from whiplash in one deliberate way: whiplash did GLM-TTS hierarchical normalization
(per-dim z-score → fuse → z-score) inside the trainer; here we return the *weighted raw fusion*
and let verl's GRPO estimator z-score within the group. The ``ser`` (emotion) dimension is off
(no per-prompt emotion reference in this data).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import torch.nn as _nn

# Pristine nn.Module hooks captured at import time -- BEFORE a later reward-side code2wav load
# (Qwen3TTSTokenizer device_map=...) can leak accelerate.init_empty_weights, which patches
# register_parameter to meta-init and makes any subsequent from_pretrained land on the meta device
# (so .to(cuda) then raises "Cannot copy out of meta tensor"). _asr_model restores these before the
# GPU-whisper load so it materializes real weights. tts_quality is imported at reward-worker startup,
# before any model loads, so these are guaranteed pristine here.
_PRISTINE_NN_HOOKS = (_nn.Module.register_parameter, _nn.Module.register_buffer)

log = logging.getLogger("verl_omni.tts_quality")

# Dimensions fused into the reward, fixed order. ``stab`` is optional (use_stability).
REWARD_KEYS = ("text", "sim", "mos", "stab")

_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


# --------------------------------------------------------------------------------------------
# Per-utterance metric record + tail flags
# --------------------------------------------------------------------------------------------
@dataclass
class UttResult:
    """Normalized per-utterance metrics + tail flags."""

    id: str
    cer: float | None = None
    spk_similarity: float | None = None
    emotion_similarity: float | None = None
    utmos: float | None = None
    duration_s: float | None = None
    synth_ok: bool = True
    truncated: bool = False
    repeated: bool = False
    cer_outlier: bool = False
    group_key: str | None = None
    hyp_text: str | None = None
    raw: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Reward config + composition (pure)
# --------------------------------------------------------------------------------------------
@dataclass
class RewardConfig:
    """Weights + knobs for the fusion. Raw rewards are NOT capped."""

    w_text: float = 1.0
    w_sim: float = 1.0
    w_mos: float = 1.0
    w_stab: float = 1.0
    # stability penalty sub-weights (graded; combined into P_stab)
    p_trunc: float = 1.0
    p_rep: float = 1.0
    p_outlier: float = 1.0
    p_fail: float = 1.0
    use_stability: bool = True
    eps: float = 1e-2  # no-variance guard for group normalization

    def weights(self) -> dict[str, float]:
        w = {"text": self.w_text, "sim": self.w_sim, "mos": self.w_mos}
        if self.use_stability:
            w["stab"] = self.w_stab
        return w


@dataclass
class ClipSchedule:
    """Adaptive clip-higher schedule: linearly interpolate ``start``→``end`` over
    ``warmup_steps``, then hold. ``start == end`` gives a static bound."""

    start: float
    end: float
    warmup_steps: int = 1

    def value(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return self.end
        frac = min(max(step, 0) / self.warmup_steps, 1.0)
        return self.start + (self.end - self.start) * frac


def stability_penalty(r: UttResult, cfg: RewardConfig) -> float:
    """Graded P_stab (>= 0) from the tail flags."""
    return (
        cfg.p_trunc * float(r.truncated)
        + cfg.p_rep * float(r.repeated)
        + cfg.p_outlier * float(r.cer_outlier)
        + cfg.p_fail * float(not r.synth_ok)
    )


def raw_rewards(r: UttResult, cfg: RewardConfig) -> dict[str, float]:
    """Raw per-dimension rewards, monotone in each metric, NOT capped.

    Missing metrics fall back to the worst plausible value so an unscored rollout is never
    silently advantaged: text → CER 1.0, sim → 0.0 cosine, mos → 1.0 (lowest MOS).
    """
    rewards = {
        "text": -(r.cer if r.cer is not None else 1.0),
        "sim": r.spk_similarity if r.spk_similarity is not None else 0.0,
        "mos": r.utmos if r.utmos is not None else 1.0,
    }
    if cfg.use_stability:
        rewards["stab"] = -stability_penalty(r, cfg)
    return rewards


def fused_reward(r: UttResult, cfg: RewardConfig) -> float:
    """Weighted sum of raw per-dimension rewards — the per-sample scalar GRPO consumes."""
    w = cfg.weights()
    raw = raw_rewards(r, cfg)
    return sum(w[k] * raw[k] for k in w)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float], mean: float) -> float:
    return (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5  # population std


def _znorm(values: list[float], eps: float) -> list[float]:
    if not values:
        return []
    mean = _mean(values)
    std = _std(values, mean)
    if std < eps:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def group_advantages(group: list[UttResult], cfg: RewardConfig) -> tuple[list[float], bool]:
    """GLM-TTS hierarchical normalization for one prompt group (per-dim z-score → fuse →
    z-score). Returns ``(advantages, has_grad)``; ``has_grad`` is False for a homogeneous group.
    Kept for parity/dynamic-sampling experiments — the default verl path uses ``fused_reward``
    + the GRPO estimator instead."""
    if not group:
        return [], False
    weights = cfg.weights()
    raw = [raw_rewards(r, cfg) for r in group]
    normed = {k: _znorm([rr[k] for rr in raw], cfg.eps) for k in weights}
    fused = [sum(weights[k] * normed[k][i] for k in weights) for i in range(len(group))]
    mean_f = _mean(fused)
    std_f = _std(fused, mean_f)
    if std_f < cfg.eps:
        return [0.0 for _ in fused], False
    return [(f - mean_f) / std_f for f in fused], True


def group_reward_means(group: list[UttResult], cfg: RewardConfig) -> dict[str, float]:
    """Mean raw reward per dimension + weighted ``total`` for logging the learning curve."""
    if not group:
        return {}
    weights = cfg.weights()
    raw = [raw_rewards(r, cfg) for r in group]
    means = {k: _mean([rr[k] for rr in raw]) for k in weights}
    means["total"] = sum(weights[k] * means[k] for k in weights)
    return means


# --------------------------------------------------------------------------------------------
# Text / CER / stability helpers (pure)
# --------------------------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — applied to both sides of CER."""
    return " ".join(_WORD_RE.sub(" ", s.lower()).split())


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float | None:
    """Character error rate over normalized text: edits / len(reference). None if no ref."""
    ref = normalize_text(reference).replace(" ", "")
    hyp = normalize_text(hypothesis).replace(" ", "")
    if not ref:
        return None
    return _levenshtein(ref, hyp) / len(ref)


def has_repetition(text: str | None, n: int = 3, max_span: int = 30) -> bool:
    """True if any span of ``n``..``max_span`` words repeats immediately (a TTS looping artifact)."""
    if not text:
        return False
    words = text.split()
    upper = min(max_span, len(words) // 2)
    for span in range(n, upper + 1):
        for i in range(len(words) - 2 * span + 1):
            if words[i : i + span] == words[i + span : i + 2 * span]:
                return True
    return False


def expected_duration_s(text: str, speaking_rate_wps: float) -> float | None:
    words = len(text.split())
    if words == 0 or speaking_rate_wps <= 0:
        return None
    return words / speaking_rate_wps


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav.astype(np.float32)
    import librosa

    return librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)


# --------------------------------------------------------------------------------------------
# In-process metric models
# --------------------------------------------------------------------------------------------
class RewardScorer:
    """Scores a rollout waveform into a :class:`UttResult` with in-process models.

    Backends load on first use and degrade gracefully. ``device`` is a torch-style string
    (``"cpu"`` or ``"cuda:N"``); the GPU index is parsed for per-rank pinning.
    """

    def __init__(
        self,
        *,
        whisper_model: str = "large-v3",
        whisper_compute_type: str = "int8",
        whisper_device: str = "cpu",  # ctranslate2 is a CUDA-12 binary; CPU avoids the cu12/cu13
                                      # clash with the CUDA-13 vllm/torch stack (whiplash saga).
        whisper_backend: str | None = None,  # None -> auto: "transformers" on cuda, "faster" on cpu
        spk_model: str = "iic/speech_eres2net_sv_en_voxceleb_16k",
        device: str = "cpu",
        language: str = "en",
        cer_outlier_threshold: float = 0.30,
        truncation_ratio: float = 0.5,
        speaking_rate_wps: float = 2.5,
        repetition_ngram: int = 3,
    ):
        self.whisper_model = whisper_model
        self.whisper_compute_type = whisper_compute_type
        self.whisper_device = whisper_device
        self.whisper_backend = whisper_backend
        self.spk_model = spk_model
        self.device = device
        self._cuda_index = (
            int(device.split(":", 1)[1]) if device.startswith("cuda") and ":" in device else 0
        )
        self.language = language
        self.cer_outlier_threshold = cer_outlier_threshold
        self.truncation_ratio = truncation_ratio
        self.speaking_rate_wps = speaking_rate_wps
        self.repetition_ngram = repetition_ngram
        self._asr = None
        self._spk = None
        self._utmos = None
        self._utmos_failed = False  # once create_model fails, give up (never retry → no loop)
        import threading

        self._asr_lock = threading.Lock()
        # Eager-load the GPU whisper model at construction (single-threaded, before any concurrent
        # reward-side model loads start) to avoid a register_parameter meta-init race: the reward
        # scores many clips concurrently, so N concurrent lazy first-loads would race with a
        # concurrent accelerate device_map load (code2wav/spk/utmos) that transiently patches
        # nn.Module.register_parameter to meta-init -> "Cannot copy out of meta tensor". CPU
        # faster-whisper stays lazy (ctranslate2, no nn.Module, no race).
        if self.whisper_backend == "transformers" or self.whisper_device.startswith("cuda"):
            try:
                self._asr_model()
            except Exception as e:  # noqa: BLE001
                log.warning("eager whisper warmup failed (%s) — will load lazily", e)

    # --- lazy backends ---------------------------------------------------------------
    def _asr_model(self):
        # Returns (backend, model). Two backends:
        #   "faster"       -- faster-whisper/ctranslate2 (CPU; cu12 binary clashes with the cu13 stack)
        #   "transformers" -- torch-native whisper (GPU; cu13-native, used for cuda + distil-whisper)
        # GPU defaults to the transformers backend so we avoid the ctranslate2 cu12/cu13 wall entirely.
        if self._asr is not None:
            return self._asr
        with self._asr_lock:                      # load exactly once per worker (see __init__ note)
            if self._asr is not None:
                return self._asr
            wd = self.whisper_device
            use_cuda = wd.startswith("cuda")
            backend = self.whisper_backend or ("transformers" if use_cuda else "faster")
            if backend == "transformers":
                import torch
                import torch.nn as nn
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

                idx = int(wd.split(":", 1)[1]) if (use_cuda and ":" in wd) else self._cuda_index
                dev = f"cuda:{idx}" if use_cuda else "cpu"
                dtype = torch.float16 if use_cuda else torch.float32
                log.info("loading transformers-whisper %r on %s", self.whisper_model, dev)
                # Belt-and-suspenders for the register_parameter meta-init race (eager warmup in
                # __init__ is the primary guard): restore the pristine nn.Module hooks captured at
                # import so from_pretrained materializes real weights instead of meta.
                nn.Module.register_parameter, nn.Module.register_buffer = _PRISTINE_NN_HOOKS
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.whisper_model, dtype=dtype, low_cpu_mem_usage=False,
                ).to(dev).eval()
                proc = AutoProcessor.from_pretrained(self.whisper_model)
                pipe = pipeline(
                    "automatic-speech-recognition", model=model,
                    tokenizer=proc.tokenizer, feature_extractor=proc.feature_extractor,
                    torch_dtype=dtype, device=dev, chunk_length_s=30,
                )
                self._asr = ("transformers", pipe)
            else:
                from faster_whisper import WhisperModel

                if use_cuda:
                    idx = int(wd.split(":", 1)[1]) if ":" in wd else self._cuda_index
                    dev, kw = "cuda", {"device_index": idx}
                else:
                    dev, kw = "cpu", {}
                ct = self.whisper_compute_type if dev == "cuda" else "int8"
                log.info("loading faster-whisper %r (%s) on %s", self.whisper_model, ct, dev)
                self._asr = ("faster", WhisperModel(self.whisper_model, device=dev, compute_type=ct, **kw))
        return self._asr

    def _spk_embed(self, wav: np.ndarray, sr: int) -> np.ndarray | None:
        try:
            if self._spk is None:
                from modelscope.pipelines import pipeline

                self._spk = pipeline(task="speaker-verification", model=self.spk_model, device=self.device)
            out = self._spk([_resample(wav, sr, 16000)], output_emb=True)
            return np.asarray(out["embs"][0], dtype=np.float32)
        except Exception as e:  # noqa: BLE001 — degrade: leave SIM unscored
            log.warning("speaker embedding unavailable (%s) — SIM left None.", e)
            return None

    def _utmos_model(self):
        if self._utmos is None and not self._utmos_failed:
            try:
                import utmosv2

                self._utmos = utmosv2.create_model(pretrained=True, device=self.device)
            except Exception as e:  # noqa: BLE001 — give up, don't retry
                log.warning("utmosv2 unavailable (%s) — MOS disabled for this run.", e)
                self._utmos_failed = True
        return self._utmos

    def _utmos_predict(self, wav: np.ndarray, sr: int) -> float | None:
        m = self._utmos_model()
        if m is None:
            return None
        try:
            out = m.predict(data=wav.astype(np.float32), sr=sr, device=self.device, verbose=False)
            return float(np.asarray(out).reshape(-1)[0])
        except Exception as e:  # noqa: BLE001 — degrade: leave MOS unscored
            log.warning("utmosv2 predict failed (%s) — MOS left None.", e)
            return None

    # --- scoring ---------------------------------------------------------------------
    def transcribe(self, wav: np.ndarray, sr: int) -> str:
        backend, model = self._asr_model()
        audio = np.asarray(_resample(wav, sr, 16000), dtype=np.float32)
        if backend == "transformers":
            import torch

            # The reward worker's torch default device can be 'meta' (leaked from prior model
            # loads); the HF pipeline's dataloader calls .item() which dies on a meta default.
            # Pin it to cpu for the inference call, then restore.
            _prev_dev = torch.get_default_device()
            torch.set_default_device("cpu")
            try:
                # distil-whisper/distil-large-v3 is English-only -> no language kwarg (errors on one)
                out = model({"array": audio, "sampling_rate": 16000})
            finally:
                torch.set_default_device(_prev_dev)
            return (out.get("text") or "").strip()
        seg_iter, _ = model.transcribe(
            audio, language=self.language, beam_size=1, condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in seg_iter).strip()

    def _assemble(
        self, *, id: str, text: str, wav: np.ndarray, sr: int, hyp: str,
        sim: float | None, mos: float | None, group_key: str | None,
    ) -> UttResult:
        c = cer(text, hyp)
        duration_s = len(wav) / sr if sr else None
        exp = expected_duration_s(text, self.speaking_rate_wps)
        truncated = duration_s is not None and exp is not None and duration_s < self.truncation_ratio * exp
        return UttResult(
            id=id, cer=c, spk_similarity=sim, utmos=mos, duration_s=duration_s,
            synth_ok=c is not None,
            cer_outlier=c is not None and c > self.cer_outlier_threshold,
            repeated=has_repetition(hyp, self.repetition_ngram), truncated=truncated,
            group_key=group_key, hyp_text=hyp,
        )

    def _sim_one(self, wav, sr, ref_audio_wav):
        if ref_audio_wav is None:
            return None
        a, b = self._spk_embed(wav, sr), self._spk_embed(*ref_audio_wav)
        return _cosine(a, b) if (a is not None and b is not None) else None

    def score(
        self, wav: np.ndarray, sr: int, *, id: str, text: str,
        ref_audio_wav: tuple[np.ndarray, int] | None = None, group_key: str | None = None,
    ) -> UttResult:
        hyp = self.transcribe(wav, sr)
        sim = self._sim_one(wav, sr, ref_audio_wav)
        mos = self._utmos_predict(wav, sr)
        return self._assemble(id=id, text=text, wav=wav, sr=sr, hyp=hyp, sim=sim, mos=mos, group_key=group_key)

    # --- batched scoring (one model pass over a whole group — keeps the GPUs fed) ------
    def score_batch(self, clips: list[dict]) -> list[UttResult]:
        """Batched scoring: one UTMOSv2 + one ERes2Net pass over the group. Whisper stays
        per-clip (faster-whisper has no cross-audio batch). ``clips`` items:
        ``{wav, sr, id, text, ref_audio_wav, group_key}``."""
        if not clips:
            return []
        sr = clips[0]["sr"]
        wavs = [c["wav"] for c in clips]
        hyps = [self.transcribe(c["wav"], c["sr"]) for c in clips]
        mos = self._utmos_batch(wavs, sr)
        sims = self._sim_batch(wavs, sr, clips[0].get("ref_audio_wav"))
        return [
            self._assemble(
                id=c["id"], text=c["text"], wav=c["wav"], sr=c["sr"], hyp=hyps[i],
                sim=sims[i], mos=mos[i], group_key=c.get("group_key"),
            )
            for i, c in enumerate(clips)
        ]

    def _utmos_batch(self, wavs: list[np.ndarray], sr: int) -> list[float | None]:
        m = self._utmos_model()
        if m is None:
            return [None] * len(wavs)
        try:
            t = max(len(w) for w in wavs)
            batch = np.zeros((len(wavs), t), dtype=np.float32)
            for i, w in enumerate(wavs):
                batch[i, : len(w)] = w.astype(np.float32)
            out = np.asarray(
                m.predict(data=batch, sr=sr, device=self.device, batch_size=max(1, len(wavs)), verbose=False)
            ).reshape(-1)
            return [float(x) for x in out]
        except Exception as e:  # noqa: BLE001 — degrade to per-clip
            log.warning("utmosv2 batch failed (%s) — per-clip fallback.", e)
            return [self._utmos_predict(w, sr) for w in wavs]

    def _sim_batch(self, wavs, sr, ref_audio_wav):
        if ref_audio_wav is None:
            return [None] * len(wavs)
        try:
            if self._spk is None:
                from modelscope.pipelines import pipeline

                self._spk = pipeline(task="speaker-verification", model=self.spk_model, device=self.device)
            inputs = [_resample(w, sr, 16000) for w in wavs]
            inputs.append(_resample(ref_audio_wav[0], ref_audio_wav[1], 16000))
            embs = [np.asarray(e, dtype=np.float32) for e in self._spk(inputs, output_emb=True)["embs"]]
            ref = embs[-1]
            return [_cosine(embs[i], ref) for i in range(len(wavs))]
        except Exception as e:  # noqa: BLE001 — degrade to per-clip
            log.warning("speaker batch failed (%s) — per-clip fallback.", e)
            return [self._sim_one(w, sr, ref_audio_wav) for w in wavs]
