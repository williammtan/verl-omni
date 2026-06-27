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
"""CPU-only unit tests for the Qwen3-TTS GRPO reward math (no models, no GPU)."""

import importlib.util
import os

# Load the (pure, numpy-only) reward module directly by path so the CPU test gate does not
# trigger verl_omni's torch-heavy package __init__.
_MOD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "verl_omni", "utils", "reward_score", "tts_quality.py"
)
import sys

_spec = importlib.util.spec_from_file_location("tts_quality", os.path.normpath(_MOD_PATH))
tts_quality = importlib.util.module_from_spec(_spec)
sys.modules["tts_quality"] = tts_quality  # dataclasses need the module registered
_spec.loader.exec_module(tts_quality)

ClipSchedule = tts_quality.ClipSchedule
RewardConfig = tts_quality.RewardConfig
UttResult = tts_quality.UttResult
_std = tts_quality._std
_znorm = tts_quality._znorm
cer = tts_quality.cer
fused_reward = tts_quality.fused_reward
group_advantages = tts_quality.group_advantages
has_repetition = tts_quality.has_repetition
normalize_text = tts_quality.normalize_text
raw_rewards = tts_quality.raw_rewards
stability_penalty = tts_quality.stability_penalty


def _utt(id_, cer=None, sim=None, mos=None, **flags):
    return UttResult(id=id_, cer=cer, spk_similarity=sim, utmos=mos, **flags)


# ---- znorm -------------------------------------------------------------------------------
def test_znorm_zeroes_when_no_variance():
    assert _znorm([0.7, 0.7, 0.7], eps=1e-2) == [0.0, 0.0, 0.0]


def test_znorm_is_zero_mean_unit_std():
    out = _znorm([1.0, 2.0, 3.0], eps=1e-2)
    assert abs(sum(out)) < 1e-9
    assert abs(_std(out, 0.0) - 1.0) < 1e-9


# ---- raw rewards -------------------------------------------------------------------------
def test_raw_rewards_directions_and_no_capping():
    cfg = RewardConfig()
    rr = raw_rewards(_utt("a", cer=0.2, sim=0.9, mos=4.1), cfg)
    assert rr["text"] == -0.2  # lower CER -> higher (less negative)
    assert rr["sim"] == 0.9
    assert rr["mos"] == 4.1
    rr2 = raw_rewards(_utt("b", cer=0.0, sim=5.0, mos=9.0), cfg)
    assert rr2["sim"] == 5.0  # cosine not capped


def test_missing_metrics_fall_back_to_worst():
    cfg = RewardConfig()
    rr = raw_rewards(_utt("a"), cfg)  # all metrics None
    assert rr["text"] == -1.0  # CER 1.0
    assert rr["sim"] == 0.0
    assert rr["mos"] == 1.0  # lowest MOS


def test_stability_penalty_graded():
    cfg = RewardConfig()
    assert stability_penalty(_utt("a", synth_ok=True), cfg) == 0.0
    pen = stability_penalty(_utt("b", truncated=True, repeated=True, cer_outlier=True, synth_ok=False), cfg)
    assert pen == 4.0  # 1+1+1+1


def test_fused_reward_weighted_sum():
    cfg = RewardConfig(w_text=1.0, w_sim=2.0, w_mos=0.5, use_stability=False)
    r = _utt("a", cer=0.1, sim=0.8, mos=4.0)
    assert abs(fused_reward(r, cfg) - (1.0 * -0.1 + 2.0 * 0.8 + 0.5 * 4.0)) < 1e-9


# ---- hierarchical group advantages (parity path) -----------------------------------------
def test_homogeneous_group_has_no_grad():
    cfg = RewardConfig()
    group = [_utt(str(i), cer=0.2, sim=0.5, mos=3.0) for i in range(4)]
    adv, has_grad = group_advantages(group, cfg)
    assert has_grad is False
    assert adv == [0.0, 0.0, 0.0, 0.0]


def test_group_advantages_zero_mean_unit_std_when_varied():
    cfg = RewardConfig(use_stability=False)
    group = [
        _utt("a", cer=0.0, sim=0.9, mos=4.5),
        _utt("b", cer=0.5, sim=0.4, mos=2.0),
        _utt("c", cer=0.2, sim=0.6, mos=3.2),
    ]
    adv, has_grad = group_advantages(group, cfg)
    assert has_grad is True
    assert abs(sum(adv)) < 1e-9
    assert abs(_std(adv, 0.0) - 1.0) < 1e-6


# ---- clip schedule -----------------------------------------------------------------------
def test_clip_scheduler_interpolates_then_holds():
    s = ClipSchedule(start=0.2, end=0.3, warmup_steps=200)
    assert abs(s.value(0) - 0.2) < 1e-9
    assert abs(s.value(100) - 0.25) < 1e-9
    assert abs(s.value(200) - 0.3) < 1e-9
    assert abs(s.value(10_000) - 0.3) < 1e-9


def test_clip_scheduler_static_when_equal():
    s = ClipSchedule(start=0.2, end=0.2, warmup_steps=200)
    assert s.value(0) == 0.2 and s.value(500) == 0.2


# ---- text / cer / repetition -------------------------------------------------------------
def test_cer_and_normalization():
    assert normalize_text("Hello, World!") == "hello world"
    assert cer("abc", "abc") == 0.0
    assert cer("abcd", "abxd") == 0.25  # 1 sub / 4 chars
    assert cer("", "anything") is None


def test_has_repetition():
    assert has_repetition("the the the cat cat cat", n=3) is False  # span<n on words... uses spans
    assert has_repetition("please hold on please hold on", n=2) is True
    assert has_repetition("a normal sentence with no loops", n=3) is False
