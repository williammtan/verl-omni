"""Patch whiplash's trainer to offload reward scoring to the HTTP reward server.

When REWARD_URL is set, the trainer uses a remote client (POSTs decoded audio, gets back metrics)
instead of the in-process RewardScorer — so no reward models load on the trainer GPU. This frees
trainer compute/memory AND removes the multi-rank reward-model-load deadlock (unlocks DDP).

Usage: python patch_remote_reward.py <whiplash_pkg_dir>   (idempotent)
"""
import sys, pathlib

QT = pathlib.Path(sys.argv[1])
F = QT / "train/grpo_trainer.py"
s = orig = F.read_text()

_CLASS = '''
class _RemoteRewardScorer:
    """Drop-in scorer that POSTs rollout audio to the HTTP reward server (verl-omni-style)."""

    def __init__(self, url: str):
        import requests
        self.url = url.rstrip("/")
        self._s = requests.Session()

    def score_batch(self, clips):
        import base64, os
        import numpy as np
        from ..parse import UttResult
        ref = os.environ.get("REF_AUDIO")
        payload = {"clips": [{
            "id": c["id"], "text": c.get("text", ""), "sr": int(c["sr"]),
            "ref_audio": ref, "group_key": c.get("group_key"),
            "wav_b64": base64.b64encode(np.asarray(c["wav"], dtype=np.float32).tobytes()).decode(),
        } for c in clips]}
        r = self._s.post(f"{self.url}/score_batch", json=payload, timeout=900).json()
        out = []
        for x in r["results"]:
            out.append(UttResult(
                id=x["id"], cer=x.get("cer"), spk_similarity=x.get("spk_similarity"),
                emotion_similarity=x.get("emotion_similarity"), utmos=x.get("utmos"),
                duration_s=x.get("duration_s"), synth_ok=x.get("synth_ok", True),
                truncated=x.get("truncated", False), repeated=x.get("repeated", False),
                cer_outlier=x.get("cer_outlier", False), hyp_text=x.get("hyp_text"),
            ))
        return out


'''

if "_RemoteRewardScorer" not in s:
    i = s.index("def run_training(")
    s = s[:i] + _CLASS + s[i:]

# Route to the remote scorer when REWARD_URL is set (else keep in-process RewardScorer).
old = "    scorer = scorer or RewardScorer(device=str(device))"
new = ("    import os as _os\n"
       "    if scorer is None and _os.environ.get('REWARD_URL'):\n"
       "        scorer = _RemoteRewardScorer(_os.environ['REWARD_URL'])\n"
       "    scorer = scorer or RewardScorer(device=str(device))")
assert old in s, "scorer construction line not found — whiplash layout changed"
s = s.replace(old, new)

if s != orig:
    F.write_text(s)
    print("patched grpo_trainer.py (remote reward)")
else:
    print("nochange")
