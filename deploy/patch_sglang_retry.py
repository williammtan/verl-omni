"""Patch whiplash's SglangRolloutEngine._gen_one to retry transient sglang errors.

A single sglang HTTP 500 / connection error (e.g. while the rollout Deployment auto-restarts after
an OOM) otherwise kills the whole multi-day run. Retry with backoff long enough to ride out a
restart (~2-3 min). Idempotent.

Usage: python patch_sglang_retry.py <whiplash_pkg_dir>
"""
import sys, pathlib

QT = pathlib.Path(sys.argv[1])
F = QT / "train/grpo_trainer.py"
s = orig = F.read_text()

old = (
    "        r = self._session.post(f\"{self.server_url}/v1/audio/speech\", json=payload, timeout=self.request_timeout)\n"
    "        r.raise_for_status()"
)
new = (
    "        import time as _t\n"
    "        _last = None\n"
    "        for _attempt in range(20):  # ride out sglang restarts (~2-3 min); a 500 must not kill the run\n"
    "            try:\n"
    "                r = self._session.post(f\"{self.server_url}/v1/audio/speech\", json=payload, timeout=self.request_timeout)\n"
    "                r.raise_for_status()\n"
    "                break\n"
    "            except Exception as _e:  # noqa: BLE001\n"
    "                _last = _e\n"
    "                if _attempt == 19:\n"
    "                    raise\n"
    "                _t.sleep(min(15.0, 3.0 + _attempt * 2.0))\n"
)

if "ride out sglang restarts" not in s:
    assert old in s, "_gen_one POST/raise block not found — whiplash layout changed"
    s = s.replace(old, new)

if s != orig:
    F.write_text(s)
    print("patched grpo_trainer.py (sglang retry)")
else:
    print("nochange")
