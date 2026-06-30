"""Patch verl's calculate_debug_metrics to dump sample-0 per-token actor vs rollout log-probs
(gated on VERL_TTS_PROBDUMP) so we can localize the rollout<->actor on-policy divergence offline.
Idempotent. The dump point already has both aligned with the response mask — no batch reconstruction.

    python deploy/patch_verl_probdump.py
"""

import pathlib
import sys

import verl

F = pathlib.Path(verl.__file__).parent / "utils" / "debug" / "metrics.py"
s = F.read_text()
ANCHOR = "    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)\n"
DUMP = (
    "    import os as _os\n"
    "    if _os.environ.get('VERL_TTS_PROBDUMP'):\n"
    "        try:\n"
    "            _d = _os.environ.get('VERL_TTS_DUMP_DIR', '/weka/whiplash-grpo/tts/dump')\n"
    "            _os.makedirs(_d, exist_ok=True)\n"
    "            torch.save({'actor_logp': actor_old_log_probs[0].detach().float().cpu(),\n"
    "                        'rollout_logp': rollout_old_log_probs[0].detach().float().cpu(),\n"
    "                        'response_mask': response_mask[0].detach().cpu(),\n"
    "                        'responses': responses[0].detach().cpu()},\n"
    "                       _os.path.join(_d, 'probdump_0.pt'))\n"
    "            print('[probdump] wrote probdump_0.pt', flush=True)\n"
    "        except Exception as _e:\n"
    "            print(f'[probdump] {_e!r}', flush=True)\n"
)

if "[probdump]" in s:
    print("verl probdump already patched")
elif ANCHOR in s:
    F.write_text(s.replace(ANCHOR, ANCHOR + DUMP))
    print(f"patched verl probdump in {F}")
else:
    print("WARN: anchor not found in metrics.py — layout changed", file=sys.stderr)
    sys.exit(1)
