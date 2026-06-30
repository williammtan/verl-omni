"""Patch vLLM-Omni's AR model runner so a SINGLE-stage Qwen3-TTS talker can run its AR decode loop.

The talker is MTP-style: each decode step needs a `postprocess` (feeding `hidden_states['last']` back).
`gpu_ar_model_runner._resolve_pooler_payload_req_ids` only runs the postprocess when the stage has a
downstream consumer (`omni_final_stage_id > 0`) OR `engine_output_type == "audio"` (the single-final-stage
"rescue"). For RL rollout we want a single talker stage emitting `codec` (keeps it a *generation* model;
`audio` makes vllm reject generation params), so we extend the rescue to fire for `codec`/`latent` too —
the per-step postprocess then runs and decode proceeds, with no 2-stage code2wav connector (which
deadlocks in the colocated verl-omni server). Idempotent.

    python deploy/patch_vllm_omni_ar_rescue.py
"""

import pathlib
import sys

import vllm_omni

F = pathlib.Path(vllm_omni.__file__).parent / "worker" / "gpu_ar_model_runner.py"
s = F.read_text()
OLD = 'if engine_output_type == "audio" and not downstream_req_ids:'
NEW = 'if engine_output_type in ("audio", "codec", "latent") and not downstream_req_ids:'

if NEW in s:
    print("vllm_omni AR rescue already patched for codec/latent")
elif OLD in s:
    F.write_text(s.replace(OLD, NEW))
    print(f"patched AR rescue (codec/latent) in {F}")
else:
    print("WARN: AR rescue line not found — vllm_omni layout changed", file=sys.stderr)
    sys.exit(1)
