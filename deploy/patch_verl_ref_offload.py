"""Patch verl's FSDP engine so a forward_only engine (the REF policy) uses verl's MANUAL param
offload (load full params to GPU before the forward, offload after) instead of FSDP-native
CPUOffload.

Why: the Qwen3-TTS actor/ref forward (`qwen3_tts_forward.assemble_talker_embeddings`) assembles the
talker input by indexing leaf embedding tables (text_embedding, codec_embedding, sub-codebooks) and
running text_projection DIRECTLY — i.e. OUTSIDE the FSDP-wrapped module forward. FSDP-native
CPUOffload only copies a unit's params to GPU inside that unit's forward, so those leaf params stay on
CPU and the ref forward dies with:
    RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:0,
    different from other tensors on cpu (... wrapper_CUDA__index_select)  [compute_ref_log_prob]
The ACTOR works because it is NOT forward_only -> verl manual offload loads ALL params to GPU before
the forward. This makes the ref take the same proven path. Memory profile is equivalent (ref is loaded
to GPU only for its forward, then offloaded). Idempotent.

    python deploy/patch_verl_ref_offload.py
"""

import pathlib
import sys

import verl

F = pathlib.Path(verl.__file__).parent / "workers" / "engine" / "fsdp" / "transformer_impl.py"
s = F.read_text()

# 1) FSDP construction: drop the forced CPUOffload + the _is_offload_param=False override for
#    forward_only, so the ref keeps verl manual offload (param_offload from config).
OLD1 = (
    "            cpu_offload = None\n"
    "            if self.engine_config.forward_only:\n"
    "                cpu_offload = CPUOffload(offload_params=True)\n"
    "                self._is_offload_param = False\n"
    "                self._is_offload_optimizer = False\n"
)
NEW1 = (
    "            cpu_offload = None\n"
    "            if self.engine_config.forward_only:\n"
    "                # qwen3_tts patch: ref uses verl MANUAL offload (param_offload), not FSDP-native\n"
    "                # CPUOffload, so the custom talker forward finds leaf embedding params on GPU.\n"
    "                self._is_offload_optimizer = False\n"
)

# 2) to(): the forward_only early-return skips the manual load/offload. Remove it so manual offload
#    (now enabled for the ref) actually loads params to GPU before the forward.
OLD2 = (
    "        if self.engine_config.forward_only:\n"
    "            # force cpu_offload\n"
    "            return\n"
)
NEW2 = (
    "        # qwen3_tts patch: forward_only ref now uses manual offload; do NOT early-return here.\n"
)

changed = False
if "qwen3_tts patch: ref uses verl MANUAL offload" in s:
    print("verl ref-offload already patched (construction)")
elif OLD1 in s:
    s = s.replace(OLD1, NEW1)
    changed = True
    print("patched verl FSDP construction (ref manual offload)")
else:
    print("WARN: ref CPUOffload construction block not found — verl layout changed", file=sys.stderr)
    sys.exit(1)

if "qwen3_tts patch: forward_only ref now uses manual offload" in s:
    print("verl ref-offload already patched (to())")
elif OLD2 in s:
    s = s.replace(OLD2, NEW2)
    changed = True
    print("patched verl FSDP to() (drop forward_only early-return)")
else:
    print("WARN: forward_only early-return in to() not found — verl layout changed", file=sys.stderr)
    sys.exit(1)

if changed:
    F.write_text(s)
    print(f"wrote {F}")
