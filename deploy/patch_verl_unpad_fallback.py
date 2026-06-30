"""Patch verl's ``_get_attention_functions`` to fall back to transformers' reference unpad/pad helpers
when ``flash_attn`` is not installed (no cu130 wheel; building it is slow/risky).

verl hard-imports ``from flash_attn.bert_padding import ...`` on CUDA with no fallback
(`verl/utils/attention_utils.py`), used by ``left_right_2_no_padding`` for the padding<->no-padding
conversion. This recipe runs ``attn_implementation: sdpa`` + ``use_remove_padding: false``, so the
flash-attn *kernels* are never needed — only these padding helpers. We must NOT install a fake
``flash_attn`` package (that makes vllm think flash-attn exists and breaks its attention backend);
instead we patch verl's import to use ``transformers.modeling_flash_attention_utils`` as a fallback.
Idempotent.

    python deploy/patch_verl_unpad_fallback.py
"""

import pathlib
import sys

import verl

F = pathlib.Path(verl.__file__).parent / "utils" / "attention_utils.py"
s = F.read_text()
OLD = "        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input"
NEW = (
    "        try:\n"
    "            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input\n"
    "        except ImportError:\n"
    "            from einops import rearrange\n"
    "            from transformers.modeling_flash_attention_utils import (\n"
    "                _index_first_axis as index_first_axis,\n"
    "                _pad_input as pad_input,\n"
    "                _unpad_input as unpad_input,\n"
    "            )"
)

if "except ImportError:" in s and "_unpad_input as unpad_input" in s:
    print("verl unpad fallback already patched")
elif OLD in s:
    F.write_text(s.replace(OLD, NEW))
    print(f"patched verl unpad fallback (transformers) in {F}")
else:
    print("WARN: verl flash_attn import line not found — layout changed", file=sys.stderr)
    sys.exit(1)
