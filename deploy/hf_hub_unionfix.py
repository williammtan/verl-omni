"""Backport of the huggingface_hub v1.3.0 strict-dataclass union fix to hub<1.0.

WHY THIS EXISTS
---------------
verl-omni's rollout backend (vllm-omni 0.22 / vllm 0.22) and the Qwen3-TTS trainer
(qwen-tts 0.1.1, which hard-pins transformers==4.57.3) must share ONE Python env.
transformers 4.57.3 requires huggingface_hub<1.0; transformers 5.x requires hub>=1.0.
On transformers 5.x the Qwen3-TTS model emits ~85-99% silent audio (QwenLM/Qwen3-TTS
issue #237 / PR #201), so the model side MUST stay on transformers 4.57.x + hub<1.0.

The ONLY thing that breaks `import vllm_omni` on hub<1.0 is a single validator bug:
huggingface_hub's @strict dataclass type-validator (`huggingface_hub/dataclasses.py`)
registers only `typing.Union` in `_BASIC_TYPE_VALIDATORS`, not `types.UnionType` (the
PEP-604 `X | Y` syntax). The `kernels` package (pulled in by the transformers stack)
declares `@strict`-validated `PythonPackage(import_name: str | None)`, instantiated at
import of `transformers.integrations.hub_kernels`. On hub 0.36.2 that raises:

    StrictDataclassFieldValidationError: Unsupported type for field 'import_name': str | None

huggingface_hub fixed this in v1.3.0 with exactly two lines: `import types` and
`_BASIC_TYPE_VALIDATORS[types.UnionType] = _validate_union`. We backport that one map
entry onto hub<1.0. `type_validator` reads the module-global dict by name on every call,
so inserting the key is sufficient — no function needs replacing. `_validate_union`
already handles `get_args(str | None) == (str, NoneType)` correctly.

USAGE
-----
Import this module once before importing vllm/vllm_omni. The canonical install is a
`sitecustomize.py` on sys.path that calls `apply()` (see deploy/sitecustomize.py), so the
patch is active at interpreter startup in the main process AND every vllm worker
subprocess. Calling `apply()` more than once is a no-op (setdefault).
"""
from __future__ import annotations

import types


def apply() -> bool:
    """Register types.UnionType in hub's strict validator table. Returns True if patched.

    Safe to call unconditionally: no-op on Python<3.10 (no UnionType), on hub>=1.3 (key
    already present), and on any hub layout where the expected symbols are absent.
    """
    if not hasattr(types, "UnionType"):  # Python < 3.10
        return False
    try:
        from huggingface_hub import dataclasses as _hf_dc
    except Exception:
        return False
    table = getattr(_hf_dc, "_BASIC_TYPE_VALIDATORS", None)
    validate_union = getattr(_hf_dc, "_validate_union", None)
    if table is None or validate_union is None:
        return False
    table.setdefault(types.UnionType, validate_union)
    return True


if __name__ == "__main__":
    print("hf_hub_unionfix.apply() ->", apply())
