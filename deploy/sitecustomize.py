"""Interpreter-startup hook: apply the huggingface_hub<1.0 strict-union backport.

Python auto-imports a top-level `sitecustomize` module at startup (before any user
imports), in the main process and in every spawned subprocess. Dropping this file on
sys.path (e.g. the venv's site-packages) guarantees the hub union shim is active before
`import vllm_omni` runs anywhere — including vllm's worker subprocesses.

This makes vllm-omni 0.22 importable alongside qwen-tts (transformers 4.57.3 + hub<1.0)
in a single env. See deploy/hf_hub_unionfix.py for the full rationale.

Kept dependency-free and fail-open: any error here must NOT break interpreter startup.
"""
try:
    import hf_hub_unionfix  # type: ignore

    hf_hub_unionfix.apply()
except Exception:
    # Never let a startup hook crash the interpreter. If the shim module is missing or
    # hub internals moved, fall through silently; the offending import will surface its
    # own clear error.
    pass
