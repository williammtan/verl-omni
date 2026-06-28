"""Patch qwen-tts 0.1.1 modeling code to run on transformers 5.x. Idempotent.

Usage: python patch_qwen_tts.py <qwen_tts_pkg_dir>
"""
import re, sys, pathlib

QT = pathlib.Path(sys.argv[1])
# All modeling files that use transformers internals (talker + speech-tokenizer decoder).
FILES = [
    QT / "core/models/modeling_qwen3_tts.py",
    QT / "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py",
]

_ROPE_HELPER = (
    "def _qtts_default_rope_init(config, device=None, **kw):\n"
    "    import torch\n"
    "    base = getattr(config, 'rope_theta', 10000.0) or 10000.0\n"
    "    dim = getattr(config, 'head_dim', None) or (config.hidden_size // config.num_attention_heads)\n"
    "    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))\n"
    "    return inv_freq, 1.0\n\n\n"
)

def patch_file(path):
    if not path.exists():
        print("skip (missing)", path.name); return
    s = orig = path.read_text()
    # 1) tf5 made check_model_inputs a bare decorator.
    s = s.replace("@check_model_inputs()", "@check_model_inputs")
    # 2) tf5 strict config attribute access (no default pad_token_id).
    s = s.replace("config.pad_token_id", 'getattr(config, "pad_token_id", None)')
    # 3) tf5 dropped 'default' from ROPE_INIT_FUNCTIONS -> route default/None to a canonical init.
    if "ROPE_INIT_FUNCTIONS[self.rope_type]" in s:
        if "_qtts_default_rope_init" not in s:
            m = re.search(r"^class ", s, re.M)  # before first top-level class = always safe
            i = m.start() if m else 0
            s = s[:i] + _ROPE_HELPER + s[i:]
        s = s.replace(
            "ROPE_INIT_FUNCTIONS[self.rope_type]",
            "(ROPE_INIT_FUNCTIONS.get(self.rope_type) or _qtts_default_rope_init)",
        )
    if s != orig:
        path.write_text(s); print("patched", path.name)
    else:
        print("nochange", path.name)

for f in FILES:
    patch_file(f)
print("PATCH_DONE")
