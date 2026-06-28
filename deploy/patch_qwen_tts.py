"""Patch qwen-tts 0.1.1 modeling code to run on transformers 5.x. Idempotent.

Usage: python patch_qwen_tts.py <qwen_tts_pkg_dir>
"""
import re, sys, pathlib

QT = pathlib.Path(sys.argv[1])
MODELING = QT / "core/models/modeling_qwen3_tts.py"
TOK = QT / "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"

def edit(path, fn):
    s = path.read_text(); ns = fn(s)
    if ns != s:
        path.write_text(ns); print(f"patched {path.name}")
    else:
        print(f"nochange {path.name}")

# 1) tf5 made check_model_inputs a bare decorator (sig: (func)).
for p in [TOK, MODELING]:
    if p.exists():
        edit(p, lambda s: s.replace("@check_model_inputs()", "@check_model_inputs"))

# 2) tf5 PreTrainedConfig attribute access is strict (no default pad_token_id).
edit(MODELING, lambda s: s.replace("config.pad_token_id", 'getattr(config, "pad_token_id", None)'))

# 3) tf5 dropped 'default' from ROPE_INIT_FUNCTIONS. Inject a canonical default-rope init
#    (inv_freq = 1/theta^(2i/dim), scaling 1.0) and route 'default'/None to it. mRoPE position
#    application is unchanged (this only sets inv_freq init).
def patch_rope(s):
    helper = (
        "\ndef _qtts_default_rope_init(config, device=None, **kw):\n"
        "    import torch\n"
        "    base = getattr(config, 'rope_theta', 10000.0)\n"
        "    dim = getattr(config, 'head_dim', None) or (config.hidden_size // config.num_attention_heads)\n"
        "    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))\n"
        "    return inv_freq, 1.0\n"
    )
    if "_qtts_default_rope_init" not in s:
        # insert helper at top level, right before the rotary class def (always safe).
        anchor = "class Qwen3TTSTalkerRotaryEmbedding"
        i = s.find(anchor)
        if i != -1:
            s = s[:i] + helper + "\n\n" + s[i:]
        else:
            s = helper + s
    s = s.replace(
        "self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]",
        "self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type) or _qtts_default_rope_init",
    )
    return s
edit(MODELING, patch_rope)
print("PATCH_DONE")
