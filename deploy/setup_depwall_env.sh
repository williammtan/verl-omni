#!/usr/bin/env bash
# Reproduce the qwen-tts <-> vLLM-Omni coexistence env on top of williammtan/verl-omni-tts:v4.
# This is NOT baked into the image: a fresh pod has tf 5.8.1 / hub 1.21.0 and no qwen-tts. Run once
# per pod (idempotent). See examples/qwen3_tts_grpo_trainer/SOLVED_DEPENDENCY_WALL.md for the why.
#
#   bash deploy/setup_depwall_env.sh
set -euo pipefail
cd /workspace/verl-omni
PY=.venv/bin/python
PIP=.venv/bin/pip

cur_tf=$($PY -c "import transformers;print(transformers.__version__)" 2>/dev/null || echo none)
if [ "$cur_tf" != "4.57.3" ]; then
  echo "[depwall] downgrading transformers $cur_tf -> 4.57.3 (pulls huggingface_hub<1.0)"
  $PIP install "transformers==4.57.3"
else
  echo "[depwall] transformers already 4.57.3"
fi

if ! $PY -c "import qwen_tts" 2>/dev/null; then
  echo "[depwall] installing qwen-tts==0.1.1 (+ sox), no deps"
  $PIP install --no-deps "qwen-tts==0.1.1" sox
else
  echo "[depwall] qwen_tts already importable"
fi

# Drop the huggingface_hub PEP-604 union validator shim into the venv site-packages. The .pth runs
# at interpreter startup in the driver AND every spawned vllm worker (a sitecustomize is shadowed).
SP=$($PY -c "import site;print(site.getsitepackages()[0])")
cp deploy/hf_hub_unionfix.py  "$SP"/hf_hub_unionfix.py
cp deploy/hf_hub_unionfix.pth "$SP"/hf_hub_unionfix.pth
echo "[depwall] shim installed at $SP"

echo "[depwall] verifying coexistence ..."
$PY - <<'EOF'
import transformers, huggingface_hub as h
print("  tf", transformers.__version__, "hub", h.__version__)
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration  # noqa: F401
print("  qwen-tts model class: OK")
import vllm_omni  # noqa: F401
print("  vllm_omni import: OK")
EOF
echo "[depwall] DONE"
