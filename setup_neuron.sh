#!/bin/bash
#
# FLUX.1-Kontext-dev LoRA training — AWS Trainium (Neuron) environment setup.
# Usage: ./setup_neuron.sh [HF_TOKEN]
#
# Run on a trn1 / trn2 instance with the Neuron driver/runtime present (Neuron
# DLAMI, or follow the AWS Neuron setup guide). Creates a fresh venv and installs
# the Neuron PyTorch stack via a SINGLE consolidated optimum-neuron install, so
# torch (Neuron build) / torch-neuronx / neuronx-cc / transformers / accelerate
# are mutually compatible. Does NOT install CUDA / bitsandbytes / flash-attn.
#
# optimum-neuron officially supports FLUX / Flux-Kontext (NeuronFluxKontext
# pipeline, FluxTransformerNeuronConfig), so this targets a supported arch.

set -euo pipefail

HF_TOKEN=${1:-${HF_TOKEN:-}}
NEURON_PIP_INDEX="https://pip.repos.neuron.amazonaws.com"

echo "=== Setting up FLUX.1-Kontext-dev training for AWS Trainium (Neuron) ==="

# 1) Verify Neuron runtime.
if ! command -v neuron-ls >/dev/null 2>&1; then
    echo "WARNING: 'neuron-ls' not found. Ensure the Neuron driver/runtime is installed"
    echo "         (use the AWS Neuron DLAMI, or follow the Neuron setup guide)."
else
    echo "Detected NeuronCores:"; neuron-ls || true
fi

# 2) Fresh venv (avoids inheriting a broken/CUDA torch).
PYTHON=${PYTHON:-python3}
echo "Python: $("$PYTHON" --version 2>&1) at $(command -v "$PYTHON")"
if [ -d ".venv-neuron" ]; then
    echo "Removing existing .venv-neuron for a clean install ..."
    rm -rf .venv-neuron
fi
"$PYTHON" -m venv .venv-neuron
# shellcheck disable=SC1091
source .venv-neuron/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Neuron pip index for this script run only (not written to global pip config).
export PIP_EXTRA_INDEX_URL="${NEURON_PIP_INDEX}"

# 3) Install the ENTIRE Neuron stack in one consolidated, compatible resolve.
#    --upgrade-strategy eager pulls the matching Neuron torch instead of leaving
#    a stale CUDA torch in place. [training] pins accelerate==1.8.1 etc. that
#    NeuronAccelerator was written against; [neuronx] pulls torch-neuronx/cc.
echo "Installing optimum-neuron[neuronx,training] ..."
python -m pip install --upgrade --upgrade-strategy eager "optimum-neuron[neuronx,training]"

# 4) Freeze Neuron-owned packages into a constraints file so the app deps below
#    can resolve transitive deps WITHOUT changing torch/transformers/accelerate.
echo "Freezing Neuron stack into constraints.txt ..."
CONSTRAINTS="$(mktemp)"
python -m pip freeze | grep -iE '^(torch|torch-xla|torch-neuronx|torchvision|neuronx-cc|neuronx-distributed|libneuronxla|transformers|tokenizers|accelerate|peft|diffusers|optimum-neuron|safetensors|huggingface-hub|numpy)==' \
    > "${CONSTRAINTS}" || true
echo "--- constraints ---"; cat "${CONSTRAINTS}"; echo "-------------------"

# 5) App-layer deps under the Neuron constraints.
echo "Installing app requirements under Neuron constraints ..."
python -m pip install -c "${CONSTRAINTS}" -r requirements-neuron.txt
rm -f "${CONSTRAINTS}"

# 6) Optional Hugging Face auth (FLUX.1-Kontext-dev is a gated model).
if [ -n "${HF_TOKEN}" ]; then
    python - <<PY
from huggingface_hub import login
login(token="${HF_TOKEN}")
print("Hugging Face auth OK")
PY
else
    echo "No HF_TOKEN provided; run 'huggingface-cli login' later (FLUX.1-Kontext-dev is gated)."
fi

# 7) Sanity check: NOT a CUDA torch, and the Neuron FLUX pieces import.
python - <<'PY'
import torch, accelerate
v = torch.__version__
print(f"torch: {v}")
assert "+cu" not in v, (
    f"FATAL: a CUDA torch ({v}) got installed; the Neuron stack was overwritten. "
    "Re-run after: pip uninstall -y torch torchvision"
)
print(f"accelerate: {accelerate.__version__}")
try:
    import torch_xla  # noqa: F401
    print("torch_xla: OK")
except Exception as e:
    print(f"WARNING: torch_xla import failed: {e!r}")
from optimum.neuron import NeuronAccelerator  # noqa: F401
print("optimum.neuron.NeuronAccelerator: OK")
print("=== Neuron stack sanity check PASSED ===")
PY

cat <<'EOF'

=== Setup complete ===

Next steps (on the trn instance):
  source .venv-neuron/bin/activate
  export FLUX_BACKEND=xla         # or rely on autodetect (NEURON_RT_VISIBLE_CORES)
  ./run_neuron.sh

NOTE: FLUX.1-Kontext-dev is a ~24GB model (gated). Ensure HF auth + ~60GB free
disk for weights + cache. Keep image size and text length FIXED so neuronx-cc
compiles one static graph.
EOF
