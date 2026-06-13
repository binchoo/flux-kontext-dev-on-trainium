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

# 4) Record the Neuron-pinned torch version (torch-neuronx requires a SPECIFIC
#    torch, e.g. 2.8.0). We must never let an app dep upgrade past it.
NEURON_TORCH_VER="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
echo "Neuron-pinned torch: ${NEURON_TORCH_VER}"

# 5) App-layer deps. diffusers / peft / the rest are NOT provided by
#    optimum-neuron and pull torch as a dependency — installing them normally
#    upgrades torch (observed: diffusers pulled torch 2.12, breaking the
#    torch-neuronx 2.8 pairing). So install the model libs with --no-deps, then
#    backfill ONLY their pure-python transitive deps (never torch/torchvision).
echo "Installing model libs (--no-deps, cannot touch torch) ..."
python -m pip install --no-deps diffusers peft

# torchvision must match the Neuron torch exactly; install no-deps so it cannot
# drag torch. (0.23.x pairs with torch 2.8.x.)
python -m pip install --no-deps "torchvision==0.23.*" || \
    echo "WARNING: torchvision 0.23.* install failed; verify torch/torchvision pairing manually."

echo "Backfilling pure-python app deps ..."
python -m pip install \
    Pillow tqdm datasets sentencepiece wandb \
    importlib_metadata regex requests filelock pyyaml

# Guard: if anything above bumped torch off the Neuron pin, fail loudly NOW.
python - <<PY
import torch
pinned = "${NEURON_TORCH_VER}"
cur = torch.__version__.split("+")[0]
assert cur == pinned, (
    f"FATAL: torch moved from Neuron pin {pinned} to {cur} during app-dep install. "
    "An app dependency upgraded torch. Re-run: "
    f"pip install --no-deps 'torch=={pinned}' 'torchvision==0.23.*'"
)
print(f"torch still pinned at {cur} after app-dep install — OK")
PY

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

# 7) Sanity check: torch is the Neuron build AND matches torch-neuronx's required
#    version. Checking only "+cu absent" is insufficient — an app dep can bump
#    torch to a newer non-CUDA build (e.g. 2.12) that still breaks the
#    torch-neuronx 2.8 pairing. So we compare torch's major.minor to
#    torch-neuronx's leading version.
python - <<'PY'
import re
import torch, accelerate
import importlib.metadata as md

v = torch.__version__
print(f"torch: {v}")
assert "+cu" not in v, (
    f"FATAL: a CUDA torch ({v}) got installed; the Neuron stack was overwritten. "
    "Re-run after: pip uninstall -y torch torchvision"
)
# torch-neuronx version looks like '2.8.0.2.10...' -> its torch pairing is 2.8.x
tnx = md.version("torch-neuronx")
tnx_mm = ".".join(tnx.split(".")[:2])          # '2.8'
torch_mm = ".".join(v.split("+")[0].split(".")[:2])  # '2.8'
assert torch_mm == tnx_mm, (
    f"FATAL: torch {torch_mm} != torch-neuronx pairing {tnx_mm} (torch-neuronx {tnx}). "
    f"An app dep upgraded torch. Fix: pip install --no-deps 'torch=={tnx_mm}.0' 'torchvision==0.23.*'"
)
print(f"torch matches torch-neuronx pairing ({tnx_mm}): OK")
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
