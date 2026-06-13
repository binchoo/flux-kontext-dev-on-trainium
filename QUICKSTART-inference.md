# FLUX.1-Kontext — Train on GPU, Serve on Neuron (QuickStart)

This is the **supported, recommended** deployment path for FLUX.1-Kontext image
editing: fine-tune on a CUDA GPU using the stock `diffusers` ecosystem, then
compile the model for **AWS Neuron** and serve inference on Inferentia/Trainium via
`optimum-neuron`'s official `NeuronFluxKontextPipeline`.

**Why this path (vs training on Neuron):** `optimum-neuron` officially supports
FLUX/Flux-Kontext for **inference** (it's in the Supported Architectures list), but
does **not** yet support `diffusers` *training* on Neuron. Inference is also where
the cost lives (it runs 24/7 at traffic scale, training is one-off), so moving
inference to Neuron captures most of the cost benefit on the well-paved path.

```
[Phase A] Fine-tune on GPU (CUDA)          stock diffusers, unchanged
   train.py on a GPU box  ->  pytorch_lora_weights.safetensors
        |
[Phase B] Compile for Neuron               optimum-neuron, official
   NeuronFluxKontextPipeline export=True  ->  compiled artifact (.neff)
        |
[Phase C] Serve on Inferentia/Trainium     NeuronFluxKontextPipeline
   pipe(image=..., prompt=...)  ->  edited image
```

> **Scope of this guide:** Phase B + C (the Neuron side). Phase A is standard GPU
> `diffusers` training — use the repo's `train.py` / `train_model.sh` on any CUDA
> GPU; nothing Neuron-specific. This guide starts from "I have (or will get) LoRA
> weights, now compile & serve on Neuron." Base-model inference (no LoRA) works the
> same way and is the fastest way to first prove the serving path.

Verified environment (pin these):

| Item | Value |
|---|---|
| Compile/serve instance | `trn1.32xlarge` or `inf2.24xlarge` (Neuron) |
| Tensor parallel | **8** (docs-recommended for FLUX) |
| Image size | 1024×1024 (baked into the .neff; fixed) |
| Python | 3.10 |
| optimum-neuron | 0.4.5 |
| **diffusers** | **0.35.\*** (hard requirement — see step 3) |
| torch | Neuron build (paired with torch-neuronx; never `+cu`) |

`black-forest-labs/FLUX.1-Kontext-dev` is a **gated** model — HF token + license
acceptance required. (License note: FLUX.1-Kontext [dev] is non-commercial; this
guide is the technical serving path, licensing is the customer's gate.)

---

## Phase A — Fine-tune on GPU (CUDA) — summary

On any CUDA GPU box (NOT the Neuron instance), with stock `diffusers`:

```bash
# standard GPU training — no Neuron involvement
pip install accelerate transformers diffusers peft wandb
./train_model.sh          # produces pytorch_lora_weights.safetensors
```

Output: a LoRA `.safetensors`. Copy it to the Neuron instance for Phase C (optional —
base-model inference needs no LoRA). **Phase A is not covered further here**; it is
ordinary GPU diffusers training.

---

## Phase B — Compile for Neuron (on the trn/inf instance)

### 1. Instance + disk

SageMaker Notebook `ml.trn1.32xlarge` (or `inf2.24xlarge`), **512 GB** EBS.
Open a terminal (JupyterLab → Terminal).

```bash
export HF_HOME=/home/ec2-user/SageMaker/hf_cache
mkdir -p "$HF_HOME"
df -h /home/ec2-user/SageMaker        # expect 400G+ free
```

### 2. Repo + Neuron env

```bash
cd /home/ec2-user/SageMaker
git clone https://github.com/binchoo/flux-kontext-dev-on-trainium.git
cd flux-kontext-dev-on-trainium
./setup_neuron.sh hf_여기에_본인_토큰   # fresh venv + optimum-neuron[neuronx]
source .venv-neuron/bin/activate
```

### 3. Pin diffusers to 0.35.\* (REQUIRED)

optimum-neuron 0.4.5 imports `diffusers.models.controlnet`, a path removed in newer
diffusers. The setup may leave a newer diffusers (e.g. 0.38) installed for training;
for **inference** you must pin 0.35.\* (no-deps so torch stays the Neuron build):

```bash
pip install --no-deps "diffusers==0.35.*"
python -c "from optimum.neuron import NeuronFluxKontextPipeline; print('import OK')"
```

Expect: `import OK`. (If you also do GPU training in this same venv, keep training
in a **separate** venv — training needs newer diffusers, inference needs 0.35.)

### 4. Compile (HF weights → .neff)

```bash
export FLUX_BACKEND=xla
TP=8 HEIGHT=1024 WIDTH=1024 \
COMPILED_DIR=/home/ec2-user/SageMaker/flux_kontext_neuron \
  ./run_infer_neuron.sh compile
```

What happens: loads the gated model, shards across 8 NeuronCores (TP8 — fits the
12B DiT comfortably), and runs `neuronx-cc`. The script passes
`disable_neuron_cache=True` to work around an optimum-neuron 0.4.5 limitation
(its multi-model compile cache only supports unet-based Stable Diffusion, not the
FLUX DiT — it would otherwise raise `NotImplementedError`).

Expect (10–30 min): model-loading bars → `Compiler status PASS` (one or more) →
`saved compiled artifact to /home/ec2-user/SageMaker/flux_kontext_neuron`.

> Ignore the red `TDRV/NRT ... Failed to allocate` lines and `nki_jit deprecated` /
> `aws-ofi-nccl ... EFA` warnings — they are background noise, not the failure. The
> real signal is `Compiler status PASS` and the saved artifact (or a Python
> Traceback if it actually fails).

---

## Phase C — Serve (edit an image)

```bash
export FLUX_BACKEND=xla
COMPILED_DIR=/home/ec2-user/SageMaker/flux_kontext_neuron \
HEIGHT=1024 WIDTH=1024 \
  ./run_infer_neuron.sh serve <input_image.png_or_URL> "Change the background to a green forest"
```

Loads the compiled artifact (no recompile) and runs the edit. Output: `output.png`.

Image size must match compile (1024×1024) — the shape is baked into the .neff.

---

## (Optional) Phase C+ — serve with your fine-tuned LoRA

Base-model serving above proves the path. To serve your Phase-A LoRA, merge it into
the base **before** compiling (Phase B), because the compiled .neff is static:

```python
# one-time, on the GPU box or any CPU box with diffusers:
from diffusers import FluxKontextPipeline
import torch
pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("pytorch_lora_weights.safetensors")
pipe.fuse_lora()                       # bake LoRA into base weights
pipe.save_pretrained("flux_kontext_merged")
```

Then point Phase B at the merged dir (`--model-id ./flux_kontext_merged`). The
compiled artifact now embeds your fine-tune. (LoRA can't be hot-swapped on a
compiled .neff — it must be fused pre-compile.)

---

## What this path avoids (vs training on Neuron)

| Concern | Train-on-Neuron | This path (serve-on-Neuron) |
|---|---|---|
| optimum-neuron support | ❌ diffusers training unsupported | ✅ official `NeuronFluxKontextPipeline` |
| Custom backend/shims | required (9 blockers) | **none** — official API only |
| Where cost is | one-off | **inference = 24/7, the real spend** |

## Known issues handled here

| Issue | Handling |
|---|---|
| `No module named 'diffusers.models.controlnet'` | pin `diffusers==0.35.*` (step 3) |
| `MultiModelCacheEntry ... NotImplementedError` | `disable_neuron_cache=True` (baked into `infer_neuron.py`) |
| 12B DiT vs 16GB/core HBM | tensor_parallel_size=8 (shards the model) |
| red `Failed to allocate` / EFA / nki_jit logs | background noise; watch for `Compiler status PASS` |

## Verification status (honest)

- ✅ `NeuronFluxKontextPipeline` imports; gated model loads (7 components).
- ✅ Cache-entry `NotImplementedError` bypassed via `disable_neuron_cache=True`.
- ⏳ `Compiler status PASS` + served image: confirm on hardware (this is the gate
  this guide drives you to). FLUX inference is officially supported by
  optimum-neuron, so this path is expected to complete — unlike the training path.
