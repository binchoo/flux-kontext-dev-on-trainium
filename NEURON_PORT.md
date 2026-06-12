# FLUX.1-Kontext-dev LoRA Training — AWS Trainium (Neuron) Port

This repo's GPU/CUDA LoRA training script has been ported to run on AWS Trainium
(Neuron/XLA via `torch-neuronx` + `optimum-neuron`), preserving the original CUDA
path behind a backend abstraction (additive / guarded).

## Why FLUX (vs Qwen-Image-Edit)

A prior attempt to port **Qwen-Image-Edit** to Trainium hit a hard wall: its
**Qwen2.5-VL** vision encoder uses **dynamic tensor shapes** (`masked_select`,
`unique_consecutive`, `cu_seqlens` variable-length packing), which `neuronx-cc`
does not support (`set-dimension-size` op rejected — AWS issue #1144).

**FLUX.1-Kontext is fundamentally Neuron-friendly:**
- Text encoders are **T5 + CLIP** with **fixed `max_length` padding** (77 / 512) —
  no dynamic shapes (verified in `utils.py`).
- `FluxTransformer2DModel` (the DiT) uses **dense fixed-shape attention** (plain
  SDPA via `dispatch_attention_fn`, no varlen/`cu_seqlens`), and no
  `masked_select`/`unique`/`nonzero`. Once image size + text length + batch are
  fixed, the entire DiT graph is **static**.
- **Decisive proof**: `optimum-neuron` ships an official, validated FLUX
  implementation including **`NeuronFluxKontextPipeline`** and
  `FluxTransformerNeuronConfig` (all-static `INPUT_ARGS`). FLUX/Flux-Kontext are
  listed as supported architectures.

## What changed (additive / guarded)

| File | Change |
|---|---|
| `neuron_backend.py` (new) | Device/exec abstraction: `cuda/xla/mps/cpu`. cuda forwards to `torch.cuda.*`; xla no-ops cache and provides `mark_step()`. torch_xla lazily imported. Selection via `FLUX_BACKEND` env or autodetect. |
| `neuron_accelerate.py` (new) | `build_accelerator()` → `accelerate.Accelerator` on GPU, `optimum.neuron.NeuronAccelerator` on XLA (drop-in subclass; loop unchanged). |
| `train.py` | `Accelerator(...)` → `build_accelerator(...)`; `mark_step()` per loop iter; `torch.cuda.*` → `backend.*`; `get_sigmas` vectorized on XLA (drops `.nonzero().item()`); 8-bit Adam → AdamW fallback on XLA; autocast device + Generator made backend-aware; validation sampling skipped on XLA (recompile avoidance). |
| `setup_neuron.sh` (new) | Fresh venv + single `optimum-neuron[neuronx,training]` install + constraints lock + sanity check (no `+cu` torch, NeuronAccelerator imports). |
| `requirements-neuron.txt` (new) | App-layer deps only; excludes CUDA-only (bitsandbytes/flash-attn) and Neuron-owned packages. |
| `run_neuron.sh` (new) | torchrun launcher (precompile → train), fixed shapes (512×512, seq 512, bs 1). |
| `neuron_static_check.py` (new) | Static verification harness (13 checks; runs without hardware). |

## CUDA path preserved

Every change branches on the backend. When the backend resolves to `cuda`, all
helpers forward to the original `torch.cuda.*` calls and `build_accelerator`
returns a vanilla `Accelerator` — GPU behaviour is unchanged.

## Run on Trainium

```bash
./setup_neuron.sh $HF_TOKEN      # fresh venv + Neuron stack (FLUX.1-Kontext is gated)
source .venv-neuron/bin/activate
export FLUX_BACKEND=xla
./run_neuron.sh                  # neuron_parallel_compile → torchrun
```

Keep image size / text length / batch fixed so `neuronx-cc` compiles one static
graph.

## Verification status (no hardware here)

- ✅ All files `py_compile` clean.
- ✅ `neuron_static_check.py`: Tier A (lint, artifacts, port-edits) + Tier B
  (real torch + simulated torch_xla: backend selection, device/mark_step) — 13/13.
- ⏳ Not yet run on hardware: actual `neuron_parallel_compile` + `torchrun`,
  1-step compile, convergence. The architecture-level risk (dynamic shapes that
  blocked Qwen) is **absent** here, and optimum-neuron's official FLUX-Kontext
  support is strong prior evidence the DiT compiles.

## XLA risks to watch on hardware

- Keep **fixed shapes** (height/width/max_sequence_length/batch) — any variation
  recompiles.
- `get_sigmas` vectorized path assumes each timestep matches exactly one schedule
  entry (true for the flow-match scheduler).
- Validation sampling is skipped on XLA by default; re-enable only with bucketed
  shapes compiled separately.
- bf16 base + fp32 LoRA/optimizer state — standard, but validate convergence.
