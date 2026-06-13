# PR/FAQ — First-Class `diffusers` Training Support in `optimum-neuron`

> **Status:** Draft proposal (internal). Working-backwards document.
> **Author:** AWS SA (field-validated on trn1.32xlarge, Jun 2026)
> **One-line ask:** Add first-class `diffusers` *training* support to `optimum-neuron`
> so media/creative customers can fine-tune open-source image-editing models on
> Trainium — starting with the model customers actually ask for, **Qwen-Image-Edit
> (Apache-2.0)** — with the same ease they already enjoy for *inference*.

---

## PRESS RELEASE

### AWS makes fine-tuning open-source image-editing models on Trainium as easy as inference

**SEATTLE — AWS today announced that `optimum-neuron` now supports training and
fine-tuning Hugging Face `diffusers` models on AWS Trainium.** Media, advertising,
and creative-tooling customers can now fine-tune permissively-licensed image-editing
models — including **Qwen-Image-Edit (Apache-2.0)**, which they can serve commercially —
on Trainium with a single configuration change, instead of hand-clearing a long tail
of integration blockers.

Until now, `optimum-neuron` shipped polished **inference** pipelines for diffusion
models but no **training** path. A field PoC exposed two distinct gaps:

1. **The integration gap (closeable today).** Fine-tuning a `diffusers` model failed
   because the accelerator assumed text-only (transformers) models, dependency
   resolution silently broke the Neuron PyTorch pairing, and host-memory handling was
   left to the user. We proved this is purely an *ecosystem-maturity* gap, not a
   hardware limit, by porting **FLUX.1-Kontext** end-to-end: once the integration
   blockers were cleared, **`neuronx-cc` compiled the model and produced a Neuron
   `.neff` artifact.**

2. **The dynamic-shape gap (the customer's actual model).** The model customers ask
   for, **Qwen-Image-Edit**, additionally uses a vision-language text encoder
   (Qwen2.5-VL) built on *dynamic tensor shapes*, which `neuronx-cc` does not support.
   This requires a second capability — static-shape handling for VLM encoders — beyond
   the diffusers integration.

With this release, AWS closes gap #1 in the framework and provides a path for gap #2,
turning "the customer tried, hit nine errors, and went back to GPU" into "point
`optimum-neuron` at the model and train."

---

## CUSTOMER PROBLEM (why this matters now)

Media/creative ML customers do not just *run* image models — they **fine-tune** them
on proprietary catalogs (products, characters, brand styles). That fine-tuning is the
differentiating workload.

The model these customers are actively requesting is **Qwen-Image-Edit**, for two
concrete reasons:
- **License:** it is **Apache-2.0** — they can fine-tune *and serve the results
  commercially* with no licensing gate. (Contrast: FLUX.1-Kontext [dev] is
  non-commercial; SD3.5 has a revenue threshold.)
- **Capability:** it is a state-of-the-art instruction-based image-editing model with
  strong text rendering.

Today on Trainium:
- **Inference** of diffusion models is first-class (`optimum-neuron` ships pipelines).
- **Training** is not supported for `diffusers` at all.
- **Qwen-Image-Edit specifically** is blocked twice over: the missing diffusers training
  path *and* its dynamic-shape VLM encoder.

The result is a **followership/DX gap**: the exact model the customer wants exists,
runs on GPU day one, and stalls on Trainium at "tried, errored, left."

---

## WHAT WE LEARNED (field evidence)

We ran two real ports on `trn1.32xlarge` to separate the two gaps.

### Case A — FLUX.1-Kontext: proves the integration gap is closeable
FLUX uses fixed-length T5/CLIP encoders and a static-shape DiT. After clearing the
integration blockers, **`neuronx-cc` compiled `FluxTransformer2DModel` successfully
(a `.neff` was generated).** This is the existence proof: **once `optimum-neuron`
supports diffusers training, a static-shape diffusion model trains on Trainium.**

Integration blockers we cleared by hand (these are what framework support removes):

| # | Blocker | Root cause | Removed by diffusers support? |
|---|---|---|---|
| 1 | `NeuronAccelerator` can't `prepare()` a diffusers model | assumes transformers | ✅ Yes |
| 2 | `AttributeError: FluxTransformer2DModel has no 'tie_weights'` | transformers-only call | ✅ Yes |
| 3 | pinned host-memory / fp32 full-model load failures | no diffusers mem mgmt | ✅ Mostly |
| 4 | `diffusers` not pulled by `[neuronx,training]` extra | packaging | ✅ Yes |
| 5 | app deps upgraded torch off the torch-neuronx pin (2.8→2.12) | no constraint guard | ✅ Yes |
| 6 | flash-attn / 8-bit Adam (CUDA-only) defaults | upstream | ✅ Yes (auto-fallback) |
| 7 | `get_sigmas` `.nonzero().item()` host-sync | diffusers training util | ✅ Yes (vectorized util) |

### Case B — Qwen-Image-Edit: the customer's model, with one more gap
Qwen-Image-Edit is **Apache-2.0** and is what customers request. But its **Qwen2.5-VL
vision encoder** uses dynamic, data-dependent shapes (`masked_select`,
`unique_consecutive`, `cu_seqlens` variable-length packing). `neuronx-cc` rejects the
resulting `set-dimension-size` op — this is an architectural property of native-resolution
VLM encoders, not a bug, and is a known open AWS Neuron issue.

**So Qwen-Image-Edit needs both:** (1) the diffusers training integration (same as FLUX),
**and** (2) static-shape handling for its VLM text encoder (fixed-resolution / padded
encoding, or an NKI/optimum-neuron static encoder path — note `optimum-neuron` already
hand-writes static encoders for inference of Qwen2-VL).

---

## WHAT WE PROPOSE

A two-track investment, ordered by leverage:

**Track 1 — `diffusers` training in `optimum-neuron` (unblocks the static-shape models now):**
1. **`diffusers`-aware `NeuronAccelerator.prepare()`** — detect `diffusers.ModelMixin`;
   skip transformers-only calls (`tie_weights`); handle bf16 load + device placement.
2. **A `[diffusers]` install extra** pinned compatibly with the Neuron torch stack.
3. **`NeuronDiffusionTrainer`** mirroring the existing inference pipelines, covering the
   DiT family (FLUX, SD3, Qwen-Image DiT).
4. **Tensor-parallel training for DiT** so 12B+ models scale past the data-parallel
   host-RAM ceiling.

**Track 2 — static-shape VLM encoder support (unblocks Qwen-Image-Edit specifically):**
5. Provide a static-shape encoding path for Qwen2.5-VL-class encoders (fixed-resolution
   padding / dense attention), reusing the static encoder work `optimum-neuron` already
   does for VLM **inference**, so the **Apache-2.0 model customers want** trains on Trainium.

**Plus:** a reference "Fine-tune Qwen-Image-Edit / FLUX on Trainium" runbook.

---

## FREQUENTLY ASKED QUESTIONS

**Q: Why lead with Qwen-Image-Edit instead of FLUX?**
Because Qwen-Image-Edit is what customers are actually asking for, and it is **Apache-2.0**
— they can commercialize the fine-tuned result. FLUX.1-Kontext [dev] is non-commercial,
so it serves as our *technical* proof-of-feasibility, not the deployable target.

**Q: If Qwen is the goal, why did you port FLUX?**
To isolate the two gaps. FLUX (static shapes) proves that **diffusers integration alone
is enough** for a diffusion model to compile and train on Trainium — we got a `.neff`.
Qwen then shows the *additional* gap (dynamic-shape VLM encoder) cleanly, rather than
conflating "diffusers isn't supported" with "the encoder uses dynamic shapes."

**Q: Will Track 1 (diffusers support) alone make Qwen-Image-Edit train?**
No — and we won't claim it. Track 1 removes ~7 of the integration blockers, but Qwen's
Qwen2.5-VL encoder still emits dynamic shapes `neuronx-cc` can't compile. Qwen needs
Track 2 as well. Being precise here prevents a false promise to the customer.

**Q: Is Qwen-Image-Edit even feasible on Trainium, or is dynamic shape a dead end?**
Feasible, but with real work. `optimum-neuron` already ships **static** hand-written
encoders for Qwen2-VL *inference*, proving the encoder can be made static-shape. Track 2
brings that capability to the training path. It is an engineering investment, not a
hardware impossibility.

**Q: Why diffusion/media, not more LLM coverage?**
`optimum-neuron`'s training models are LLM-only today (llama, qwen3, granite); no
diffusion/DiT training model exists. Media/creative is a large underserved Trainium
segment whose core workload is diffusion fine-tuning on permissive (Apache/MIT) models
they can commercialize — a new vertical, not incremental LLM coverage.

**Q: What's the minimum viable version?**
Track 1 data-parallel + bf16 load + `prepare()`/extra fixes. That already lets customers
LoRA-fine-tune static-shape diffusion models (frozen base + adapters — the common case).
TP and Track 2 (VLM static encoder) are fast-follows prioritized by the Qwen demand.

**Q: How do we measure success?**
Time-to-first-training-step for a stock `diffusers` repo drops from "nine manual blockers
/ multi-day" to "one config change / minutes," and Qwen-Image-Edit (Apache-2.0)
fine-tunes on Trainium via a published runbook a customer follows unassisted.

**Q: What's the risk of not doing this?**
Customers evaluating Trainium for image-editing fine-tuning hit the documented wall,
conclude "Trainium can't do diffusion training," and standardize on GPU. The existing
inference investment never converts to the training workloads that drive sustained
compute consumption — and the requested Apache-2.0 model (Qwen-Image-Edit) goes to a
competitor's accelerator.

---

## APPENDIX — Evidence artifacts
- FLUX port + 7 integration-blocker fix history: `binchoo/flux-kontext-dev-on-trainium`
- Compilation proof: Neuron `.neff` generated for `FluxTransformer2DModel` on trn1.32xlarge
- Qwen-Image-Edit dynamic-shape analysis (the `set-dimension-size` block): prior technical review
- General porting process (model triage → guard → compile): `QUICKSTART.md` Appendix
