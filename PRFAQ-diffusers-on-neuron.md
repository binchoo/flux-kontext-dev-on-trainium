# PR/FAQ — Static-Shape VLM Encoder Support (and `diffusers` Training) in the Neuron SDK

> **Status:** Draft proposal (internal). Working-backwards document.
> **Author:** AWS SA (field-validated on trn1.32xlarge, Jun 2026)
> **The blocker only AWS can clear (P5):** `neuronx-cc` is a **static-shape compiler**.
> It rejects the XLA `set-dimension-size` op emitted by the dynamic, data-dependent
> shapes in **Qwen2.5-VL** — the vision-language text encoder inside **Qwen-Image-Edit
> (Apache-2.0)**, the model the customer actually wants. No dependency fix, config flag,
> or agent skill can clear this; it requires either a model rewrite (static-shape encoder)
> or **first-class static-shape VLM-encoder support in the Neuron SDK / `optimum-neuron`**.
> Tracked upstream as **aws-neuron-sdk #1144**.
> **One-line ask:** (a) add first-class `diffusers` *training* to `optimum-neuron`
> (removes the P1–P4 integration friction we can already work around), **and**
> (b) extend the static VLM-encoder path `optimum-neuron` already ships for *inference*
> to the encoders these image-editing models use — for **both inference and training** —
> so the customer's Apache-2.0 model can compile on Trainium.

---

## PRESS RELEASE

### AWS makes the Apache-2.0 image-editing model customers want — Qwen-Image-Edit — compile and train on Trainium

**SEATTLE — AWS today announced that the Neuron SDK now provides static-shape support
for vision-language text encoders (Qwen2.5-VL class) and first-class `diffusers`
training in `optimum-neuron`.** Media, advertising, and creative-tooling customers can
now fine-tune permissively-licensed image-editing models — including
**Qwen-Image-Edit (Apache-2.0)**, which they can serve commercially — on Trainium,
instead of hitting a compiler wall that previously left the model un-runnable for both
inference *and* training.

A field PoC exposed two classes of blocker that look the same to a customer ("it
errored") but are fundamentally different in who can fix them:

1. **Integration friction (P1–P4 — we already work around it).** Fine-tuning a
   `diffusers` model failed because the accelerator assumed text-only (transformers)
   models, dependency resolution silently broke the Neuron PyTorch pairing, and
   host-memory handling was left to the user. These are *ecosystem-maturity* gaps, not
   hardware limits — clearable by config, packaging, and an agent skill today. We
   proved it by porting **FLUX.1-Kontext** end-to-end: once cleared, **`neuronx-cc`
   compiled the model and produced a Neuron `.neff` artifact** on trn1.32xlarge.

2. **Architectural incompatibility (P5 — only AWS can clear it).** The model customers
   actually ask for, **Qwen-Image-Edit**, carries a **Qwen2.5-VL** vision-language text
   encoder built on *dynamic, data-dependent tensor shapes*. `neuronx-cc` is a
   **static-shape compiler**: it rejects the XLA `set-dimension-size` op these shapes
   produce (open issue **aws-neuron-sdk #1144**). This is not a dependency or config
   problem — no shim or skill resolves it. It blocks Qwen on Neuron for **both inference
   and training**, and `optimum-neuron` does not list Qwen-Image / Qwen2.5-VL for
   inference at all. Clearing it requires a static-shape encoder, in the model or — far
   better — in the SDK.

FLUX (static shapes, officially supported for inference) is the proof that the wall is
the *dynamic-shape architecture, not Trainium hardware*. With this release AWS removes
the P1–P4 friction in the framework **and** extends the static VLM-encoder capability it
already ships for inference to cover these encoders — turning "the customer tried, the
compiler rejected the model, and they went back to GPU" into "point `optimum-neuron` at
the model and train."

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
- **Inference** of *static-shape* diffusion models is first-class (`optimum-neuron` ships
  pipelines — e.g. FLUX).
- **Training** is not supported for `diffusers` at all (P1–P4 integration friction —
  which we can work around).
- **Qwen-Image-Edit specifically** is blocked twice over: the missing diffusers training
  path (P1–P4) *and* — the hard one — its **dynamic-shape Qwen2.5-VL encoder (P5)**, which
  `neuronx-cc` cannot compile. Because P5 is a compiler-level architectural
  incompatibility, it blocks Qwen for **inference too**: `optimum-neuron` lists only the
  Qwen2 *text* LLM for inference, not Qwen-Image / Qwen2.5-VL. There is no supported
  Qwen-Image-Edit path on Trainium today, in either direction.

The result is a **followership/DX gap with a hard floor**: the integration friction we
can clear ourselves; but the exact model the customer wants — Apache-2.0, runs on GPU day
one — hits a compiler wall on Trainium that **only AWS can move**.

---

## WHAT WE LEARNED (field evidence)

We ran two real ports on `trn1.32xlarge` to separate the two gaps.

### Case A — FLUX.1-Kontext: proves P1–P4 are *our* problem to clear, and the hardware is fine
FLUX uses fixed-length T5/CLIP encoders and a static-shape DiT. After clearing the
integration blockers by hand, **`neuronx-cc` compiled `FluxTransformer2DModel` successfully
(a `.neff` was generated on trn1.32xlarge).** This is the existence proof on two counts:
**(i)** the P1–P4 friction is purely ecosystem maturity — clearable by us via config,
packaging, and an agent skill; **(ii)** Trainium compiles and trains a static-shape
diffusion model just fine. The wall Qwen hits is therefore *not* the hardware.

P1–P4 blockers we cleared by hand (framework support removes these — **no AWS compiler
change required**):

| # | Blocker | Root cause | Who clears it |
|---|---|---|---|
| P1 | `NeuronAccelerator` can't `prepare()` a diffusers model | assumes transformers | Us (config / framework) |
| P2 | `AttributeError: FluxTransformer2DModel has no 'tie_weights'` | transformers-only call | Us (config / framework) |
| P3 | pinned host-memory / fp32 full-model load failures | no diffusers mem mgmt | Us (mostly) |
| P4a | `diffusers` not pulled by `[neuronx,training]` extra | packaging | Us (extra / pin) |
| P4b | app deps upgraded torch off the torch-neuronx pin (2.8→2.12) | no constraint guard | Us (constraint guard) |
| P4c | flash-attn / 8-bit Adam (CUDA-only) defaults | upstream | Us (auto-fallback) |
| P4d | `get_sigmas` `.nonzero().item()` host-sync | diffusers training util | Us (vectorized util) |

### Case B — Qwen-Image-Edit: the P5 wall only AWS can move
Qwen-Image-Edit is **Apache-2.0** and is the model customers request. Its **Qwen2.5-VL**
vision-language text encoder uses dynamic, data-dependent shapes — `masked_select` (in
`get_window_index`), `unique_consecutive`, and `cu_seqlens` variable-length packing.
These compile to the XLA `set-dimension-size` op, which **`neuronx-cc`, a static-shape
compiler, rejects** (open issue **aws-neuron-sdk #1144**). This is an architectural
property of native-resolution VLM encoders, not a bug we can patch.

**P5 is categorically different from P1–P4:**
- It is **not** clearable by a dependency fix, config flag, shim, or agent skill — those
  cleared P1–P4; none of them changes what the static-shape compiler will accept.
- It blocks Qwen for **both inference and training** — the same root cause is why
  `optimum-neuron` does not offer a Qwen-Image / Qwen2.5-VL inference path at all.
- It requires either a **model rewrite** (static-shape encoder) or, far better,
  **first-class static-shape VLM-encoder support in the Neuron SDK / `optimum-neuron`**.

**So Qwen-Image-Edit needs BOTH** the P1–P4 diffusers-training support (same as FLUX)
**and** the P5 static-encoder support — listing only one is a false promise to the
customer. The P5 ask is feasible: `optimum-neuron` already hand-writes **static** encoders
for some VLMs for *inference* (e.g. a Qwen2-VL inference path exists in NxDI), proving the
capability is possible; it needs to extend to the encoders these diffusion editing models
use, for both inference and training.

---

## WHAT WE PROPOSE

Two asks. Track 2 is the spine — it is the only one **we cannot do ourselves**.

**Track 1 — broad ask: `diffusers` training in `optimum-neuron` (removes P1–P4 friction
we currently hand-clear; unblocks static-shape models like FLUX):**
1. **`diffusers`-aware `NeuronAccelerator.prepare()`** — detect `diffusers.ModelMixin`;
   skip transformers-only calls (`tie_weights`); handle bf16 load + device placement.
2. **A `[diffusers]` install extra** pinned compatibly with the Neuron torch stack.
3. **`NeuronDiffusionTrainer`** mirroring the existing inference pipelines, covering the
   DiT family (FLUX, SD3, Qwen-Image DiT).
4. **Tensor-parallel training for DiT** so 12B+ models scale past the data-parallel
   host-RAM ceiling.

**Track 2 — the P5 ask (AWS-only): static-shape VLM-encoder support in the Neuron SDK /
`optimum-neuron`, for both inference and training:**
5. Provide a **static-shape encoding path for Qwen2.5-VL-class encoders** (fixed-resolution
   padding / dense attention in place of `masked_select` / `cu_seqlens` packing), so
   `neuronx-cc` no longer emits `set-dimension-size`. **Extend the static VLM-encoder work
   `optimum-neuron` already ships for inference** (e.g. the Qwen2-VL inference path in NxDI)
   to the encoders these image-editing diffusion models use — covering **inference *and*
   training**. Without this, the **Apache-2.0 model customers want never compiles on
   Trainium**, in either direction. This is the item only AWS can deliver.

**Plus:** a reference "Fine-tune Qwen-Image-Edit / FLUX on Trainium" runbook.

---

## FREQUENTLY ASKED QUESTIONS

**Q: Why lead with Qwen-Image-Edit instead of FLUX?**
Because Qwen-Image-Edit is what customers are actually asking for, and it is **Apache-2.0**
— they can commercialize the fine-tuned result. FLUX.1-Kontext [dev] is non-commercial,
so it serves as our *technical* proof-of-feasibility, not the deployable target.

**Q: If Qwen is the goal, why did you port FLUX?**
To isolate P1–P4 from P5. FLUX (static shapes) proves that for a *static-shape* diffusion
model **the diffusers integration is the only gap** — clear P1–P4 and it compiles and
trains on Trainium (we got a `.neff`). Qwen then exposes P5 (the dynamic-shape VLM encoder)
cleanly, rather than conflating "diffusers isn't supported" (which we can fix) with "the
compiler rejects the architecture" (which only AWS can fix).

**Q: Why can't an agent or a config change just fix P5 like it fixed the other blockers?**
Because P5 is not an integration problem — it is the **static-shape compiler rejecting the
model's architecture**. P1–P4 were the framework/packaging assuming a text model, the
wrong torch pin, CUDA-only defaults, a host-sync util: all clearable by config, packaging,
or an agent skill, because the model itself was already compilable. P5 is different: the
Qwen2.5-VL encoder emits dynamic, data-dependent shapes (`set-dimension-size`), and
`neuronx-cc` is a static-shape compiler that will not accept that op. No shim, dependency
pin, flag, or skill changes what the compiler accepts. The fix lives **inside the model
(rewrite to static shapes) or inside the Neuron SDK** — which is exactly why this is an
AWS ask, not something we can clear in the field.

**Q: Will Track 1 (diffusers support) alone make Qwen-Image-Edit train?**
No — and we won't claim it. Track 1 removes the P1–P4 integration blockers, but Qwen's
Qwen2.5-VL encoder still emits dynamic shapes `neuronx-cc` can't compile (P5). Qwen needs
**both** tracks. Listing only one would be a false promise to the customer.

**Q: Does "train on GPU, serve on Neuron" rescue Qwen-Image-Edit?**
No. The hybrid pattern works for **FLUX** — static shapes, and `optimum-neuron` officially
supports FLUX *inference* — so you can train on GPU and serve the result on Neuron. It does
**not** rescue Qwen, because Qwen's P5 root cause (the dynamic-shape Qwen2.5-VL encoder)
breaks **inference too**: `optimum-neuron` offers no Qwen-Image / Qwen2.5-VL inference path
at all. You cannot "just serve it on Neuron" because it won't compile in either direction.
The hybrid escape hatch only exists for models that are already static-shape — which is
the very thing Qwen is not, until Track 2 ships.

**Q: Is Qwen-Image-Edit even feasible on Trainium, or is dynamic shape a dead end?**
Feasible, with real work. `optimum-neuron` **already ships static, hand-written encoders
for some VLMs for *inference*** (e.g. a Qwen2-VL inference path in NxDI), which proves a
VLM encoder can be made static-shape and compile on Neuron. Track 2 extends that proven
capability to the encoders these diffusion editing models use, for both inference and
training. It is an engineering investment, not a hardware impossibility.

**Q: Why diffusion/media, not more LLM coverage?**
`optimum-neuron`'s training models are LLM-only today (llama, qwen3, granite); no
diffusion/DiT training model exists. Media/creative is a large underserved Trainium
segment whose core workload is diffusion fine-tuning on permissive (Apache/MIT) models
they can commercialize — a new vertical, not incremental LLM coverage.

**Q: What's the minimum viable version?**
Track 1 data-parallel + bf16 load + `prepare()`/extra fixes ships value immediately for
static-shape models (LoRA fine-tune FLUX-class DiTs — frozen base + adapters). But the MVP
that actually lands the requested model is **Track 2**: without static VLM-encoder support,
Qwen-Image-Edit does not compile at all. Track 2 should be scoped and prioritized as the
load-bearing item, not a fast-follow, since it is the AWS-only blocker the customer is
stuck behind.

**Q: How do we measure success?**
Two distinct bars. **Track 1:** time-to-first-training-step for a stock static-shape
`diffusers` repo drops from "P1–P4 hand-cleared / multi-day" to "one config change /
minutes." **Track 2 (the P5 bar):** Qwen-Image-Edit (Apache-2.0) **compiles** on Trainium
for the first time — for inference and training — and fine-tunes via a published runbook a
customer follows unassisted. Track 2 success is gated on the AWS SDK change; no runbook
delivers it until then.

**Q: What's the risk of not doing this?**
Customers evaluating Trainium for image-editing fine-tuning hit the documented wall,
conclude "Trainium can't do diffusion training," and standardize on GPU. The existing
inference investment never converts to the training workloads that drive sustained
compute consumption — and the requested Apache-2.0 model (Qwen-Image-Edit) goes to a
competitor's accelerator.

---

## APPENDIX — Evidence artifacts
- FLUX port + P1–P4 integration-blocker fix history: `binchoo/flux-kontext-dev-on-trainium`
- Compilation proof (hardware is fine): Neuron `.neff` generated for `FluxTransformer2DModel`
  on trn1.32xlarge
- P5 root cause — Qwen2.5-VL dynamic-shape analysis (`masked_select` in `get_window_index`,
  `unique_consecutive`, `cu_seqlens` packing → XLA `set-dimension-size` rejected by
  `neuronx-cc`): prior technical review; tracked upstream as **aws-neuron-sdk #1144**
- Feasibility precedent for the P5 ask: `optimum-neuron` / NxDI static VLM-encoder
  inference path (e.g. Qwen2-VL) — proves a static-shape encoder is achievable
- Coverage gap: `optimum-neuron` lists Qwen2 *text* LLM only — no Qwen-Image / Qwen2.5-VL
  inference support
- General porting process (model triage → guard → compile): `QUICKSTART.md` Appendix
