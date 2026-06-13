# Diffusers on AWS Neuron — UML Communication Diagram

A UML **communication (collaboration) diagram**: objects/participants connected by
links, with **numbered messages** encoding the sequence (1, 2, 2.1, 3, 3.1 …).
This emphasizes *who collaborates with whom*; the numbering carries the order.

- **Editable / SVG:** `mlops-communication-diagram.drawio` → `mlops-communication-diagram.svg`
  (open the .drawio in app.diagrams.net to edit).

> Mermaid has no native communication-diagram type, so the canonical artifact is the
> .drawio/.svg above. The numbered message list below is the textual equivalent.

## Participants (objects)

| Object | Role |
|---|---|
| `:MLEngineer` | drives the workflow |
| `:HuggingFaceHub` | model source |
| `:GPUTrainer (CUDA diffusers)` | fine-tunes on GPU |
| `:NeuronAgent (triage + port)` | neuron-agentic-development: compatibility triage & porting |
| `:Trainium (neuronx-cc)` | compiles to .neff |
| `:ArtifactRegistry (S3 / ECR)` | stores weights + .neff |
| `:SageMakerEndpoint (inf2 / trn)` | serves inference |

## Numbered messages (the sequence)

| # | From → To | Message |
|---|---|---|
| 1 | MLEngineer → HuggingFaceHub | `selectModel()` |
| 2 | MLEngineer → GPUTrainer | `fineTune(LoRA)` |
| 2.1 | GPUTrainer → ArtifactRegistry | `putWeights()` |
| 3 | MLEngineer → NeuronAgent | `triage(model)` |
| 3.1 | NeuronAgent → self | `checkDynamicShape() / optimumSupported?` |
| 3.2 | NeuronAgent → self | `[unsupported] portToNeuronOps()` (static-shape / NKI) · `[supported]` reuse HF/AWS port (FLUX/SD/SDXL) |
| 4 | NeuronAgent → Trainium | `compile(export=True, tp=8)` |
| 4.1 | Trainium → self | `HLO → .neff` |
| 4.2 | Trainium → self | `validate(golden)` |
| 5 | Trainium → ArtifactRegistry | `putArtifact(.neff)` |
| 6 | MLEngineer → SageMakerEndpoint | `deploy(model_data=.neff)` |
| 6.1 | SageMakerEndpoint → ArtifactRegistry | `load(.neff)` — no runtime compile |
| 7 | MLEngineer → SageMakerEndpoint | `infer(image, prompt) → editedImage` |

**Branch (message 3.2):** supported architectures (FLUX/SD/SDXL) are already ported by
HF/AWS → no custom porting; unsupported ones (Qwen-Image-Edit / dynamic-shape VLM
encoders) require porting to Neuron ops before compile.
