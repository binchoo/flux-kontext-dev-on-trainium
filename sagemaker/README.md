# FLUX.1-Kontext (Neuron / inf2) — SageMaker Endpoint Deployment

End-to-end runbook to deploy an **already-compiled** FLUX.1-Kontext Neuron
artifact as a SageMaker real-time endpoint, using the **model.tar.gz + S3**
pattern with the **AWS Neuron PyTorch inference DLC** (not a custom container).

> ## NOTICE — NON-COMMERCIAL USE ONLY
> FLUX.1-Kontext [dev] is licensed by Black Forest Labs under a **Non-Commercial
> License**. These assets are for **evaluation / proof-of-concept only** and
> **must not** be used for commercial serving. You are responsible for license
> compliance. See the model card:
> <https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev>

---

## What this deploys

- **Model:** compiled `NeuronFluxKontextPipeline` artifact (output of
  `pipe.save_pretrained()`), pre-compiled with `export=True`,
  `tensor_parallel_size=8`, `1024x1024`, `bf16`.
- **Instance:** `ml.inf2.24xlarge` — 6 Inferentia2 chips = **12 NeuronCores**.
  The TP8 artifact needs **8 NeuronCores**, so it fits (with headroom). Smaller
  inf2 sizes do **not** have 8 cores and will not load this artifact.
- **Container:** AWS `pytorch-inference-neuronx` DLC (generic PyTorch-on-Neuron
  inference image — the right one for diffusers/optimum-neuron, **not** the LLM
  TGI image).
- **Latency:** cold load (load + Neuron device init) **~50s**; inference **~10s**
  for 1024x1024 / 28 steps / TP8.

### Fixed-shape constraint
Height/width/TP/batch are **baked into the compiled `.neff`** at export time.
This artifact is **1024x1024, TP8, batch=1**. The handler resizes every input
image to 1024x1024. To serve a different resolution you must **re-export** the
artifact (see `infer_neuron.py --export` in the repo root).

### Self-contained artifact
The base HF repo `black-forest-labs/FLUX.1-Kontext-dev` is **gated**, but the
compiled `save_pretrained()` artifact bundles everything needed to serve. The
handler forces `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, so **no HF
download and no HF token** are needed at serve time. If loading ever tries to
hit the Hub, the artifact is incomplete — re-export it rather than adding a token.

---

## Files

| File | Purpose |
|------|---------|
| `code/inference.py`     | SageMaker handler: `model_fn` / `input_fn` / `predict_fn` / `output_fn`. |
| `code/requirements.txt` | Extra serve-time pip deps (`diffusers==0.35.*`, `optimum-neuron[neuronx]`). |
| `build_model_tar.sh`    | Package compiled artifact + `code/` into `model.tar.gz`, upload to S3. |
| `deploy_endpoint.py`    | Create SageMaker Model + deploy to `ml.inf2.24xlarge`. |
| `invoke_endpoint.py`    | boto3 test client — sends an edit request, saves `output.png`. |

### model.tar.gz layout
```
model.tar.gz
├── model_index.json        ← compiled artifact files at the ARCHIVE ROOT
├── scheduler/
├── transformer/              (save_pretrained output, including compiled .neff)
├── text_encoder/ , text_encoder_2/
├── tokenizer/ ...
└── code/
    ├── inference.py        ← handler (SageMaker reads from code/)
    └── requirements.txt
```
The extracted archive root becomes `model_dir` passed to `model_fn`.

---

## Prerequisites

- The compiled artifact exists (default `/home/ec2-user/SageMaker/flux_kontext_neuron`).
- An S3 bucket you can write to.   *(VERIFY: set your bucket name.)*
- A SageMaker **execution IAM role** ARN with SageMaker + S3 read.  *(VERIFY: set your role ARN.)*
- `awscli`, `boto3`, `sagemaker` installed locally; AWS creds configured for the region.
- inf2 service quota for `ml.inf2.24xlarge` endpoint usage in the region.  *(VERIFY: may need a quota increase.)*

---

## Runbook

### 1. Build the tarball and upload to S3
```bash
cd sagemaker
./build_model_tar.sh \
  --compiled-dir /home/ec2-user/SageMaker/flux_kontext_neuron \
  --bucket <YOUR_BUCKET> \
  --prefix flux-kontext-neuron \
  --region us-west-2
# -> prints: s3://<YOUR_BUCKET>/flux-kontext-neuron/model.tar.gz
```

### 2. Look up the Neuron inference DLC image tag  (VERIFY — do not guess)
The URI pattern is:
```
763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:<TAG>
```
`763104351884` is the AWS DLC ECR account (all regions). List current tags:
```bash
aws ecr list-images \
  --registry-id 763104351884 \
  --repository-name pytorch-inference-neuronx \
  --region us-west-2 \
  --query 'imageIds[].imageTag' --output text | tr '\t' '\n' | sort
```
Example tags seen in the DLC catalog (confirm the current one — tags roll forward):
`2.9.0-neuronx-py312-sdk2.30.0-ubuntu24.04`,
`2.8.0-neuronx-py311-sdk2.26.1-ubuntu22.04`,
`2.7.0-neuronx-py310-sdk2.25.0-ubuntu22.04`.
Authoritative catalog:
<https://aws.github.io/deep-learning-containers/reference/available_images/>

> Note: `deploy_endpoint.py` will **refuse to guess** a tag — pass `--image-uri`.
> Pick a tag whose bundled Neuron SDK is compatible with the SDK used to compile
> the artifact (the artifact was built on AWS Neuron 2.30 in this project).

### 3. Deploy the endpoint
```bash
python deploy_endpoint.py \
  --model-data s3://<YOUR_BUCKET>/flux-kontext-neuron/model.tar.gz \
  --role arn:aws:iam::<ACCT>:role/<SageMakerExecutionRole> \
  --region us-west-2 \
  --image-uri 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:<TAG> \
  --endpoint-name flux-kontext-neuron
```
`container_startup_health_check_timeout` defaults to **900s** here to absorb the
~50s+ Neuron cold load and first warm-up. Deployment to `InService` typically
takes several minutes (instance provisioning + load).

### 4. Invoke (test)
```bash
python invoke_endpoint.py \
  --endpoint-name flux-kontext-neuron \
  --region us-west-2 \
  --image-url https://example.com/cat.png \
  --prompt "Add a red wizard hat to the cat" \
  --out output.png
# or: --image-file ./cat.png   (sent as base64)
```
Request JSON: `{prompt, image (base64 or URL), guidance_scale?, num_inference_steps?, seed?}`.
Response JSON: `{image_base64}` (PNG).

### 5. DELETE the endpoint to stop billing  (IMPORTANT)
```bash
aws sagemaker delete-endpoint        --endpoint-name flux-kontext-neuron --region us-west-2
aws sagemaker delete-endpoint-config --endpoint-config-name flux-kontext-neuron --region us-west-2
aws sagemaker delete-model           --model-name <model-name-from-console> --region us-west-2
```

> ### Cost caveat
> `ml.inf2.24xlarge` is a **large, expensive** instance billed **per hour for as
> long as the endpoint is InService** — whether or not it serves traffic. For a
> PoC, **deploy → test → delete promptly**. Leaving it running overnight is costly.

---

## Spots to VERIFY before running
- **DLC image tag** — list ECR / catalog and pick a compatible tag (step 2). Not hard-coded.
- **S3 bucket** name + **IAM execution role** ARN.
- **inf2.24xlarge service quota** in your region.
- **`code/requirements.txt`** — if your chosen DLC tag already bundles a
  compatible `optimum-neuron`/`diffusers`, those lines are redundant (harmless).
