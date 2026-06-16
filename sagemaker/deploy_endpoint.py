#!/usr/bin/env python
# coding=utf-8
# =============================================================================
# NOTICE - NON-COMMERCIAL USE ONLY
# Deploys FLUX.1-Kontext [dev] for EVALUATION / PoC only. FLUX.1-Kontext [dev]
# is under a Non-Commercial License. Do not use for commercial serving.
# =============================================================================
"""
Deploy the compiled FLUX.1-Kontext Neuron model to a SageMaker real-time
endpoint on ml.inf2.24xlarge, using the model.tar.gz + S3 pattern with the
AWS Neuron PyTorch inference DLC (NOT a custom container).

Prereqs:
  - model.tar.gz already uploaded to S3 (see build_model_tar.sh).
  - An IAM execution role with SageMaker + S3 read access.
  - sagemaker + boto3 installed locally.

Usage:
  python deploy_endpoint.py \
      --model-data s3://my-bucket/flux-kontext-neuron/model.tar.gz \
      --role arn:aws:iam::<acct>:role/<SageMakerExecutionRole> \
      --region us-west-2 \
      --image-uri 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:<TAG>
"""

import argparse

import boto3
import sagemaker
from sagemaker.model import Model


# inf2.24xlarge = 6 Inferentia2 chips x 2 NeuronCores = 12 NeuronCores total.
# The compiled artifact is TP8, so it needs 8 NeuronCores (fits, with headroom).
INSTANCE_TYPE = "ml.inf2.24xlarge"


def get_image_uri(region, explicit_uri=None):
    """Resolve the Neuron inference DLC image URI.

    PREFERRED: pass --image-uri explicitly with a tag you've verified exists.

    --------------------------------------------------------------------------
    DLC URI PATTERN (verify the exact tag before use!):

      763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-inference-neuronx:<TAG>

    763104351884 is the AWS Deep Learning Containers ECR account (all regions).
    The repo is `pytorch-inference-neuronx` -- the GENERIC PyTorch-on-NeuronX
    inference DLC. This is the right image for a diffusers/optimum-neuron
    pipeline. Do NOT use the LLM image (get_huggingface_llm_image_uri /
    text-generation-inference-neuronx) -- that serves TGI/LLMs, not diffusers.

    Example tags seen in the DLC catalog (CONFIRM the current one for your
    region with the CLI command below before deploying -- tags roll forward):
        2.9.0-neuronx-py312-sdk2.30.0-ubuntu24.04
        2.8.0-neuronx-py311-sdk2.26.1-ubuntu22.04
        2.7.0-neuronx-py310-sdk2.25.0-ubuntu22.04

    VERIFY available tags (recommended) -- list the ECR repo directly:
        aws ecr list-images \\
          --registry-id 763104351884 \\
          --repository-name pytorch-inference-neuronx \\
          --region us-west-2 \\
          --query 'imageIds[].imageTag' --output text | tr '\\t' '\\n' | sort

    Authoritative catalog (always check this for the latest tags):
        https://github.com/aws/deep-learning-containers/blob/master/available_images.md
        (rendered: https://aws.github.io/deep-learning-containers/reference/available_images/)
    --------------------------------------------------------------------------

    Optionally, the SageMaker SDK's image_uris.retrieve CAN build this URI, but
    coverage/aliases for the neuronx inference image vary by SDK version, so the
    explicit --image-uri path above is the safest. Commented example:

        # from sagemaker import image_uris
        # uri = image_uris.retrieve(
        #     framework="pytorch",                 # or "huggingface"
        #     region=region,
        #     version="2.9.0",                     # match a real DLC version
        #     instance_type="ml.inf2.24xlarge",    # drives the neuronx variant
        #     image_scope="inference",
        # )
        # return uri

    If neither an explicit URI nor a confirmed retrieve() call is available,
    we refuse to guess (a wrong-but-plausible URI fails opaquely at deploy).
    """
    if explicit_uri:
        return explicit_uri
    raise SystemExit(
        "No --image-uri provided. Look up the current pytorch-inference-neuronx "
        "tag (see the `aws ecr list-images` command / available_images.md in this "
        "function's docstring) and pass it via --image-uri. Refusing to guess a tag."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-data", required=True, help="s3://.../model.tar.gz")
    p.add_argument("--role", required=True, help="SageMaker execution role ARN")
    p.add_argument("--region", default="us-west-2")
    p.add_argument(
        "--image-uri",
        default=None,
        help="Neuron inference DLC image URI (verify the tag; see get_image_uri docstring).",
    )
    p.add_argument("--endpoint-name", default="flux-kontext-neuron")
    p.add_argument("--instance-type", default=INSTANCE_TYPE)
    p.add_argument(
        "--health-check-timeout",
        type=int,
        default=900,
        help="container_startup_health_check_timeout seconds. >=600 for the ~50s+ "
        "Neuron load and first warm-up. Default 900 for headroom.",
    )
    p.add_argument(
        "--inference-ami-version",
        default="al2-ami-sagemaker-inference-neuron-2",
        help="SageMaker ProductionVariant InferenceAmiVersion. "
        "Default 'al2-ami-sagemaker-inference-neuron-2' ships Neuron Driver 2.19 "
        "(SageMaker's built-in default 2.10.11 is too old for the DLC image). "
        "Pass an empty string to omit the field and use the SageMaker default.",
    )
    args = p.parse_args()

    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    image_uri = get_image_uri(args.region, args.image_uri)
    print(f"[deploy] image_uri = {image_uri}")
    print(f"[deploy] model_data = {args.model_data}")

    model = Model(
        image_uri=image_uri,
        model_data=args.model_data,
        role=args.role,
        sagemaker_session=sm_session,
        # Tell the inference toolkit where the handler lives inside the tarball.
        # (Files are under code/inference.py at the archive root.)
        env={
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            # The pipeline self-manages the 8 NeuronCores it needs for TP8 on the
            # 12-core inf2.24xlarge. Pin visibility/count explicitly so the worker
            # claims exactly 8 cores (avoids over-subscription if multiple workers).
            # Pin the EXACT cores (0-7), not just the count. On the endpoint the
            # host exposes ALL cores (NEURON_CORE_HOST_TOTAL=64 on trn1), and
            # NEURON_RT_NUM_CORES=8 ("use 8") let the runtime fail at
            # init_sp_resource (event semaphore vaddr) -> NRT_RESOURCE / Allocation
            # Failure. Locally `--device /dev/neuron0..7` worked because it
            # physically exposed exactly those. VISIBLE_CORES reproduces that:
            # claim cores 0-7 explicitly so the TP8 graph maps cleanly.
            "NEURON_RT_VISIBLE_CORES": "0-7",
            # CRITICAL: force EXACTLY ONE torchserve worker. By default torchserve
            # scales workers to CPU count (trn1 = 128 vCPU -> dozens of workers).
            # Each worker runs model_fn and tries to claim the 8 TP NeuronCores,
            # so they fight over the cores and the lazy `from optimum.neuron import
            # NeuronFluxKontextPipeline` collides across processes -> "cannot import
            # name" + "Backend worker process died". One worker = one model owning
            # the 8 cores. (SAGEMAKER_MODEL_SERVER_WORKERS is the toolkit knob;
            # the min/max default workers are the torchserve-level guards.)
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            "TS_DEFAULT_WORKERS_PER_MODEL": "1",
            "TS_MIN_WORKERS": "1",
            "TS_MAX_WORKERS": "1",
            # Keep model loading offline (gated base repo; artifact is self-contained).
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )

    print(f"[deploy] creating endpoint '{args.endpoint_name}' on {args.instance_type} ...")

    if args.inference_ami_version:
        # model.deploy() has no path to set InferenceAmiVersion on the
        # ProductionVariant — drop to boto3 for the three-step create flow.
        print(f"[deploy] InferenceAmiVersion = {args.inference_ami_version}")
        sm_client = boto_session.client("sagemaker")
        model_name = args.endpoint_name + "-model"
        endpoint_config_name = args.endpoint_name + "-config"

        # 1. Register the model object so the endpoint config can reference it.
        model.name = model_name
        model._create_sagemaker_model(
            instance_type=args.instance_type,
            accelerator_type=None,
            tags=None,
        )

        # 2. Endpoint config with the InferenceAmiVersion ProductionVariant field.
        production_variant = {
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialInstanceCount": 1,
            "InstanceType": args.instance_type,
            "InferenceAmiVersion": args.inference_ami_version,
            "ContainerStartupHealthCheckTimeoutInSeconds": args.health_check_timeout,
            "ModelDataDownloadTimeoutInSeconds": 1200,
        }
        sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[production_variant],
        )

        # 3. Create and wait for endpoint.
        sm_client.create_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        waiter = sm_client.get_waiter("endpoint_in_service")
        print("[deploy] waiting for endpoint to reach InService (this takes a few minutes) ...")
        waiter.wait(
            EndpointName=args.endpoint_name,
            WaiterConfig={"Delay": 30, "MaxAttempts": 60},
        )
    else:
        model.deploy(
            initial_instance_count=1,
            instance_type=args.instance_type,
            endpoint_name=args.endpoint_name,
            # Cold load + Neuron device init is ~50s; give the health check room.
            container_startup_health_check_timeout=args.health_check_timeout,
            # Neuron load can exceed the default model-download grace too.
            model_data_download_timeout=1200,
        )

    print(f"[deploy] endpoint '{args.endpoint_name}' is InService.")
    print("[deploy] REMEMBER: ml.inf2.24xlarge bills per-hour while running.")
    print("[deploy] Delete it when done:  aws sagemaker delete-endpoint "
          f"--endpoint-name {args.endpoint_name} --region {args.region}")


if __name__ == "__main__":
    main()
