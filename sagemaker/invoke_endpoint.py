#!/usr/bin/env python
# coding=utf-8
# =============================================================================
# NOTICE - NON-COMMERCIAL USE ONLY
# Invokes a FLUX.1-Kontext [dev] endpoint deployed for EVALUATION / PoC only.
# FLUX.1-Kontext [dev] is under a Non-Commercial License.
# =============================================================================
"""
Send a test image-edit request to the deployed FLUX.1-Kontext Neuron endpoint
and save the returned image to output.png.

The handler accepts an image as a URL or base64. This script defaults to a URL
(simplest), and also supports a local --image-file (sent as base64).

Usage:
  # using an image URL:
  python invoke_endpoint.py \
      --endpoint-name flux-kontext-neuron \
      --region us-west-2 \
      --image-url https://example.com/cat.png \
      --prompt "Add a red wizard hat to the cat"

  # using a local file:
  python invoke_endpoint.py \
      --endpoint-name flux-kontext-neuron \
      --image-file ./cat.png \
      --prompt "Change the background to a green forest"
"""

import argparse
import base64
import json

import boto3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-name", default="flux-kontext-neuron")
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--image-url", default=None, help="Public image URL to edit.")
    p.add_argument("--image-file", default=None, help="Local image path (sent as base64).")
    p.add_argument("--prompt", default="Change the background to a green forest")
    p.add_argument("--guidance-scale", type=float, default=2.5)
    p.add_argument("--num-inference-steps", type=int, default=28)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="output.png")
    args = p.parse_args()

    if not args.image_url and not args.image_file:
        raise SystemExit("Provide --image-url OR --image-file.")

    if args.image_file:
        with open(args.image_file, "rb") as f:
            image_payload = base64.b64encode(f.read()).decode("utf-8")
    else:
        image_payload = args.image_url

    body = {
        "prompt": args.prompt,
        "image": image_payload,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
    }

    runtime = boto3.client("sagemaker-runtime", region_name=args.region)
    print(f"[invoke] calling endpoint '{args.endpoint_name}' (first call may take ~10s+) ...")
    resp = runtime.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(body).encode("utf-8"),
    )

    result = json.loads(resp["Body"].read().decode("utf-8"))
    image_b64 = result["image_base64"]
    with open(args.out, "wb") as f:
        f.write(base64.b64decode(image_b64))
    print(f"[invoke] saved edited image to {args.out}")


if __name__ == "__main__":
    main()
