#!/bin/bash
#
# FLUX.1-Kontext INFERENCE on AWS Neuron — compile then serve.
# This is the SUPPORTED path (optimum-neuron NeuronFluxKontextPipeline).
# Pattern: train on GPU, serve on Neuron.
#
# Usage:
#   ./run_infer_neuron.sh compile         # phase 1: HF weights -> .neff
#   ./run_infer_neuron.sh serve <image> "<prompt>"   # phase 2: edit an image

set -euo pipefail

MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-Kontext-dev}"
COMPILED_DIR="${COMPILED_DIR:-/home/ec2-user/SageMaker/flux_kontext_neuron}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
TP="${TP:-8}"            # tensor-parallel degree (docs: 8 on inf2.24xlarge / trn1)

export FLUX_BACKEND=xla
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/var/tmp/neuron-compile-cache}"

PHASE="${1:-compile}"

case "$PHASE" in
  compile)
    echo "=== Phase 1: compile $MODEL_ID -> Neuron (${HEIGHT}x${WIDTH}, tp=${TP}) ==="
    python infer_neuron.py --export \
      --model-id "$MODEL_ID" \
      --compiled-dir "$COMPILED_DIR" \
      --height "$HEIGHT" --width "$WIDTH" \
      --tensor-parallel-size "$TP"
    echo "=== compile done -> $COMPILED_DIR ==="
    ;;
  serve)
    IMAGE="${2:?usage: ./run_infer_neuron.sh serve <image> \"<prompt>\"}"
    PROMPT="${3:-Change the background to a green forest}"
    echo "=== Phase 2: serve (edit $IMAGE) ==="
    python infer_neuron.py \
      --compiled-dir "$COMPILED_DIR" \
      --height "$HEIGHT" --width "$WIDTH" \
      --image "$IMAGE" --prompt "$PROMPT" \
      --out output.png
    echo "=== done -> output.png ==="
    ;;
  *)
    echo "usage: $0 {compile|serve <image> \"<prompt>\"}"; exit 1
    ;;
esac
