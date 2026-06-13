#!/usr/bin/env bash
# =============================================================================
# ONE-SHOT: DLC-fixed build -> rebuild .neff (sdk2.25) -> package -> push -> deploy
# Working-backward: the inference DLC is the fixed runtime; the .neff is REBUILT
# inside it so compile==serve. Run on the SageMaker/inf2 instance (needs NeuronCore
# for the rebuild + Docker + aws creds). NON-COMMERCIAL eval only (FLUX dev license).
#
# Usage:
#   ./oneshot_dlc.sh gate       # step 0: verify DLC can import NeuronFluxKontextPipeline
#   ./oneshot_dlc.sh rebuild    # step 1: build image + rebuild .neff inside it
#   ./oneshot_dlc.sh deploy     # step 2: package -> S3 -> ECR push -> endpoint
#   ./oneshot_dlc.sh all        # rebuild + deploy (after gate passes)
# =============================================================================
set -euo pipefail

REGION="${REGION:-us-west-2}"
ACCT="${ACCT:-487498333664}"
DLC="${DLC:-763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.7.0-neuronx-py310-sdk2.25.0-ubuntu22.04}"
IMAGE="${IMAGE:-flux-kontext-neuron-dlc}"
TAG="${TAG:-sdk2.25}"
MYIMG="${ACCT}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE}:${TAG}"

BUCKET="${BUCKET:-sagemaker-us-west-2-487498333664}"
PREFIX="${PREFIX:-flux-kontext-neuron}"
ROLE="${ROLE:-arn:aws:iam::487498333664:role/service-role/AmazonSageMaker-ExecutionRole-20260406T131382}"
ENDPOINT="${ENDPOINT:-flux-kontext-neuron}"

MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-Kontext-dev}"
HEIGHT="${HEIGHT:-1024}"; WIDTH="${WIDTH:-1024}"; TP="${TP:-8}"
HOST_NEFF="${HOST_NEFF:-/home/ec2-user/SageMaker/flux_kontext_neuron_sdk225}"  # NEW dir (don't clobber old .neff)
HF_HOME_HOST="${HF_HOME_HOST:-/home/ec2-user/SageMaker/hf_cache}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

login_dlc() { aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin 763104351884.dkr.ecr."$REGION".amazonaws.com; }
login_mine(){ aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCT".dkr.ecr."$REGION".amazonaws.com; }

gate() {
  echo "=== GATE: does the DLC import NeuronFluxKontextPipeline (+optimum 0.4.5/diffusers)? ==="
  login_dlc
  docker run --rm --entrypoint bash "$DLC" -lc '
    pip install -q --no-deps "optimum-neuron==0.4.5" "optimum==2.0.0" "diffusers==0.35.2" "accelerate==1.8.1" 2>&1 | tail -3
    pip install -q "huggingface-hub" "safetensors" "httpx" "pydantic" 2>&1 | tail -2
    python -c "import transformers,diffusers; print(\"tf\",transformers.__version__,\"df\",diffusers.__version__)"
    python -c "from optimum.neuron import NeuronFluxKontextPipeline; print(\"KONTEXT OK\")"
  '
  echo "=== GATE done. If 'KONTEXT OK' printed, proceed to: $0 rebuild ==="
}

build_image() {
  echo "=== build serving image from DLC: $IMAGE:$TAG ==="
  login_dlc
  docker build --build-arg BASE="$DLC" -t "${IMAGE}:${TAG}" -f "$HERE/Dockerfile.dlc" "$HERE"
}

rebuild_neff() {
  echo "=== rebuild .neff INSIDE the image (neuronx-cc 2.25 == serve runtime) -> $HOST_NEFF ==="
  mkdir -p "$HOST_NEFF" "$HF_HOME_HOST"
  # Run compile in the SAME image we serve with, mounting NeuronCores + repo + caches.
  # infer_neuron.py --export lives at the repo root (one level up from sagemaker/).
  REPO_ROOT="$(cd "$HERE/.." && pwd)"
  docker run --rm \
    --device /dev/neuron0 --device /dev/neuron1 --device /dev/neuron2 --device /dev/neuron3 \
    --device /dev/neuron4 --device /dev/neuron5 --device /dev/neuron6 --device /dev/neuron7 \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HOME=/hf -e HF_HUB_DISABLE_XET=1 \
    -e NEURON_RT_VISIBLE_CORES=0-7 \
    -v "$REPO_ROOT":/work -v "$HOST_NEFF":/neff -v "$HF_HOME_HOST":/hf \
    --entrypoint bash "${IMAGE}:${TAG}" -lc "
      cd /work &&
      python infer_neuron.py --export \
        --model-id '$MODEL_ID' --compiled-dir /neff \
        --height $HEIGHT --width $WIDTH --tensor-parallel-size $TP
    "
  echo "=== .neff rebuilt at $HOST_NEFF (sdk2.25) ==="
}

package_push_deploy() {
  echo "=== package model.tar.gz (.neff @ $HOST_NEFF + code/) -> S3 ==="
  bash "$HERE/build_model_tar.sh" --compiled-dir "$HOST_NEFF" --bucket "$BUCKET" --prefix "$PREFIX" --region "$REGION"

  echo "=== push serving image to ECR ==="
  aws ecr create-repository --repository-name "$IMAGE" --region "$REGION" >/dev/null 2>&1 || true
  login_mine
  docker tag "${IMAGE}:${TAG}" "$MYIMG"
  docker push "$MYIMG"

  echo "=== deploy endpoint ==="
  python "$HERE/deploy_endpoint.py" \
    --model-data "s3://${BUCKET}/${PREFIX}/model.tar.gz" \
    --role "$ROLE" --region "$REGION" \
    --image-uri "$MYIMG" \
    --endpoint-name "$ENDPOINT" \
    --health-check-timeout 1800
  echo "=== deployed. tail logs: aws logs tail /aws/sagemaker/Endpoints/$ENDPOINT --region $REGION --follow ==="
}

case "${1:-all}" in
  gate)    gate ;;
  rebuild) build_image; rebuild_neff ;;
  deploy)  package_push_deploy ;;
  all)     build_image; rebuild_neff; package_push_deploy ;;
  *) echo "usage: $0 {gate|rebuild|deploy|all}"; exit 1 ;;
esac
