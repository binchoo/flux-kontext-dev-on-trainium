#!/usr/bin/env bash
# =============================================================================
# Cleanup — remove today's deploy attempts (endpoint, config, model, ECR images,
# stale local docker images) so we start the DLC-fixed path from a clean slate.
# Safe to re-run; each step ignores "not found".
# =============================================================================
set -uo pipefail

REGION="${REGION:-us-west-2}"
ACCT="${ACCT:-487498333664}"
ENDPOINT="${ENDPOINT:-flux-kontext-neuron}"
ECR_REPO="${ECR_REPO:-flux-kontext-neuron-fix}"

echo "=== [1/5] delete endpoint: $ENDPOINT ==="
aws sagemaker delete-endpoint --endpoint-name "$ENDPOINT" --region "$REGION" 2>/dev/null \
  && echo "  endpoint deleted" || echo "  (no endpoint)"

echo "=== [2/5] delete endpoint-config: $ENDPOINT ==="
aws sagemaker delete-endpoint-config --endpoint-config-name "$ENDPOINT" --region "$REGION" 2>/dev/null \
  && echo "  config deleted" || echo "  (no config)"

echo "=== [3/5] delete SageMaker models tagged to this endpoint name ==="
# SageMaker SDK names models like "<endpoint>-<timestamp>"; list & delete matches.
for m in $(aws sagemaker list-models --region "$REGION" \
            --name-contains "$ENDPOINT" \
            --query 'Models[].ModelName' --output text 2>/dev/null); do
  aws sagemaker delete-model --model-name "$m" --region "$REGION" 2>/dev/null \
    && echo "  model deleted: $m"
done
echo "  done"

echo "=== [4/5] delete stale ECR images in $ECR_REPO (keep repo) ==="
# Remove the custom-container attempts (ubuntu310, py310sdk2.21, sdk2.30...) so
# we don't accidentally redeploy a broken tag. Comment out to keep history.
IMG_IDS=$(aws ecr list-images --repository-name "$ECR_REPO" --region "$REGION" \
            --query 'imageIds[*]' --output json 2>/dev/null)
if [ -n "$IMG_IDS" ] && [ "$IMG_IDS" != "[]" ] && [ "$IMG_IDS" != "null" ]; then
  aws ecr batch-delete-image --repository-name "$ECR_REPO" --region "$REGION" \
    --image-ids "$IMG_IDS" >/dev/null 2>&1 && echo "  ECR images purged" || echo "  (ECR purge skipped)"
else
  echo "  (no ECR images / repo absent)"
fi

echo "=== [5/5] prune local docker (failed build layers, dangling) ==="
docker rmi flux-kontext-neuron-fix:ubuntu310 2>/dev/null || true
docker rmi flux-kontext-neuron-fix:py310sdk2.21 2>/dev/null || true
docker rmi flux-kontext-neuron-fix:sdk2.30 2>/dev/null || true
docker image prune -f >/dev/null 2>&1 || true
echo "  local docker pruned"

echo ""
echo "=== CLEANUP DONE — clean slate for the DLC-fixed rebuild path ==="
