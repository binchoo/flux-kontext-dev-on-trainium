#!/usr/bin/env bash
# =============================================================================
# NOTICE - NON-COMMERCIAL USE ONLY
# This packages FLUX.1-Kontext [dev] for EVALUATION / PoC deployment only.
# FLUX.1-Kontext [dev] is under a Non-Commercial License. Do not serve commercially.
# =============================================================================
#
# Build model.tar.gz for SageMaker (model.tar.gz + S3 pattern) and upload to S3.
#
# Final tar layout (SageMaker extracts this to /opt/ml/model at container start;
# that extracted root is passed to model_fn as `model_dir`):
#
#   model.tar.gz
#   |-- model_index.json            <- compiled artifact files at the ARCHIVE ROOT
#   |-- scheduler/
#   |-- transformer/                  (the save_pretrained() output, incl. .neff)
#   |-- text_encoder/
#   |-- text_encoder_2/
#   |-- tokenizer/ ...
#   `-- code/
#       |-- inference.py            <- handler (SageMaker looks in code/)
#       `-- requirements.txt        <- extra pip deps installed at startup
#
# The compiled-artifact files MUST be at the archive root (NOT nested in a
# subfolder), so that model_dir == the from_pretrained() directory. The handler
# also tolerates a single nested subfolder as a fallback, but root is canonical.
#
# Usage:
#   ./build_model_tar.sh \
#       --compiled-dir /home/ec2-user/SageMaker/flux_kontext_neuron \
#       --bucket my-sagemaker-bucket \
#       --prefix flux-kontext-neuron \
#       --region us-west-2
#
set -euo pipefail

# -------- defaults (override via flags) --------
COMPILED_DIR="/home/ec2-user/SageMaker/flux_kontext_neuron"
BUCKET=""                       # REQUIRED: your S3 bucket (no s3:// prefix)
PREFIX="flux-kontext-neuron"
REGION="us-west-2"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this sagemaker/ dir
CODE_DIR="${WORKDIR}/code"
OUTPUT_TAR="${WORKDIR}/model.tar.gz"

usage() {
  echo "Usage: $0 --bucket <s3-bucket> [--compiled-dir DIR] [--prefix P] [--region R]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compiled-dir) COMPILED_DIR="$2"; shift 2 ;;
    --bucket)       BUCKET="$2";       shift 2 ;;
    --prefix)       PREFIX="$2";       shift 2 ;;
    --region)       REGION="$2";       shift 2 ;;
    -h|--help)      usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

if [[ -z "${BUCKET}" ]]; then
  echo "ERROR: --bucket is required (the target S3 bucket)."
  usage
fi
if [[ ! -d "${COMPILED_DIR}" ]]; then
  echo "ERROR: compiled artifact dir not found: ${COMPILED_DIR}"
  exit 1
fi
if [[ ! -f "${CODE_DIR}/inference.py" ]]; then
  echo "ERROR: ${CODE_DIR}/inference.py not found (expected next to this script)."
  exit 1
fi

echo "[1/4] Staging archive contents ..."
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

# Copy the compiled artifact files to the archive ROOT.
# Trailing /. copies the *contents* of COMPILED_DIR, not the dir itself.
cp -a "${COMPILED_DIR}/." "${STAGE}/"

# Copy the handler code under code/.
mkdir -p "${STAGE}/code"
cp -a "${CODE_DIR}/inference.py" "${STAGE}/code/"
cp -a "${CODE_DIR}/requirements.txt" "${STAGE}/code/"

echo "[2/4] Verifying compiled artifact root has model_index.json ..."
if [[ ! -f "${STAGE}/model_index.json" ]]; then
  echo "WARN: ${STAGE}/model_index.json not found at archive root."
  echo "      Confirm ${COMPILED_DIR} is a save_pretrained() output."
fi

echo "[3/4] Creating ${OUTPUT_TAR} ..."
# Build the tarball from inside STAGE so paths are relative (no leading dir).
tar -C "${STAGE}" -czf "${OUTPUT_TAR}" .
echo "      Tar contents (top level):"
tar -tzf "${OUTPUT_TAR}" | sed 's#^\./##' | awk -F/ '{print $1}' | sort -u | sed 's/^/        /'

echo "[4/4] Uploading to s3://${BUCKET}/${PREFIX}/model.tar.gz ..."
aws s3 cp "${OUTPUT_TAR}" "s3://${BUCKET}/${PREFIX}/model.tar.gz" --region "${REGION}"

echo ""
echo "DONE. model_data S3 URI:"
echo "  s3://${BUCKET}/${PREFIX}/model.tar.gz"
echo ""
echo "Pass this URI to deploy_endpoint.py via --model-data."
