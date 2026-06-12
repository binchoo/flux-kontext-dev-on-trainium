#!/bin/bash
#
# Launch FLUX.1-Kontext-dev LoRA training on AWS Trainium (Neuron).
#
# Two phases:
#   1. neuron_parallel_compile — traces the graph and warms the compile cache
#      WITHOUT real weight updates (avoids a multi-minute stall on the first step).
#   2. torchrun — the actual training run, reusing the warm cache.
#
# Shapes are pinned (height/width/max_sequence_length/batch) so neuronx-cc
# compiles ONE static graph. Changing them triggers recompilation.

set -euo pipefail

# --- data columns / model / dataset (mirror train_model.sh) ---
export SOURCE_COLUMN="${SOURCE_COLUMN:-ghost_image}"
export TARGET_COLUMN="${TARGET_COLUMN:-target}"
export CAPTION_COLUMN="${CAPTION_COLUMN:-Prompt}"
export MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-Kontext-dev}"
export TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-raresense/SAKS_Jewelry}"
export VAL_DATASET_NAME="${VAL_DATASET_NAME:-raresense/SAKS_Jewelry_test}"
export OUTPUT_DIR="${OUTPUT_DIR:-SAKS_Lora_Training_Kontext_dev_neuron}"

# --- Neuron env ---
export FLUX_BACKEND=xla
export WANDB_MODE="${WANDB_MODE:-offline}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-/var/tmp/neuron-compile-cache}"
NUM_CORES="${NUM_CORES:-1}"

# Fixed shapes — keep constant for a single static graph.
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"

COMMON_ARGS=(
  --pretrained_model_name_or_path="$MODEL_NAME"
  --output_dir="$OUTPUT_DIR"
  --dataset_name="$TRAIN_DATASET_NAME"
  --validation_dataset_name="$VAL_DATASET_NAME"
  --source_column="$SOURCE_COLUMN"
  --target_column="$TARGET_COLUMN"
  --caption_column="$CAPTION_COLUMN"
  --mixed_precision="bf16"
  --train_batch_size=1
  --guidance_scale=1
  --rank=128
  --gradient_accumulation_steps=8
  --optimizer="adamw"
  --learning_rate=1e-5
  --lr_scheduler="constant"
  --lr_warmup_steps=0
  --max_train_steps=10050
  --num_train_epochs=5
  --seed=42
  --height="$HEIGHT"
  --width="$WIDTH"
  --max_sequence_length="$MAX_SEQ_LEN"
  --checkpointing_steps=2000
  --report_to="wandb"
)
# Note: --use_8bit_adam is intentionally omitted (CUDA-only; XLA uses AdamW).
# Validation (--validation_check/--validation_steps) is omitted by default since
# sampling uses different shapes and would recompile; enable once shapes bucketed.

echo "=== Phase 1/2: neuron_parallel_compile (graph warm-up, no weight updates) ==="
neuron_parallel_compile \
  torchrun --nproc_per_node="${NUM_CORES}" train.py "${COMMON_ARGS[@]}" || {
    echo "neuron_parallel_compile unavailable or failed; skipping warm-up."
    echo "(The first real step will compile instead — slower but correct.)"
  }

echo "=== Phase 2/2: training ==="
torchrun --nproc_per_node="${NUM_CORES}" train.py "${COMMON_ARGS[@]}"
