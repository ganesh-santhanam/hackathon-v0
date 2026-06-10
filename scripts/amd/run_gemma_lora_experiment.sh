#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
MAX_STEPS="${MAX_STEPS:-100}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EVAL_SUBSET_SIZE="${EVAL_SUBSET_SIZE:-64}"
TRAIN_FILE="${TRAIN_FILE:-data/lora/train.jsonl}"
EVAL_FILE="${EVAL_FILE:-data/lora/eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data/amd/lora}"
ADAPTER_DIR="${ADAPTER_DIR:-data/amd/lora/qwen4b_adapter}"
JUDGE_DATASET="${JUDGE_DATASET:-data/evals/eval_dataset.jsonl}"
BASE_RESULTS="${BASE_RESULTS:-data/evals/base_results.jsonl}"
LORA_RESULTS="${LORA_RESULTS:-data/evals/lora_results.jsonl}"
DRY_RUN="${DRY_RUN:-0}"

COMMON_ARGS=(
  --model-name "${MODEL_NAME}"
  --train-file "${TRAIN_FILE}"
  --eval-file "${EVAL_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --adapter-dir "${ADAPTER_DIR}"
  --max-steps "${MAX_STEPS}"
  --num-epochs "${NUM_EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --eval-subset-size "${EVAL_SUBSET_SIZE}"
  --bf16
)

if [[ "${DRY_RUN}" == "1" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

echo "Step 1: train Qwen LoRA adapter"
"${PYTHON_BIN}" scripts/amd/train_gemma_lora.py \
  --mode train \
  "${COMMON_ARGS[@]}"

echo "Step 2: build LLM-as-Judge evaluation dataset"
"${PYTHON_BIN}" scripts/run_llm_judge_eval.py build-dataset \
  --output "${JUDGE_DATASET}"

echo "Step 3: generate base and LoRA candidate outputs"
"${PYTHON_BIN}" scripts/amd/train_gemma_lora.py \
  --mode generate \
  "${COMMON_ARGS[@]}" \
  --judge-dataset "${JUDGE_DATASET}" \
  --base-results-output "${BASE_RESULTS}" \
  --lora-results-output "${LORA_RESULTS}"

echo "Done."
echo "Adapter: ${ADAPTER_DIR}"
echo "Base candidate results: ${BASE_RESULTS}"
echo "LoRA candidate results: ${LORA_RESULTS}"
echo "Next: run GPT-OSS judge with scripts/run_llm_judge_eval.py judge"
