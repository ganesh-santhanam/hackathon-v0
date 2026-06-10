#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss:20b}"
MAX_STEPS="${MAX_STEPS:-100}"
LIMIT="${LIMIT:-}"
BF16="${BF16:-1}"
DRY_RUN="${DRY_RUN:-0}"

SOURCE_FAILURE_ROWS="${SOURCE_FAILURE_ROWS:-100}"
TRAIN_FILE="${TRAIN_FILE:-data/lora/train.jsonl}"
EVAL_FILE="${EVAL_FILE:-data/lora/eval.jsonl}"
AMD_OUTPUT_DIR="${AMD_OUTPUT_DIR:-data/amd/lora}"
ADAPTER_DIR="${ADAPTER_DIR:-data/amd/lora/qwen4b_adapter}"
EVAL_DATASET="${EVAL_DATASET:-data/evals/eval_dataset.jsonl}"
BASE_RESULTS="${BASE_RESULTS:-data/evals/base_results.jsonl}"
LORA_RESULTS="${LORA_RESULTS:-data/evals/lora_results.jsonl}"
JUDGE_SCORES="${JUDGE_SCORES:-data/evals/judge_scores.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-data/evals/summary.json}"
SUMMARY_CSV="${SUMMARY_CSV:-data/evals/summary.csv}"
REPORT_MD="${REPORT_MD:-data/evals/llm_judge_report.md}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai-compatible}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-http://localhost:8000/v1/chat/completions}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EVAL_SUBSET_SIZE="${EVAL_SUBSET_SIZE:-64}"

export PYTHONPATH="${PYTHONPATH:-src}"

section() {
  printf '\n========== %s ==========\n' "$1"
}

fail() {
  printf '\nERROR: %s\n' "$1" >&2
  exit 1
}

run() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
  "$@"
}

require_file() {
  local path="$1"
  local hint="$2"
  [[ -f "${path}" ]] || fail "Missing ${path}. ${hint}"
}

optional_limit_args=()
if [[ -n "${LIMIT}" ]]; then
  optional_limit_args=(--limit "${LIMIT}")
fi

bf16_args=()
if [[ "${BF16}" == "1" || "${BF16}" == "true" || "${BF16}" == "TRUE" ]]; then
  bf16_args=(--bf16)
fi

dry_run_args=()
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "TRUE" ]]; then
  dry_run_args=(--dry-run)
fi

section "1. Environment Check"
[[ -f "README.md" && -f "scripts/amd/run_full_amd_experiment.sh" ]] || fail "Run this script from the repository root."
[[ -d "src/industrial_ai" ]] || fail "Cannot find src/industrial_ai. Run from the repository root."
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python not found: ${PYTHON_BIN}. Set PYTHON_BIN=/path/to/python."
run "${PYTHON_BIN}" --version

"${PYTHON_BIN}" - <<'PY' || fail "Python cannot import the project. Set PYTHONPATH=src and run from repo root."
import industrial_ai
print("project_import_ok")
PY

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "TRUE" ]]; then
  echo "DRY_RUN enabled: GPU/model dependency checks will be non-blocking."
else
  "${PYTHON_BIN}" - <<'PY' || fail "Missing AMD training dependencies. Run: bash scripts/amd/setup_amd_lora.sh"
import importlib.util
missing = [name for name in ["torch", "transformers", "peft", "datasets"] if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing: " + ", ".join(missing))
PY
  "${PYTHON_BIN}" - <<'PY' || fail "No ROCm/GPU device visible to PyTorch. Verify AMD Cloud GPU setup or use DRY_RUN=1."
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
print("gpu_device:", torch.cuda.get_device_name(0))
print("bf16_supported:", torch.cuda.is_bf16_supported())
PY
fi

section "2. Dataset And Corpus Generation"
mkdir -p data/incidents data/lora data/evals "${AMD_OUTPUT_DIR}"
if [[ -f "data/incidents/ai4i_incident_corpus.jsonl" ]]; then
  echo "Using existing data/incidents/ai4i_incident_corpus.jsonl"
else
  require_file "ai4i_dataset/ai4i2020.csv" \
    "Add the AI4I CSV or generate the incident corpus before running this experiment."
  run "${PYTHON_BIN}" -m industrial_ai.incidents.generate \
    --source-failure-rows "${SOURCE_FAILURE_ROWS}"
fi
require_file "data/incidents/ai4i_incident_corpus.jsonl" "Corpus generation did not produce the expected file."

section "3. LoRA Dataset Preparation"
run "${PYTHON_BIN}" scripts/prepare_lora_dataset.py \
  --input data/incidents/ai4i_incident_corpus.jsonl \
  --output-dir data/lora \
  "${optional_limit_args[@]}"
require_file "${TRAIN_FILE}" "LoRA train split was not created."
require_file "${EVAL_FILE}" "LoRA eval split was not created."

section "4. Qwen Base/LoRA Training"
run "${PYTHON_BIN}" scripts/amd/train_gemma_lora.py \
  --mode train \
  --model-name "${BASE_MODEL}" \
  --train-file "${TRAIN_FILE}" \
  --eval-file "${EVAL_FILE}" \
  --output-dir "${AMD_OUTPUT_DIR}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --max-steps "${MAX_STEPS}" \
  --num-epochs "${NUM_EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --eval-subset-size "${EVAL_SUBSET_SIZE}" \
  "${bf16_args[@]}" \
  "${dry_run_args[@]}"
require_file "${AMD_OUTPUT_DIR}/training_metrics.json" "Training metrics were not written."
require_file "${AMD_OUTPUT_DIR}/training_log.txt" "Training log was not written."

section "5. Base + LoRA Candidate Generation"
run "${PYTHON_BIN}" scripts/run_llm_judge_eval.py build-dataset \
  --output "${EVAL_DATASET}" \
  "${optional_limit_args[@]}"
require_file "${EVAL_DATASET}" "Judge eval dataset was not written."

run "${PYTHON_BIN}" scripts/amd/train_gemma_lora.py \
  --mode generate \
  --model-name "${BASE_MODEL}" \
  --train-file "${TRAIN_FILE}" \
  --eval-file "${EVAL_FILE}" \
  --output-dir "${AMD_OUTPUT_DIR}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --eval-subset-size "${EVAL_SUBSET_SIZE}" \
  --judge-dataset "${EVAL_DATASET}" \
  --base-results-output "${BASE_RESULTS}" \
  --lora-results-output "${LORA_RESULTS}" \
  "${bf16_args[@]}" \
  "${dry_run_args[@]}"
require_file "${BASE_RESULTS}" "Base candidate results were not written."
require_file "${LORA_RESULTS}" "LoRA candidate results were not written."

section "6. LLM-as-Judge Scoring"
run "${PYTHON_BIN}" scripts/run_llm_judge_eval.py judge \
  --dataset "${EVAL_DATASET}" \
  --base-results "${BASE_RESULTS}" \
  --lora-results "${LORA_RESULTS}" \
  --output "${JUDGE_SCORES}" \
  --judge-model "${JUDGE_MODEL}" \
  --provider "${JUDGE_PROVIDER}" \
  --endpoint "${JUDGE_ENDPOINT}" \
  "${dry_run_args[@]}"
require_file "${JUDGE_SCORES}" "Judge scores were not written."

section "7. Summary And Report Generation"
run "${PYTHON_BIN}" scripts/run_llm_judge_eval.py summarize \
  --judge-scores "${JUDGE_SCORES}" \
  --summary-json "${SUMMARY_JSON}" \
  --summary-csv "${SUMMARY_CSV}"
run "${PYTHON_BIN}" scripts/run_llm_judge_eval.py report \
  --summary-json "${SUMMARY_JSON}" \
  --output "${REPORT_MD}"

section "8. Artifact Export List"
cat <<EOF
Training:
- ${AMD_OUTPUT_DIR}/training_metrics.json
- ${AMD_OUTPUT_DIR}/training_log.txt
- ${ADAPTER_DIR}/

LLM-as-Judge:
- ${BASE_RESULTS}
- ${LORA_RESULTS}
- ${JUDGE_SCORES}
- ${SUMMARY_JSON}
- ${SUMMARY_CSV}
- ${REPORT_MD}

PPT-ready:
- ${AMD_OUTPUT_DIR}/training_metrics.json
- ${SUMMARY_CSV}
- ${SUMMARY_JSON}
- ${REPORT_MD}
EOF

section "Complete"
echo "Full AMD LoRA + judge experiment finished."
