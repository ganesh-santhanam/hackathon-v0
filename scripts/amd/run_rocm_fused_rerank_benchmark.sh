#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

CANDIDATES="${CANDIDATES:-10000,100000,1000000}"
EMBEDDING_DIMS="${EMBEDDING_DIMS:-384,768}"
BATCH_SIZES="${BATCH_SIZES:-1,8,32}"
MODES="${MODES:-fp32,fp16,bf16,fp8,tf32}"
RUNS="${RUNS:-5}"
WARMUP_RUNS="${WARMUP_RUNS:-2}"
TOP_K="${TOP_K:-10}"
CHART="${CHART:-1}"

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

run() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  "$@"
}

[[ -d "scripts/amd" ]] || fail "Run this script from the repository root."
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python not found: ${PYTHON_BIN}."

"${PYTHON_BIN}" - <<'PY' || fail "PyTorch is required. Install the ROCm PyTorch build on AMD Cloud."
import torch
print("torch:", torch.__version__)
print("hip:", getattr(torch.version, "hip", None))
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
print("gpu:", torch.cuda.get_device_name(0))
PY

chart_args=()
if [[ "${CHART}" == "1" || "${CHART}" == "true" || "${CHART}" == "TRUE" ]]; then
  chart_args=(--chart)
fi

export PYTHONPATH="${PYTHONPATH:-src}"

run "${PYTHON_BIN}" scripts/amd/rocm_fused_rerank_benchmark.py \
  --device cuda \
  --candidates "${CANDIDATES}" \
  --embedding-dims "${EMBEDDING_DIMS}" \
  --batch-sizes "${BATCH_SIZES}" \
  --modes "${MODES}" \
  --runs "${RUNS}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --top-k "${TOP_K}" \
  "${chart_args[@]}"

printf '\nBenchmark artifacts:\n'
printf -- '- data/benchmarks/rocm_fused_rerank_results.csv\n'
printf -- '- data/benchmarks/rocm_fused_rerank_results.json\n'
printf -- '- data/benchmarks/rocm_fused_rerank_report.md\n'
printf -- '- data/benchmarks/rocm_fused_rerank_latency.svg (when CHART=1)\n'
