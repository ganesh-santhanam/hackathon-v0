#!/usr/bin/env bash
set -euo pipefail

EXPORT_DIR="${EXPORT_DIR:-exports}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ZIP_PATH="${EXPORT_DIR}/industrial_ai_amd_artifacts_${TIMESTAMP}.zip"

mkdir -p "${EXPORT_DIR}"

include_paths=(
  "data/evals"
  "data/benchmarks"
  "data/amd/lora/training_metrics.json"
  "data/amd/lora/training_log.txt"
  "docs/progress.md"
)

existing_paths=()
for path in "${include_paths[@]}"; do
  if [[ -e "${path}" ]]; then
    existing_paths+=("${path}")
  else
    printf 'Skipping missing artifact: %s\n' "${path}" >&2
  fi
done

if [[ "${#existing_paths[@]}" -eq 0 ]]; then
  printf 'ERROR: No exportable artifacts found.\n' >&2
  exit 1
fi

if command -v zip >/dev/null 2>&1; then
  zip -r "${ZIP_PATH}" "${existing_paths[@]}" \
    -x '*/.cache/*' \
    -x '*/huggingface/*' \
    -x '*/HF_HOME/*' \
    -x '*/hub/*' \
    -x '*/models--*/*' \
    -x '*/checkpoint-*/*' \
    -x '*/trainer_checkpoints/*' \
    -x '*.safetensors' \
    -x '*.bin' \
    -x '*.pt' \
    -x '*.pth'
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    printf 'ERROR: zip is not installed and Python fallback was not found.\n' >&2
    exit 1
  }
  "${PYTHON_BIN}" -c '
from pathlib import Path
import sys
import zipfile

zip_path = Path(sys.argv[1])
paths = [Path(path) for path in sys.argv[2:]]
excluded_suffixes = {".safetensors", ".bin", ".pt", ".pth"}
excluded_parts = {
    ".cache",
    "huggingface",
    "HF_HOME",
    "hub",
    "trainer_checkpoints",
}

def excluded(path: Path) -> bool:
    if path.suffix in excluded_suffixes:
        return True
    if any(part in excluded_parts for part in path.parts):
        return True
    return any(part.startswith("checkpoint-") or part.startswith("models--") for part in path.parts)

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for input_path in paths:
        if input_path.is_file() and not excluded(input_path):
            archive.write(input_path, input_path.as_posix())
        elif input_path.is_dir():
            for file_path in input_path.rglob("*"):
                if file_path.is_file() and not excluded(file_path):
                    archive.write(file_path, file_path.as_posix())
' "${ZIP_PATH}" "${existing_paths[@]}"
fi

printf 'Wrote export zip: %s\n' "${ZIP_PATH}"
