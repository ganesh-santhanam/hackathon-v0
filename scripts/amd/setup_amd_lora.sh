#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

if command -v rocminfo >/dev/null 2>&1; then
  echo "ROCm detected:"
  rocminfo | grep -m 1 "Name:" || true
else
  echo "Warning: rocminfo not found. Continue only if this AMD Cloud image already exposes ROCm to PyTorch."
fi

if "${PYTHON_BIN}" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") else 1)
PY
then
  echo "torch is already installed."
else
  echo "Installing ROCm PyTorch wheels."
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install --index-url https://download.pytorch.org/whl/rocm6.1 \
    torch torchvision torchaudio
fi

"${PYTHON_BIN}" -m pip install --upgrade \
  "transformers>=4.44.0" \
  "peft>=0.12.0" \
  "accelerate>=0.33.0" \
  "datasets>=2.20.0" \
  "sentencepiece>=0.2.0" \
  "protobuf>=4.25.0"

"${PYTHON_BIN}" - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
    print("bf16_supported:", torch.cuda.is_bf16_supported())
else:
    print("Warning: no GPU visible to PyTorch. Training script will fail gracefully unless --dry-run is used.")
PY
