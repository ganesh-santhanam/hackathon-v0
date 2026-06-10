# AMD Cloud Qwen LoRA Experiment

This workflow fine-tunes a Qwen-compatible Hugging Face model with LoRA on the
industrial incident instruction dataset, then emits candidate outputs for the
existing LLM-as-Judge pipeline.

It is standalone. It does not modify application logic or Streamlit.

## Inputs

```text
data/lora/train.jsonl
data/lora/eval.jsonl
```

Generate them if needed:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_lora_dataset.py
```

Each row has:

```json
{
  "instruction": "Generate a concise root cause analysis from the incident evidence.",
  "input": "Document title: ...",
  "output": "Likely root cause: ...",
  "task_type": "rca_generation"
}
```

## Added Scripts

```text
scripts/amd/setup_amd_lora.sh
scripts/amd/train_gemma_lora.py
scripts/amd/run_gemma_lora_experiment.sh
```

## Default Model

The default model is:

```text
Qwen/Qwen3-4B-Instruct-2507
```

This is a small Qwen instruction checkpoint suitable for a practical
hackathon LoRA run. To use a specific Qwen-compatible checkpoint available
in your Hugging Face account or AMD image, pass:

```bash
--model-name <your-qwen-compatible-model>
```

or set:

```bash
export MODEL_NAME=<your-qwen-compatible-model>
```

## AMD Cloud Setup

From the repository root on the AMD Cloud VM:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/amd/setup_amd_lora.sh
```

If the model is gated, authenticate with Hugging Face:

```bash
huggingface-cli login
```

Check GPU visibility:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
print(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "no bf16")
PY
```

## Train LoRA Adapter

Minimal MI300X-oriented run:

```bash
PYTHONPATH=src python scripts/amd/train_gemma_lora.py \
  --mode train \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --train-file data/lora/train.jsonl \
  --eval-file data/lora/eval.jsonl \
  --output-dir data/amd/lora \
  --max-steps 100 \
  --num-epochs 1 \
  --learning-rate 2e-4 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --bf16
```

Smoke run on AMD Cloud:

```bash
PYTHONPATH=src python scripts/amd/train_gemma_lora.py \
  --mode train \
  --smoke-test \
  --max-steps 4 \
  --bf16
```

Local dry run, with no model downloads:

```bash
PYTHONPATH=src python scripts/amd/train_gemma_lora.py --mode train --dry-run
```

The script fails gracefully if no ROCm/GPU device is visible, unless `--dry-run`
or `--force-cpu` is used.

## Expected Training Outputs

```text
data/amd/lora/qwen4b_adapter/
data/amd/lora/training_metrics.json
data/amd/lora/training_log.txt
```

Do not commit adapter weights or generated training artifacts.

## Generate Candidate Outputs For LLM-as-Judge

First build the judge dataset:

```bash
PYTHONPATH=src python scripts/run_llm_judge_eval.py build-dataset \
  --output data/evals/eval_dataset.jsonl
```

Then generate both base and LoRA outputs from the Hugging Face model stack:

```bash
PYTHONPATH=src python scripts/amd/train_gemma_lora.py \
  --mode generate \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --adapter-dir data/amd/lora/qwen4b_adapter \
  --judge-dataset data/evals/eval_dataset.jsonl \
  --base-results-output data/evals/base_results.jsonl \
  --lora-results-output data/evals/lora_results.jsonl \
  --bf16
```

Generated files:

```text
data/evals/base_results.jsonl
data/evals/lora_results.jsonl
```

Those files are compatible with:

```bash
PYTHONPATH=src python scripts/run_llm_judge_eval.py judge \
  --dataset data/evals/eval_dataset.jsonl \
  --base-results data/evals/base_results.jsonl \
  --lora-results data/evals/lora_results.jsonl \
  --judge-model gpt-oss:20b
```

Then summarize:

```bash
PYTHONPATH=src python scripts/run_llm_judge_eval.py summarize
PYTHONPATH=src python scripts/run_llm_judge_eval.py report
```

Final judge artifacts:

```text
data/evals/judge_scores.jsonl
data/evals/summary.json
data/evals/summary.csv
data/evals/llm_judge_report.md
```

## One-Command AMD Experiment

```bash
PYTHONPATH=src PYTHON_BIN=.venv/bin/python \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  MAX_STEPS=100 \
  NUM_EPOCHS=1 \
  BATCH_SIZE=1 \
  GRAD_ACCUM=8 \
  bash scripts/amd/run_gemma_lora_experiment.sh
```

Dry-run version:

```bash
PYTHONPATH=src PYTHON_BIN=.venv/bin/python DRY_RUN=1 \
  bash scripts/amd/run_gemma_lora_experiment.sh
```

## Expected Runtime

Approximate MI300X ranges:

- setup: 5-15 minutes, depending image and package cache
- smoke test: 1-3 minutes after model download
- 100 LoRA steps: 10-40 minutes, depending checkpoint, sequence length, and storage
- candidate generation for 50 examples: 10-45 minutes
- GPT-OSS judge pass: depends on the serving endpoint, usually another 10-45 minutes

First run can be slower because Hugging Face downloads the base model.

## Practical Notes

- Use `--eval-subset-size 64` by default to keep evaluation quick.
- `--bf16` is recommended on MI300X when PyTorch reports BF16 support.
- Gradient checkpointing is enabled by default to reduce memory pressure.
- The scripts intentionally avoid quantization-specific dependencies because
  common CUDA-first quantization paths are brittle on ROCm.
- If local development has no ROCm device, use `--dry-run` for script validation.
