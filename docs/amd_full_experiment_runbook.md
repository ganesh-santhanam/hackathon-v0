# AMD Full LoRA + Judge Experiment Runbook

This runbook is the shortest path for running the full AMD Cloud experiment:

1. Environment check
2. Incident corpus generation
3. LoRA train/eval dataset preparation
4. Gemma base/LoRA training
5. Base and LoRA candidate generation
6. GPT-OSS LLM-as-Judge scoring
7. Summary and slide-ready report generation
8. Artifact export list

The workflow is standalone. It does not modify application logic or Streamlit.

## Main Script

```text
scripts/amd/run_full_amd_experiment.sh
```

## Environment Variables

Required knobs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BASE_MODEL` | `google/gemma-3-4b-it` | Hugging Face Gemma-compatible base checkpoint |
| `JUDGE_MODEL` | `gpt-oss:20b` | Judge model name served by the judge endpoint |
| `MAX_STEPS` | `100` | LoRA training step cap |
| `LIMIT` | unset | Optional small example limit for dataset and judge dataset generation |
| `BF16` | `1` | Pass `--bf16` to training/generation |

Useful optional knobs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PYTHON_BIN` | `.venv/bin/python` | Python interpreter |
| `DRY_RUN` | `0` | Validate flow without GPU/model downloads |
| `JUDGE_PROVIDER` | `openai-compatible` | Judge endpoint type |
| `JUDGE_ENDPOINT` | `http://localhost:8000/v1/chat/completions` | GPT-OSS serving URL |
| `SOURCE_FAILURE_ROWS` | `100` | AI4I failure rows used when generating corpus |
| `EVAL_SUBSET_SIZE` | `64` | Eval subset for LoRA training/generation |
| `BATCH_SIZE` | `1` | Per-device batch size |
| `GRAD_ACCUM` | `8` | Gradient accumulation steps |

## AMD Cloud Setup

From a fresh AMD Cloud VM:

```bash
git clone <repo-url>
cd Hackathon
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/amd/setup_amd_lora.sh
```

If the Gemma checkpoint is gated:

```bash
huggingface-cli login
```

Start or confirm the GPT-OSS 20B judge endpoint. The default assumes an
OpenAI-compatible endpoint:

```text
http://localhost:8000/v1/chat/completions
```

## One-Command Full Run

```bash
PYTHONPATH=src \
PYTHON_BIN=.venv/bin/python \
BASE_MODEL=google/gemma-3-4b-it \
JUDGE_MODEL=gpt-oss:20b \
MAX_STEPS=100 \
BF16=1 \
bash scripts/amd/run_full_amd_experiment.sh
```

Small AMD smoke run:

```bash
PYTHONPATH=src \
PYTHON_BIN=.venv/bin/python \
LIMIT=5 \
MAX_STEPS=4 \
BF16=1 \
bash scripts/amd/run_full_amd_experiment.sh
```

Local dry run with no AMD GPU or model downloads:

```bash
PYTHONPATH=src \
PYTHON_BIN=.venv/bin/python \
DRY_RUN=1 \
LIMIT=2 \
MAX_STEPS=4 \
bash scripts/amd/run_full_amd_experiment.sh
```

If GPT-OSS is served through Ollama instead of an OpenAI-compatible server:

```bash
PYTHONPATH=src \
PYTHON_BIN=.venv/bin/python \
JUDGE_PROVIDER=ollama \
JUDGE_ENDPOINT=http://localhost:11434/api/generate \
JUDGE_MODEL=gpt-oss:20b \
bash scripts/amd/run_full_amd_experiment.sh
```

## Expected Outputs For PPT

Training:

```text
data/amd/lora/training_metrics.json
data/amd/lora/training_log.txt
data/amd/lora/gemma3b_adapter/
```

Candidate outputs:

```text
data/evals/base_results.jsonl
data/evals/lora_results.jsonl
```

Judge outputs:

```text
data/evals/judge_scores.jsonl
data/evals/summary.json
data/evals/summary.csv
data/evals/llm_judge_report.md
```

Best PPT screenshots:

- `data/amd/lora/training_metrics.json`: training loss, eval loss, runtime, BF16 status
- `data/evals/summary.csv`: model comparison table
- `data/evals/summary.json`: judge metadata and metric distributions
- `data/evals/llm_judge_report.md`: slide-ready tables and conclusions
- terminal output from `scripts/amd/run_full_amd_experiment.sh`: artifact export list

## What The Script Does

The wrapper runs these commands internally:

```bash
python -m industrial_ai.incidents.generate
python scripts/prepare_lora_dataset.py
python scripts/amd/train_gemma_lora.py --mode train
python scripts/run_llm_judge_eval.py build-dataset
python scripts/amd/train_gemma_lora.py --mode generate
python scripts/run_llm_judge_eval.py judge
python scripts/run_llm_judge_eval.py summarize
python scripts/run_llm_judge_eval.py report
```

It prints section headers and stops at the first failed step with a direct error
message.

## Failure Modes

Missing Python interpreter:

```text
ERROR: Python not found: .venv/bin/python. Set PYTHON_BIN=/path/to/python.
```

Fix:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Missing training dependencies:

```text
ERROR: Missing AMD training dependencies. Run: bash scripts/amd/setup_amd_lora.sh
```

Fix:

```bash
bash scripts/amd/setup_amd_lora.sh
```

No ROCm/GPU visible:

```text
ERROR: No ROCm/GPU device visible to PyTorch. Verify AMD Cloud GPU setup or use DRY_RUN=1.
```

Fix:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY
```

Missing AI4I source CSV:

```text
ERROR: Missing ai4i_dataset/ai4i2020.csv.
```

Fix: place the AI4I CSV at `ai4i_dataset/ai4i2020.csv`, or provide a generated
`data/incidents/ai4i_incident_corpus.jsonl`.

Gemma model access denied:

```text
401 Client Error
```

Fix:

```bash
huggingface-cli login
```

Judge endpoint unavailable:

```text
url_error / connection refused
```

Fix: start GPT-OSS 20B serving and set `JUDGE_PROVIDER`, `JUDGE_ENDPOINT`, and
`JUDGE_MODEL` to match the server.

Out of memory:

Fix by lowering:

```bash
MAX_STEPS=25 EVAL_SUBSET_SIZE=16 BATCH_SIZE=1 GRAD_ACCUM=16
```

or use a smaller `BASE_MODEL`.

## Expected Runtime

Approximate MI300X runtime:

- environment setup: 5-15 minutes
- corpus and dataset generation: under 1 minute
- LoRA training, 100 steps: 10-40 minutes after model download
- base and LoRA generation, 50 examples: 10-45 minutes
- GPT-OSS judge scoring: 10-45 minutes depending endpoint throughput
- summary/report generation: under 1 minute

First run can be slower due to Hugging Face model downloads.
