# Evaluation Framework

The demo evaluation stack has three layers:

1. incident answer quality through LLM-as-judge outputs in `data/evals/`
2. LoRA training evidence in `data/amd/lora/`
3. ROCm reranking throughput in `data/benchmarks/`

These layers are intentionally file-based so the hackathon demo stays local-first
and easy to export.

## Current Artifacts

```text
data/evals/eval_dataset.jsonl
data/evals/base_results.jsonl
data/evals/lora_results.jsonl
data/evals/judge_scores.jsonl
data/evals/summary.csv
data/evals/summary.json
data/evals/llm_judge_report.md
data/amd/lora/training_metrics.json
data/amd/lora/training_log.txt
data/benchmarks/rocm_fused_rerank_results.csv
data/benchmarks/rocm_fused_rerank_results.json
data/benchmarks/rocm_fused_rerank_report.md
```

## AMD Experiment Flow

Run LoRA and judge evaluation:

```bash
PYTHONPATH=src \
PYTHON_BIN=.venv/bin/python \
BASE_MODEL=google/gemma-3-4b-it \
JUDGE_MODEL=gpt-oss:20b \
MAX_STEPS=100 \
BF16=1 \
bash scripts/amd/run_full_amd_experiment.sh
```

Run ROCm reranking:

```bash
PYTHONPATH=src \
PYTHON_BIN=.venv/bin/python \
bash scripts/amd/run_rocm_fused_rerank_benchmark.sh
```

Export slide artifacts:

```bash
bash scripts/amd/export_artifacts.sh
```

The export zip is written under `exports/` with a UTC timestamp.

## What To Put In Slides

Use `data/evals/summary.csv` for answer-quality comparison. Use
`data/amd/lora/training_metrics.json` for training runtime, BF16 status, and loss
signals. Use `data/benchmarks/rocm_fused_rerank_results.csv` for ROCm latency,
throughput, speedup, and top-k overlap.

The strongest demo story is:

```text
Better industrial answers from the LoRA experiment, plus faster incident memory
reranking on AMD ROCm for the retrieval path.
```

## Export Policy

`scripts/amd/export_artifacts.sh` includes:

- `data/evals/`
- `data/benchmarks/`
- `data/amd/lora/training_metrics.json`
- `data/amd/lora/training_log.txt`
- `docs/progress.md`

It excludes Hugging Face caches, model weights, trainer checkpoints, and
`.safetensors` files. This keeps the artifact zip suitable for sharing and PPT
prep without bundling large model assets.
