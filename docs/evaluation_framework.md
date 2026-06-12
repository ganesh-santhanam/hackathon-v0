# Unified Evaluation Framework

This framework builds a complete evaluation package without changing application
behavior or the Streamlit UI. It is file-based so the same commands work on a
local NVIDIA workstation and on AMD Cloud MI300X instances.

## Outputs

The default output directory is:

```bash
data/evals/full_run/
```

The generated package contains:

- `evaluation_summary.json`
- `evaluation_summary.csv`
- `evaluation_report.md`
- `evaluation_report.html`
- `evaluation_report.pdf` when `wkhtmltopdf` or `pandoc` is available
- `charts/`
- `system_metrics.json`
- `hardware_profile.json`

## Commands

Run the full packaging flow:

```bash
PYTHONPATH=src python scripts/evals/run_full_evaluation.py
```

Capture only hardware and runtime metrics:

```bash
PYTHONPATH=src python scripts/evals/capture_system_metrics.py \
  --output-dir data/evals/full_run \
  --sample-seconds 10 \
  --interval-seconds 1
```

Generate reports from existing artifacts:

```bash
PYTHONPATH=src python scripts/evals/generate_evaluation_report.py \
  --output-dir data/evals/full_run
```

Optionally wrap training, inference, or benchmark commands so the wall clock
timings are captured in `system_metrics.json`:

```bash
PYTHONPATH=src python scripts/evals/run_full_evaluation.py \
  --training-command bash scripts/amd/run_gemma_lora_experiment.sh \
  --benchmark-command bash scripts/amd/run_rocm_fused_rerank_benchmark.sh
```

## Metric Policy

The framework never fabricates metrics. If a measured artifact is missing, the
metric is emitted as `NOT AVAILABLE` with a reason in both JSON and CSV outputs.

Measured artifacts currently consumed:

- `data/evals/summary.json`
- `data/evals/summary.csv`
- `data/evals/base_results.jsonl`
- `data/evals/lora_results.jsonl`
- `data/evals/judge_scores.jsonl`
- `data/amd/lora/training_metrics.json`
- `data/benchmarks/kernel_comparison.json`
- `data/benchmarks/rocm_fused_rerank_results.json`
- `data/incidents/ai4i_incident_corpus.jsonl`
- `data/evaluation/scenarios.json`
- `data/vision_examples/*.png`
- `models/telemetry_metrics.json` when present

## System And Hardware Metrics

`capture_system_metrics.py` records:

- hostname, OS, CPU model, RAM total
- GPU name, GPU count, and GPU memory
- Python, Torch, ROCm, CUDA, Transformers, PEFT, and vLLM versions
- sampled GPU VRAM, utilization, power draw, and temperature
- wall clock, training, inference, and benchmark runtime fields

GPU sampling uses `nvidia-smi` on local NVIDIA systems and `rocm-smi` on AMD
systems. Torch is used as a fallback for static GPU identity when CLI telemetry
is unavailable.

## Evaluation Domains

Telemetry metrics include accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and
confusion matrix when `models/telemetry_metrics.json` exists. SHAP global and
local explanations are reported only when SHAP artifacts are added.

Vision metrics include accuracy, precision, recall, and F1 when a persisted
vision metrics artifact is available. Existing heatmaps and annotated overlays
from `data/vision_examples/` are copied into the package.

Retrieval metrics include corpus size and duplicate rate from the incident
corpus. Recall@K, Precision@K, MRR, NDCG, and latency require a labeled
retrieval eval set or persisted retrieval benchmark artifact.

LLM metrics are aggregated from the LLM-as-judge summary. Hallucination score,
groundedness, relevance, judge model, and examples evaluated are included when
available. Token usage and first-token latency require endpoint-level tracing.

Agent metrics are reserved for persisted workflow traces. The package emits a
trace placeholder chart and marks workflow completion, tool success, retries,
steps, runtime, and failure breakdown as unavailable until traces are recorded.

Policy and severity metrics are computed from deterministic evaluation scenarios
and include severity accuracy, confusion matrix, false SEV1, and missed SEV1.
Override and escalation rates require operational logs.

LoRA metrics include base model, adapter model, adapter size, train loss, eval
loss, train runtime, GPU hours, judge model, examples, successes, and quality
scores when the AMD LoRA artifacts exist.

AMD/GPU benchmark metrics aggregate ROCm kernel comparison and fused reranking
artifacts. Coverage is explicitly tracked for PyTorch eager, torch.compile,
rocBLAS, Triton, BF16, FP16, FP32, and FP8.

Business metrics are clearly labeled as illustrative estimates. They are not
presented as measured production impact.

## Report Sections

The Markdown and HTML reports include:

- Hackathon Summary with the highest-signal metrics
- Executive Summary
- Portfolio Report deep dive
- Slide-ready lineage tables
- Chart inventory
- Missing metric table with reasons
- Explicit model lineage
- Explicit hardware lineage

## Extending Coverage

To raise report completeness, add measured artifacts instead of hard-coded
values:

- persist telemetry SHAP outputs under `models/`
- add a labeled retrieval eval set with expected relevant document IDs
- emit LLM first-token latency and token usage from the model endpoint
- persist agent workflow traces and tool-call outcomes
- record override and escalation logs for policy metrics
- run the package once on Local RTX 5070 and once on AMD MI300X, preserving both
  output directories for side-by-side lineage
