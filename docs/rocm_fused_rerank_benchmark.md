# ROCm Fused Rerank Benchmark

This benchmark measures the industrial incident memory reranking kernel used in
the demo architecture, without modifying application logic or Streamlit.

For each query and candidate incident it:

1. normalizes query and incident embeddings
2. computes embedding similarity
3. computes a telemetry penalty from `air_temperature`, `process_temperature`,
   `rotational_speed`, `torque`, and `tool_wear`
4. combines `final_score = embedding_score - telemetry_penalty + weighted_rerank_bonus`
5. returns top-k candidates

## Files

```text
scripts/amd/rocm_fused_rerank_benchmark.py
scripts/amd/run_rocm_fused_rerank_benchmark.sh
data/benchmarks/rocm_fused_rerank_results.csv
data/benchmarks/rocm_fused_rerank_results.json
data/benchmarks/rocm_fused_rerank_report.md
data/benchmarks/rocm_fused_rerank_latency.svg
```

The benchmark generates synthetic embeddings and telemetry with fixed seeds. It
does not read production incident memory or change the app.

## Local Dry Run

Use this on a laptop or CPU-only machine to verify wiring:

```bash
PYTHONPATH=src .venv/bin/python scripts/amd/rocm_fused_rerank_benchmark.py \
  --dry-run \
  --chart
```

Dry run shrinks the benchmark to one small CPU-compatible case and skips GPU-only
precision modes gracefully.

## AMD Cloud Command

Run the real ROCm benchmark from the repository root:

```bash
PYTHONPATH=src \
PYTHON_BIN=.venv/bin/python \
CANDIDATES=10000,100000,1000000 \
EMBEDDING_DIMS=384,768 \
BATCH_SIZES=1,8,32 \
MODES=fp32,fp16,bf16,fp8,tf32 \
RUNS=5 \
WARMUP_RUNS=2 \
CHART=1 \
bash scripts/amd/run_rocm_fused_rerank_benchmark.sh
```

The wrapper checks that PyTorch can see a GPU before running. If no AMD GPU is
visible, it exits with a direct error instead of producing misleading CPU numbers.

## Precision Modes

- `fp32`: baseline for accuracy and speedup comparisons.
- `fp16`: enabled when a CUDA/ROCm device is visible.
- `bf16`: enabled when PyTorch reports BF16 support.
- `fp8`: reported as skipped unless native FP8 matmul is cleanly exposed by the
  installed PyTorch build.
- `tf32`: skipped on ROCm unless PyTorch exposes it cleanly. TF32 is mainly a
  CUDA/NVIDIA control and should not be presented as an AMD result unless the
  runtime explicitly supports it.

Unsupported modes are written to the CSV, JSON, and Markdown report with
`status=skipped` and a skip reason.

## Metrics

Each successful row includes:

- `latency_ms`
- `candidates_per_second`
- `speedup_vs_fp32`
- `effective_ops_per_second`
- `peak_vram_gb`
- `gpu_utilization`, when `rocm-smi` is available
- `max_abs_error_vs_fp32`
- `mean_abs_error_vs_fp32`
- `top_k_overlap_vs_fp32`

Use `latency_ms` and `candidates_per_second` for the primary slide. Use
`top_k_overlap_vs_fp32` to show whether reduced precision preserved the same
incident ranking.

## PPT Usage

Recommended slide assets:

- `data/benchmarks/rocm_fused_rerank_report.md`: summary table and skipped modes.
- `data/benchmarks/rocm_fused_rerank_results.csv`: chart source for latency and
  throughput bars.
- `data/benchmarks/rocm_fused_rerank_latency.svg`: quick visual if generated.
- `data/benchmarks/rocm_fused_rerank_results.json`: reproducibility metadata.

Suggested slide headline:

```text
ROCm accelerates telemetry-aware incident memory reranking while preserving top-k quality.
```
