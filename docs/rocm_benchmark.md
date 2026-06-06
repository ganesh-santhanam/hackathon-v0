# ROCm LLM Benchmark

This project includes a small LLM benchmark harness that runs locally today and can be reused on AMD Cloud later. It does not require ROCm or an AMD GPU on a local machine.

## Local CPU Baseline

Use Ollama locally for a CPU or whatever backend Ollama is already using. Streaming mode is recommended because it can measure time to first token (TTFT):

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_llm.py \
  --provider ollama \
  --endpoint http://localhost:11434/api/generate \
  --model gemma3:4b \
  --hardware-label local-cpu \
  --prompt "What is the likely cause of a tool wear failure?" \
  --runs 3 \
  --stream
```

Results append to:

```text
data/benchmarks/llm_benchmark.jsonl
```

The aggregate summary is written to:

```text
data/benchmarks/llm_benchmark_summary.json
```

If Ollama is not running, use a dry run to verify wiring without making a network request:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_llm.py --dry-run --runs 1 --stream
```

## Future AMD Cloud ROCm Run

Real ROCm benchmark numbers require an AMD GPU VM and a ROCm-backed serving stack, such as vLLM exposing an OpenAI-compatible chat completions endpoint.

Example once an OpenAI-compatible endpoint is available:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_llm.py \
  --provider openai-compatible \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model <served-model-name> \
  --hardware-label amd-cloud-rocm-mi300 \
  --prompt "What is the likely cause of a tool wear failure?" \
  --runs 10 \
  --stream \
  --timeout-seconds 180
```

If the endpoint requires an API key, set:

```bash
export OPENAI_API_KEY=<token>
```

## Comparing Results

Each JSONL record includes:

- `total_latency_ms`
- `ttft_ms`
- `generation_latency_ms`
- `prompt_char_count`
- `output_char_count`
- `timestamp`
- `provider`
- `endpoint`
- `model`
- `hardware_label`
- `success`
- `error_type`
- `status_code`
- `timeout_seconds`
- `streaming`
- optional token counts from the endpoint
- optional `tokens_per_second`

TTFT means time to first token: the elapsed time from sending the request until the first generated content chunk arrives. Streaming is needed for accurate TTFT because non-streaming responses only arrive after the full completion is done. If an OpenAI-compatible endpoint does not expose usable streaming chunks, `ttft_ms` may be `null`.

The summary JSON includes:

- `success_count`
- `error_count`
- `p50_total_latency_ms`
- `p95_total_latency_ms`
- `p99_total_latency_ms`
- `p50_ttft_ms`
- `p95_ttft_ms`
- `mean_tokens_per_second`

Interpretation:

- `p50` is the median run and is useful for typical latency.
- `p95` shows slow-tail behavior that users notice during demos.
- `p99` is the worst-tail estimate for larger run counts.
- `mean_tokens_per_second` is throughput during generation when token counts are returned.

Compare runs by filtering on `hardware_label`, `provider`, and `model`. For a quick shell view:

```bash
tail -n 20 data/benchmarks/llm_benchmark.jsonl
```

Use the same prompt and `--runs` count when comparing local CPU and AMD Cloud ROCm results.

## Notes

- The harness does not install ROCm.
- The harness does not require AMD hardware locally.
- Ollama token counts are recorded when Ollama returns `prompt_eval_count` and `eval_count`.
- OpenAI-compatible token counts are recorded when the endpoint returns a `usage` object.
- OpenAI-compatible streaming parses Server-Sent Events (`data:` chunks) when the serving stack emits them.
