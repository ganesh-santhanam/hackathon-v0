# Evaluation

## Fast Validation

```bash
make test
make eval
```

`make test` runs the pytest suite. `make eval` runs the deterministic severity harness.

## Demo Correctness Rig

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.evaluation.test_rig --category cable
```

This checks telemetry, vision, policy, and approval behavior using local artifacts.

## Unified Evaluation Package

```bash
PYTHONPATH=src .venv/bin/python scripts/evals/run_full_evaluation.py
```

Outputs are written to `data/evals/full_run/` and include summary JSON/CSV, Markdown,
HTML, charts, and hardware/runtime metadata when available.

## LLM Judge

The LLM judge harness can use local Ollama or an OpenAI-compatible endpoint such as
local vLLM. It should not require paid cloud APIs.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_llm_judge_eval.py --help
```

## Metric Policy

The evaluation tooling should not fabricate metrics. Missing artifacts should be
reported as unavailable with a reason.

## Regression Focus

Keep deterministic tests around:

- severity rules
- approval lifecycle
- retrieval thresholding and reranking
- Streamlit helper behavior
- RAG fallback behavior
- evaluation scenario pass/fail semantics
