# LoRA Experiment

This project can use LoRA later to adapt a small local LLM to industrial incident language, RCA style, maintenance recommendations, and evidence-focused explanations. This step only prepares train/eval JSONL files. It does not train a model and does not require a GPU.

## Base Model

Good initial candidates are small instruction-tuned models that can run in local or AMD Cloud workflows:

- Gemma small instruction model
- Qwen small instruction model

The exact model should be chosen during the future training phase based on memory limits, license constraints, and AMD MI300 availability.

## Dataset Source

The dataset is generated from:

```text
data/incidents/ai4i_incident_corpus.jsonl
```

That corpus is already generated from AI4I failure rows into incident reports, RCA reports, and maintenance notes.

## Dataset Format

Each JSONL row uses a simple instruction-tuning shape:

```json
{
  "instruction": "Generate a concise root cause analysis from the incident evidence.",
  "input": "Document title: ...",
  "output": "Likely root cause: ...",
  "task_type": "rca_generation",
  "source_document_id": "ai4i-00078-rca_report",
  "source_document_type": "rca_report"
}
```

Generated task types:

- `rca_generation`
- `recommended_action_generation`
- `severity_explanation`
- `maintenance_summary`
- `evidence_extraction`

## Generate Train/Eval JSONL

Run from the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_lora_dataset.py
```

For a small smoke test:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_lora_dataset.py --limit 20
```

Outputs:

```text
data/lora/train.jsonl
data/lora/eval.jsonl
```

Generation is deterministic by default:

- fixed seed: `42`
- default eval ratio: `0.2`
- deterministic input order before seeded split

## Future AMD MI300 Training Path

On AMD Cloud with MI300, the future workflow would be:

1. Generate `data/lora/train.jsonl` and `data/lora/eval.jsonl`.
2. Select a Gemma or Qwen small instruction model.
3. Train LoRA adapters with a ROCm-compatible training stack.
4. Save adapter artifacts under `models/` or a separate ignored adapter directory.
5. Compare base model vs LoRA on held-out incident prompts.

This repository does not train LoRA yet.

## Base Vs LoRA Evaluation Plan

Use the same held-out prompts for both models and compare:

- RCA correctness against failure mode metadata
- evidence faithfulness
- actionability of recommended maintenance steps
- severity explanation consistency
- latency and throughput with `scripts/benchmark_llm.py`

## Risks And Limitations

- The corpus is synthetic/generated from AI4I rows, not real maintenance logs.
- Outputs may teach the model the deterministic phrasing used by the generator.
- Severity examples are simplified and should not replace the existing policy engine.
- Training requires separate GPU setup and should be validated on AMD Cloud before claiming ROCm performance.
- Generated `data/lora/*.jsonl` files are ignored by default to avoid committing large or repeatedly regenerated artifacts.
