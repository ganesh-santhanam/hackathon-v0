# Industrial AI Hackathon

Industrial incident investigation assistant.

Datasets:
- AI4I predictive maintenance
- Fan telemetry
- Pump telemetry
- MVTec anomaly detection

## Current Scope

This repo is being built in small steps. The current working slice is the Tier 0
incident-analysis pipeline:

```text
AI4I telemetry row
  -> failure prediction
  -> incident corpus generation
  -> local Qdrant retrieval
  -> deterministic RAG answer
  -> severity policy
  -> human approval
  -> evaluation harness
```

## AI4I Telemetry Agent

Train the baseline model:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.telemetry.train
```

The model and metrics are written under `models/`, which is ignored by Git.
Metrics include confusion matrices for thresholds `0.3`, `0.5`, and `0.7`.

Run a prediction:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.telemetry.predict \
  --machine-id FAN-023 \
  --type M \
  --air-temperature-k 301.1 \
  --process-temperature-k 311.6 \
  --rotational-speed-rpm 1266 \
  --torque-nm 55.5 \
  --tool-wear-min 210
```

The prediction output is demo-oriented:

```text
Machine: FAN-023
Failure Probability: 81%
Risk Level: HIGH
Top Feature Importances:
- torque_nm
- tool_wear_min
- rotational_speed_rpm
Evidence:
- Tool wear unusually high
- Torque outside normal range
- Rotational speed anomaly
```

Run tests:

```bash
.venv/bin/pytest -q
```

## Repository Status

The current implementation includes:

- AI4I telemetry prediction
- structured incident corpus generation
- local Qdrant indexing and retrieval
- deterministic RAG-style answer formatting
- severity assignment
- JSON-backed human approval records
- deterministic evaluation scenarios
- thin Streamlit demo UI
- MVTec comparison and autoencoder vision anomaly CLIs

The remaining work is to replace the deterministic stand-ins with the
fuller agentic workflow described in the hackathon brief.

## AMD LoRA Evaluation Results

Latest AMD Cloud recovery run:

| Field | Value |
| --- | --- |
| Base Model | `Qwen/Qwen3-4B-Instruct-2507` |
| LoRA Model | `Qwen/Qwen3-4B-Instruct-2507` with adapter `data/amd/lora/qwen4b_adapter` |
| Judge Model | `Qwen/Qwen3-14B` served through vLLM's OpenAI-compatible API |
| Hardware | AMD MI300X-class `gfx942` GPU on ROCm 7.0 / HIP 7.0 |
| Precision | BF16 for LoRA training, candidate generation, and vLLM judge serving |

Training used 40 generated LoRA examples and 10 eval examples with
`LIMIT=10`.

| Training Metric | Value |
| --- | ---: |
| train_loss | `0.1870` |
| eval_loss | `0.02746` |
| elapsed_seconds | `290.289` |

LLM-as-Judge evaluated 10 incidents, producing 20 scored candidate responses
with `20/20` successful judge records.

| Judge Metric | Base | LoRA | Improvement |
| --- | ---: | ---: | ---: |
| hallucination_score | `1.0` | `1.0` | `0.0%` |
| rca_quality | `3.7` | `4.6` | `24.32%` |
| actionability | `4.4` | `4.4` | `0.0%` |
| severity_reasoning | `3.8` | `4.6` | `21.05%` |

Generated evaluation artifacts are written under `data/evals/`; LoRA training
metrics are written under `data/amd/lora/`. These generated artifacts are not
committed by default.

## Streamlit Demo UI

Run the one-page demo app:

```bash
PYTHONPATH=src .venv/bin/streamlit run src/industrial_ai/demo/streamlit_app.py
```

The app accepts telemetry inputs and runs the local investigation workflow:

```text
prediction -> optional visual inspection -> retrieval -> deterministic RAG answer -> severity -> approval
```

It does not use LangGraph or external LLM calls.

The dashboard also includes an `Evaluation` tab. It reuses the held-out test rig
and shows:

- data sources used
- scenario count
- pass/fail counts
- pass rate
- scenario table with expected, actual, result, and key inputs
- filters for all, passed, and failed scenarios

The investigation view includes an expandable `Severity Rules` section loaded
from the production severity policy source. It shows criteria, approval
requirements, the triggered rule reason, and the exact inputs used.

The `Policy Management` tab shows the active production policy name, version,
last modified timestamp, active rules, approval requirements, and the currently
triggered severity decision. Its `Edit Policy` action is simulated/read-only.

## MVTec Vision Checks

Run the transparent comparison baseline first:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_compare \
  mvtec_anomaly_detection/cable/test/bent_wire/000.png
```

This compares the image against good MVTec references and reports a local patch
anomaly score. It is useful as a baseline, but subtle defects can overlap with
normal images.

Evaluate the comparison baseline by active MVTec category:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.evaluate comparison
```

Train the small deep-learning autoencoder on good references:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_autoencoder train cable \
  --epochs 5 \
  --reference-limit 50
```

Run the learned detector:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_autoencoder predict \
  mvtec_anomaly_detection/cable/test/bent_wire/000.png \
  --model-path models/mvtec_autoencoder_cable.pt
```

The autoencoder reconstructs images learned from good examples and uses
reconstruction error as the anomaly score. It does not call external services.

Train a ResNet18 embedding profile for one category:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet train cable \
  --reference-limit 50
```

Calibrate that category threshold:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet calibrate \
  --model-path models/mvtec_resnet_cable.npz \
  --metric f1
```

Train and calibrate all active industrial categories:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet train-all \
  --reference-limit 50 \
  --metric f1
```

Predict with the ResNet embedding detector:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet predict \
  mvtec_anomaly_detection/cable/test/bent_wire/000.png \
  --model-path models/mvtec_resnet_cable.npz
```

Evaluate the ResNet detector for the trained category:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet evaluate \
  --model-path models/mvtec_resnet_cable.npz
```

The ResNet detector uses embeddings rather than supervised defect labels:
good images define the normal feature center, and distance from that center is
the anomaly score. For best accuracy, use pretrained ResNet weights. If weights
are not available locally, the first run may need to download them.

Run the held-out demo correctness rig:

```bash
TORCH_HOME=/tmp/torch-cache PYTHONPATH=src .venv/bin/python -m industrial_ai.evaluation.test_rig \
  --category cable \
  --category grid \
  --category metal_nut \
  --category screw \
  --category transistor
```

The rig checks AI4I held-out rows, MVTec `test/` images, severity rules, and
approval behavior. It exits non-zero if any scenario fails.

## Incident Corpus

Generate local structured incident documents from AI4I failure rows:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.generate \
  --source-failure-rows 100
```

This writes:

- `data/incidents/ai4i_incident_corpus.jsonl`
- `data/incidents/manifest.json`

The default output is 300 documents: 100 incident reports, 100 RCA reports, and
100 maintenance notes. Qdrant is intentionally not part of this step.

## Local Qdrant Memory

Index the generated incident corpus into a local Qdrant collection:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.memory index
```

Search the indexed corpus:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.memory search \
  "tool wear and torque anomaly" \
  --top-k 2 \
  --score-threshold 0.5
```

Filter by document type:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.memory search \
  "tool wear and torque anomaly" \
  --top-k 2 \
  --document-type rca_report \
  --score-threshold 0.5
```

The local Qdrant index is written under `data/qdrant/`, which is ignored by Git.
This step only retrieves similar documents; it does not generate LLM/RAG answers.
Search output includes `top_score`, `score_threshold`, and a message. If all
retrieved documents are below the threshold, the message is
`No relevant incidents found` and `results` is empty.

Search can optionally rerank vector results using exact telemetry inputs:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.memory search \
  "tool wear and torque anomaly" \
  --top-k 3 \
  --score-threshold 0.5 \
  --telemetry-rerank \
  --tool-wear-min 210 \
  --torque-nm 55.5 \
  --rotational-speed-rpm 1266 \
  --air-temperature-k 301.1 \
  --process-temperature-k 311.6
```

Telemetry-aware results include `vector_score`,
`telemetry_similarity_score`, and `combined_score`.

## Simple RAG Answer

Generate a deterministic evidence-based answer from retrieved incident
documents:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.rag.answer \
  "What is the likely cause of a tool wear failure?"
```

Optionally synthesize the answer with a local Ollama model:

```bash
OLLAMA_MODEL=gemma3:4b PYTHONPATH=src .venv/bin/python -m industrial_ai.rag.answer \
  "What is the likely cause of a tool wear failure?" \
  --llm
```

`OLLAMA_MODEL` defaults to `gemma3:4b`. If Ollama is unavailable, the command
falls back to deterministic RAG and reports the fallback reason in
`limitations`. Use `--no-fallback` to fail instead.

The command retrieves the top 3 relevant incident documents and formats an
answer with:

- `likely_root_cause`
- `confidence`
- `supporting_incidents`
- `evidence`
- `recommended_action`
- `limitations`

No cloud LLM is used. If retrieval finds no relevant incidents, it returns a
clear no-evidence response.

## Severity Policy

Assign deterministic incident severity from failure probability and retrieval
confidence:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.policy.severity \
  --failure-probability 0.82 \
  --rag-confidence high
```

Rules:

```text
Failure probability > 80% AND visual defect detected -> SEV1
Failure probability > 80% AND RAG confidence = high -> SEV1
Failure probability > 50% -> SEV2
Else -> SEV3
```

## Human Approval

Create a JSON-backed approval record:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.approvals.approval create \
  INCIDENT-001 \
  --severity SEV1
```

Approve or reject:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.approvals.approval approve INCIDENT-001
PYTHONPATH=src .venv/bin/python -m industrial_ai.approvals.approval reject INCIDENT-001
```

Show current status:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.approvals.approval show INCIDENT-001
```

SEV1 incidents require approval and start as `pending`. SEV2 and SEV3 incidents
do not require approval and start as `not_required`. The local approval store is
written to `data/approvals/`, which is ignored by Git.

## Evaluation Harness

Run deterministic evaluation scenarios:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.evaluation.harness
```

Scenarios live in:

```text
data/evaluation/scenarios.json
```

The harness reports:

- scenario
- expected severity
- actual severity
- pass/fail

The current scenario set contains 12 severity-policy scenarios.
