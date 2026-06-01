# Industrial AI Hackathon

Industrial incident investigation assistant.

Datasets:
- AI4I predictive maintenance
- Fan telemetry
- Pump telemetry
- MVTec anomaly detection

## Current Scope

This repo is being built in small steps. The current working slice is the Tier 0
AI4I telemetry agent:

```text
Telemetry row
  -> failure prediction
  -> confidence score
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

## Simple RAG Answer

Generate a deterministic evidence-based answer from retrieved incident
documents:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.rag.answer \
  "What is the likely cause of a tool wear failure?"
```

The command retrieves the top 3 relevant incident documents and formats an
answer with:

- `likely_root_cause`
- `confidence`
- `supporting_incidents`
- `evidence`
- `recommended_action`

This step does not call an external LLM. If retrieval finds no relevant
incidents, it returns a clear no-evidence response.
