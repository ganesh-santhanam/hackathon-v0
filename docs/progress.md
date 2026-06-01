# Progress Notes

This file records each larger implementation step so the project stays easy to
review and commit in small chunks.

## 2026-06-01 - AI4I Telemetry Agent Baseline

### Goal

Build the first Tier 0 slice from `Hackathon.pdf`:

```text
Telemetry row
  -> failure probability
  -> risk level
  -> evidence
```

This is intentionally scoped to AI4I telemetry only. Qdrant, RAG, vision, UI,
and approval workflows are not included yet.

### What Changed

- Added a minimal Python package under `src/industrial_ai`.
- Added central project paths in `src/industrial_ai/paths.py`.
- Added an AI4I dataset loader in `src/industrial_ai/telemetry/ai4i.py`.
- Added an XGBoost training pipeline in `src/industrial_ai/telemetry/train.py`.
- Added a risk-oriented prediction CLI/API in
  `src/industrial_ai/telemetry/predict.py`.
- Added model persistence through `joblib`: training saves the fitted pipeline
  under `models/`, and prediction loads the saved pipeline from disk.
- Added top feature importances to prediction output using the saved pipeline's
  fitted preprocessor and XGBoost classifier.
- Added focused tests under `tests/telemetry`.
- Added `pyproject.toml` for pytest and ruff configuration.
- Updated `.gitignore` so large/local artifacts stay out of Git.

### Data Handling

- Source dataset: `ai4i_dataset/ai4i2020.csv`.
- Target: overall `Machine failure` label.
- Features:
  - `type`
  - `air_temperature_k`
  - `process_temperature_k`
  - `rotational_speed_rpm`
  - `torque_nm`
  - `tool_wear_min`
- Excluded from features:
  - `udi`
  - `product_id`
  - `twf`
  - `hdf`
  - `pwf`
  - `osf`
  - `rnf`

The failure-type columns are excluded to avoid label leakage.

### Training

The training script uses:

- `train_test_split` with `test_size=0.2`
- `stratify=dataset.target`
- preprocessing inside a sklearn `Pipeline`
- `OneHotEncoder` for the `type` column
- `XGBClassifier` for binary classification

The split happens before preprocessing. The encoder and model are fit only on
the training split.

Run training:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.telemetry.train
```

Generated artifacts:

- `models/telemetry_model.joblib`
- `models/telemetry_metrics.json`

Both are ignored by Git.

Prediction loads `models/telemetry_model.joblib` from disk. It does not retrain.

### Current Metrics

Held-out test metrics:

- ROC AUC: `0.9711`
- Average precision: `0.7042`

Confusion matrices at multiple thresholds:

| Threshold | TN | FP | FN | TP | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 1743 | 189 | 4 | 64 | 0.253 | 0.941 | 0.399 |
| 0.5 | 1809 | 123 | 9 | 59 | 0.324 | 0.868 | 0.472 |
| 0.7 | 1874 | 58 | 13 | 55 | 0.487 | 0.809 | 0.608 |

For the demo, the CLI does not expose threshold selection directly. It maps
probability to risk bands:

| Risk Level | Probability |
|---|---:|
| HIGH | `>= 0.7` |
| MEDIUM | `>= 0.5` |
| LOW | `< 0.5` |

### Demo Command

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

Example output:

```json
{
  "machine_id": "FAN-023",
  "failure_probability": 0.8140159845352173,
  "failure_probability_percent": 81,
  "risk_level": "HIGH",
  "top_feature_importances": [
    {
      "feature": "rotational_speed_rpm",
      "importance": 0.27111953496932983
    },
    {
      "feature": "tool_wear_min",
      "importance": 0.24079281091690063
    },
    {
      "feature": "torque_nm",
      "importance": 0.24077202379703522
    }
  ],
  "evidence": [
    "Tool wear unusually high",
    "Torque outside normal range",
    "Rotational speed anomaly"
  ]
}
```

### Verification

Current checks:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```

Latest result:

- `6 passed`
- `All checks passed`

## 2026-06-01 - AI4I Incident Corpus

### Goal

Create the local document corpus needed before Qdrant/RAG:

```text
AI4I failure rows
  -> incident reports
  -> RCA reports
  -> maintenance notes
  -> structured local documents
```

Qdrant is intentionally not implemented in this step.

### What Changed

- Added `src/industrial_ai/incidents/generate.py`.
- Added deterministic structured document generation from AI4I rows.
- Added tests under `tests/incidents`.
- Generated a local corpus under `data/incidents/`.

### Output

Generated files:

- `data/incidents/ai4i_incident_corpus.jsonl`
- `data/incidents/manifest.json`

Corpus summary:

- Source rows: `100` AI4I failure rows
- Total documents: `300`
- Document types:
  - `incident_report`
  - `rca_report`
  - `maintenance_note`
- Size on disk: about `280 KB`

Each JSONL document includes:

- `document_id`
- `document_type`
- `source_dataset`
- `source_row_id`
- `machine_id`
- `title`
- `body`
- `metadata`
- `evidence`

### Generate Command

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.generate \
  --source-failure-rows 100
```

### Verification

Current checks:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```

Latest focused generator result:

- `tests/incidents/test_generate.py` -> `3 passed`

### Next Step

The next Tier 0 step should be small:

```text
Qdrant memory
  -> create collection
  -> embed local incident documents
  -> retrieve similar incidents
```

The corpus now exists, so the next step can focus only on storage and retrieval.

## 2026-06-01 - Local Qdrant Memory

### Goal

Index the generated incident corpus locally and retrieve similar incident
documents:

```text
Question
  -> sentence-transformers embedding
  -> local Qdrant search
  -> similar incident documents
```

LLM/RAG answering is intentionally not implemented in this step.

### What Changed

- Added `src/industrial_ai/incidents/memory.py`.
- Added local Qdrant indexing for `data/incidents/ai4i_incident_corpus.jsonl`.
- Added sentence-transformers embeddings using
  `sentence-transformers/all-MiniLM-L6-v2`.
- Added retrieval with:
  - query text
  - `top_k`
  - optional `document_type` filter
  - configurable score threshold
- Added CLI commands for indexing and searching.
- Added tests with a temporary corpus, fake embedder, and temporary Qdrant path.
- Ignored `data/qdrant/` because it is a rebuildable local index.

### Stored Payload

Each indexed Qdrant point stores:

- `document_id`
- `document_type`
- `machine_id`
- `title`
- `body`
- `metadata`
- `evidence`

### Index Command

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.memory index
```

Verified result:

```json
{
  "indexed_documents": 300,
  "collection_name": "incident_documents"
}
```

### Search Command

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.memory search \
  "tool wear and torque anomaly" \
  --top-k 2 \
  --score-threshold 0.5
```

Filtered search:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.memory search \
  "tool wear and torque anomaly" \
  --top-k 2 \
  --document-type rca_report \
  --score-threshold 0.5
```

Search output includes `top_score`, `score_threshold`, `message`, and
`results`. If every retrieved document is below the configured threshold, the
message is `No relevant incidents found` and `results` is empty.

Example top-level output:

```json
{
  "query": "tool wear and torque anomaly",
  "top_k": 2,
  "score_threshold": 0.5,
  "top_score": 0.7210513969669443,
  "message": "Relevant incidents found",
  "results": [
    {
      "score": 0.7210513969669443,
      "document_id": "ai4i-01997-rca_report",
      "document_type": "rca_report",
      "machine_id": "AI4I-01997",
      "title": "RCA Report - AI4I-01997"
    }
  ]
}
```

### Verification

Current checks:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```

Focused memory tests:

- `tests/incidents/test_memory.py` covers indexing, search, document type
  filtering, score thresholds, top score, and no-relevant-results handling.

Real local Qdrant verification:

- Indexed `300` documents into collection `incident_documents`
- Searched unfiltered query successfully
- Searched with `--document-type rca_report` successfully

Note: the first real run may need network access to download or validate the
sentence-transformers model from Hugging Face. The Qdrant index itself remains
local.

## 2026-06-01 - Simple RAG Answer Command

### Goal

Add a small answer layer on top of incident retrieval:

```text
Question
  -> retrieve top 3 relevant incidents
  -> deterministic structured answer
```

This is not an LLM integration. The answer is generated only from retrieved
incident documents.

### What Changed

- Added `src/industrial_ai/rag/answer.py`.
- Added command:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.rag.answer \
  "What is the likely cause of a tool wear failure?"
```

- Reuses existing incident retrieval.
- Retrieves top 3 relevant documents by default.
- Returns a no-evidence response when retrieval finds no relevant incidents.
- Formats answers with:
  - `likely_root_cause`
  - `confidence`
  - `supporting_incidents`
  - `evidence`
  - `recommended_action`

### Behavior

If relevant incidents are found, the command infers the likely root cause from
the retrieved documents' `metadata.failure_modes`, assigns confidence from the
top retrieval score, and lists supporting incident IDs and evidence.

If no relevant incidents are found, the command returns:

```json
{
  "likely_root_cause": "No evidence available",
  "confidence": "none",
  "supporting_incidents": [],
  "evidence": [
    "No relevant incidents found"
  ],
  "recommended_action": "Do not infer a root cause. Collect more evidence or lower the retrieval threshold."
}
```

### Verification

Focused tests:

- `tests/rag/test_answer.py` -> `6 passed`

This step intentionally does not include Streamlit, LangGraph, or external LLM
calls.

## 2026-06-01 - Severity Policy Engine

### Goal

Add deterministic severity assignment:

```text
Failure probability + RAG confidence
  -> SEV1 / SEV2 / SEV3
```

### Rules

```text
Failure probability > 80%
AND RAG confidence = high
-> SEV1

Failure probability > 50%
-> SEV2

Else
-> SEV3
```

The boundary behavior is intentional:

- Exactly `0.80` is not SEV1 unless probability is greater than `0.80`.
- Exactly `0.50` is SEV3 unless probability is greater than `0.50`.

### What Changed

- Added `src/industrial_ai/policy/severity.py`.
- Added command:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.policy.severity \
  --failure-probability 0.82 \
  --rag-confidence high
```

- Added tests under `tests/policy`.

### Example Output

```json
{
  "severity": "SEV1",
  "reason": "Failure probability is above 80% and RAG confidence is high.",
  "inputs": {
    "failure_probability": 0.82,
    "rag_confidence": "high"
  }
}
```

### Verification

Focused tests:

- `tests/policy/test_severity.py` -> `9 passed`

This step intentionally does not include UI, workflow orchestration, or LLM
calls.

## 2026-06-01 - Human Approval JSON Backend

### Goal

Add a minimal human approval layer without UI:

```text
Incident severity
  -> approval_required
  -> pending / approved / rejected / not_required
```

### Rules

- `SEV1` requires approval and starts as `pending`.
- `SEV2` and `SEV3` do not require approval and start as `not_required`.

### What Changed

- Added `src/industrial_ai/approvals/approval.py`.
- Added JSON-backed local storage at `data/approvals/approvals.json`.
- Added CLI commands:
  - `create`
  - `show`
  - `approve`
  - `reject`
- Added tests under `tests/approvals`.
- Ignored `data/approvals/` because it is local mutable demo state.

### Commands

Create:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.approvals.approval create \
  INCIDENT-001 \
  --severity SEV1
```

Approve:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.approvals.approval approve INCIDENT-001
```

Reject:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.approvals.approval reject INCIDENT-001
```

Show:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.approvals.approval show INCIDENT-001
```

### Example Record

```json
{
  "incident_id": "INCIDENT-001",
  "severity": "SEV1",
  "approval_required": true,
  "status": "pending"
}
```

### Verification

Focused tests:

- `tests/approvals/test_approval.py` -> `7 passed`

This step intentionally does not include Streamlit or any approval UI.
