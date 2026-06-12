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

## 2026-06-01 - Evaluation Harness

### Goal

Add a deterministic evaluation layer:

```text
Scenario
  -> expected severity
  -> actual severity
  -> pass / fail
```

This first harness evaluates the severity policy only. It does not call the
telemetry model, Qdrant, RAG command, or any external service.

### What Changed

- Added `src/industrial_ai/evaluation/harness.py`.
- Added scenario data in `data/evaluation/scenarios.json`.
- Added tests under `tests/evaluation`.
- Added CLI command:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.evaluation.harness
```

### Scenario Set

Current scenario count: `12`.

The scenarios cover:

- SEV1 high probability + high RAG confidence
- SEV2 high probability without high RAG confidence
- SEV2 moderate probability
- SEV3 exact 50 percent boundary
- SEV3 low probability
- percentage-style probability inputs
- exact 80 percent boundary behavior

### Example Summary

```json
{
  "total": 12,
  "passed": 12,
  "failed": 0,
  "pass_rate": 1.0
}
```

Each result includes:

- `scenario_id`
- `description`
- `expected_severity`
- `actual_severity`
- `passed`
- `reason`

### Verification

Focused tests:

- `tests/evaluation/test_harness.py` -> `4 passed`

Real harness run:

- `12` total scenarios
- `12` passed
- `0` failed
- `1.0` pass rate

## 2026-06-01 - Current Project Status

### Scope Completed So Far

The repo now contains a complete local Tier 0 pipeline:

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

### Implemented Pieces

- `src/industrial_ai/telemetry`
  - dataset loading
  - XGBoost training
  - prediction CLI
  - feature importance output
- `src/industrial_ai/incidents`
  - structured incident, RCA, and maintenance-note generation
  - local Qdrant indexing and search
- `src/industrial_ai/rag`
  - deterministic evidence-based answer formatting
- `src/industrial_ai/policy`
  - severity assignment from probability and retrieval confidence
- `src/industrial_ai/approvals`
  - JSON-backed approval state machine
- `src/industrial_ai/evaluation`
  - deterministic severity-policy scenarios
  - pass/fail summary output

### Current Artifacts

- `models/telemetry_model.joblib`
- `models/telemetry_metrics.json`
- `data/incidents/ai4i_incident_corpus.jsonl`
- `data/incidents/manifest.json`
- `data/qdrant/`
- `data/approvals/`
- `data/evaluation/scenarios.json`

### Verification Snapshot

Latest recorded checks:

- `72 passed`
- `All checks passed`

### Remaining Gap

The current implementation is still mostly deterministic and local. The next
step is to wire these pieces into a more complete agent workflow with richer
orchestration and, if required, UI support.

## 2026-06-02 - Thin Streamlit Demo UI

### Goal

Add a one-page demo app that runs the existing local investigation workflow:

```text
telemetry inputs
  -> prediction
  -> retrieval from local Qdrant
  -> deterministic RAG answer
  -> severity policy
  -> approval record
```

### What Changed

- Added `src/industrial_ai/demo/investigation.py`.
- Added `src/industrial_ai/demo/streamlit_app.py`.
- Added tests under `tests/demo`.
- Updated demo documentation with the Streamlit launch command.

### Launch Command

```bash
PYTHONPATH=src .venv/bin/streamlit run src/industrial_ai/demo/streamlit_app.py
```

### UI Output

The app displays:

- failure probability
- risk level
- prediction evidence
- similar incidents
- likely root cause
- recommended action
- severity
- approval ID and status

### Scope Boundaries

This step intentionally does not include:

- LangGraph
- vision
- external LLM calls

### Verification

Focused tests:

- `tests/demo/test_investigation.py` -> `3 passed`

## 2026-06-02 - MVTec Vision Baseline And Autoencoder

### Goal

Add the first vision slice in two steps:

```text
MVTec image
  -> comparison baseline against good references
  -> deep-learning autoencoder anomaly score
```

### What Changed

- Added `src/industrial_ai/vision/mvtec_compare.py`.
- Added `src/industrial_ai/vision/mvtec_autoencoder.py`.
- Added `src/industrial_ai/vision/__init__.py`.
- Added MVTec dataset path constant in `src/industrial_ai/paths.py`.
- Added focused tests under `tests/vision`.
- Updated demo commands and README documentation.

### Comparison Baseline

The comparison CLI uses good MVTec training images as references. It computes a
mean good reference image and reports the highest local patch difference as the
anomaly score.

Run:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_compare \
  mvtec_anomaly_detection/cable/test/bent_wire/000.png
```

Calibration examples showed why this baseline is only a first step:
global nearest-reference distance missed visible defects, while local patch
distance separated obvious defects better. Subtle defects still overlap
with normal samples.

### Deep Learning Autoencoder

The autoencoder CLI trains a small PyTorch convolutional autoencoder on good
MVTec reference images and saves the model under `models/`.

Train:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_autoencoder train cable \
  --epochs 5 \
  --reference-limit 50
```

Predict:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_autoencoder predict \
  mvtec_anomaly_detection/cable/test/bent_wire/000.png \
  --model-path models/mvtec_autoencoder_cable.pt
```

### Scope Boundaries

This step does not connect vision to the Streamlit investigation workflow yet.
It also does not use pretrained external vision models or external LLM calls.

### Verification

Focused checks:

- `tests/vision` -> `8 passed`
- `ruff check src/industrial_ai/vision tests/vision` -> `All checks passed`

## 2026-06-02 - Per-Category Vision Evaluation And ResNet Embeddings

### Goal

Measure image anomaly detection by category first, then add a stronger
ResNet-based anomaly detector:

```text
industrial MVTec categories
  -> per-category comparison metrics
  -> ResNet embedding normal profile
  -> per-category ResNet evaluation
```

### Active Categories

The active demo set is now:

- `cable`
- `grid`
- `metal_nut`
- `screw`
- `transistor`

Other MVTec folders are kept under `mvtec_anomaly_detection/To Avoid` and are
ignored by the evaluator.

### What Changed

- Added `src/industrial_ai/vision/evaluate.py`.
- Added `src/industrial_ai/vision/mvtec_resnet.py`.
- Added per-category evaluation tests.
- Added ResNet profile/scoring tests.
- Added `torchvision` to requirements for ResNet18 support.
- Updated demo commands for cable-focused vision examples.

### Comparison Baseline Metrics

Command:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.evaluate comparison
```

Measured on the five active categories:

| Category | Total | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| cable | 150 | 0.6133 | 0.6133 | 1.0000 | 0.7603 |
| grid | 78 | 0.7308 | 0.7308 | 1.0000 | 0.8444 |
| metal_nut | 115 | 0.8087 | 0.8087 | 1.0000 | 0.8942 |
| screw | 160 | 0.7438 | 0.7438 | 1.0000 | 0.8530 |
| transistor | 100 | 0.5100 | 0.4471 | 0.9500 | 0.6080 |
| overall | 603 | 0.6833 | 0.6786 | 0.9950 | 0.8069 |

The comparison baseline is high-recall but over-flags good images. That is why
the ResNet embedding detector is the next candidate for higher accuracy.

### ResNet Embedding Detector

The ResNet detector uses ResNet18 as a feature extractor:

```text
good images -> ResNet18 embeddings -> normal center
test image -> embedding distance from normal center -> anomaly score
```

Train one profile per category:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet train cable \
  --reference-limit 50
```

Predict:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet predict \
  mvtec_anomaly_detection/cable/test/bent_wire/000.png \
  --model-path models/mvtec_resnet_cable.npz
```

Evaluate:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet evaluate \
  --model-path models/mvtec_resnet_cable.npz
```

For highest accuracy, use pretrained ResNet weights. If they are not already
cached locally, the first train command may need network access to download
them. Use `--no-pretrained` only for local smoke testing.

### Verification

Focused checks:

- `tests/vision` -> `14 passed`
- `ruff check src/industrial_ai/vision tests/vision` -> `All checks passed`

## 2026-06-02 - Vision UI Wiring And ResNet Calibration

### Goal

Close the two remaining Tier 0-plus vision gaps:

```text
Streamlit investigation
  -> optional image inspection
  -> vision evidence added to retrieval
  -> RAG/severity/approval continue unchanged

ResNet profile
  -> score category test images
  -> choose best threshold for F1/accuracy/precision/recall
  -> save calibrated threshold
```

### What Changed

- Updated `src/industrial_ai/demo/investigation.py` to accept optional vision
  image input.
- Updated `src/industrial_ai/demo/streamlit_app.py` with visual inspection
  controls.
- Added normalized vision output to investigation results.
- Vision evidence now augments the retrieval query when a defect is detected.
- Added `calibrate` and `train-all` commands to
  `src/industrial_ai/vision/mvtec_resnet.py`.
- Added tests for vision retrieval context and ResNet threshold calibration.

### Streamlit Vision Behavior

The UI now supports:

- enabling visual inspection
- uploading an inspection image
- selecting one of the active industrial MVTec categories
- choosing `auto`, `resnet`, or `comparison`

`auto` uses a saved ResNet profile when one exists under `models/`; otherwise it
falls back to the comparison baseline.

### Severity Policy Update

Visual defects now participate in severity assignment:

```text
Failure probability > 80% AND visual defect detected -> SEV1
Failure probability > 80% AND RAG confidence = high -> SEV1
Failure probability > 50% -> SEV2
Else -> SEV3
```

This means a case like `81% failure probability + detected visual defect` now
requires human approval because it becomes `SEV1`.

### ResNet Calibration Commands

Calibrate one profile:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet calibrate \
  --model-path models/mvtec_resnet_cable.npz \
  --metric f1
```

Train and calibrate all active categories:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet train-all \
  --reference-limit 50 \
  --metric f1
```

### Calibration Snapshot

Cable pretrained ResNet profile after F1 calibration:

- threshold: `0.2772`
- accuracy: `0.7333`
- precision: `0.7241`
- recall: `0.9130`
- F1: `0.8077`

### Verification

Current checks:

- `tests/demo tests/vision` -> `20 passed`
- `ruff check src/industrial_ai/demo src/industrial_ai/vision tests/demo tests/vision`
  -> `All checks passed`

Full suite:

- `72 passed`
- `ruff check src tests` -> `All checks passed`
## 2026-06-02 - Vision Evidence Workflow Integration

### Goal

Integrate vision outputs into the investigation workflow without changing vision
inference internals:

```text
telemetry prediction
  -> optional vision result
  -> normalized evidence objects
  -> RAG query construction
  -> severity policy
  -> approval state
  -> Streamlit combined evidence display
```

### What Changed

- Added normalized `EvidenceItem` objects in
  `src/industrial_ai/demo/investigation.py`.
- Investigation results now include combined telemetry and vision evidence.
- RAG query construction now uses normalized evidence summaries when vision is
  available.
- Streamlit now shows a `Combined Evidence` section.
- Severity policy now enforces:

```text
high telemetry risk + visual defect -> SEV1
high telemetry risk + no visual defect -> SEV2
low telemetry risk + visual defect -> SEV3
low telemetry risk + no visual defect -> Normal
SEV1 -> human approval required
```

### Test Coverage

Added/updated tests for:

- high telemetry + visual defect
- low telemetry + visual defect
- high telemetry + no visual defect
- no vision input
- normalized telemetry/vision evidence objects

### Verification

Full suite:

- `86 passed`
- `ruff check src tests` -> `All checks passed`

## 2026-06-02 - Telemetry-Aware Incident Reranking

### Goal

Improve similar incident retrieval when exact AI4I telemetry inputs are
available while keeping vector retrieval as the first stage.

### What Changed

- Added optional telemetry-aware reranking in
  `src/industrial_ai/incidents/memory.py`.
- Added `TelemetryQuery` and telemetry similarity scoring for:
  - `tool_wear_min`
  - `torque_nm`
  - `rotational_speed_rpm`
  - `air_temperature_k`
  - `process_temperature_k`
- Search results now expose:
  - `vector_score`
  - `telemetry_similarity_score`
  - `combined_score`
- Added `--telemetry-rerank` and telemetry CLI flags to incident search.
- The investigation workflow now passes the current telemetry reading into
  retrieval reranking.
- Streamlit similar-incident cards now show the combined score and score
  components.
- Added synthetic document tests for telemetry reranking.

### CLI

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

### Verification

Focused checks:

- `tests/incidents tests/demo` -> `26 passed`
- `ruff check src/industrial_ai/incidents src/industrial_ai/demo tests/incidents tests/demo`
  -> `All checks passed`

Full suite:

- `86 passed`
- `ruff check src tests` -> `All checks passed`

## 2026-06-02 - Dashboard Policy Management

### Goal

Add policy management visibility without duplicating or editing production
policy logic in the UI.

### What Changed

- Added `SeverityPolicy` metadata in `src/industrial_ai/policy/severity.py`.
- Added policy name, version, and last modified timestamp from the production
  policy source.
- Added a `Policy Management` tab to Streamlit.
- The tab displays active rules, approval requirements, and the latest triggered
  severity decision when an investigation has run.
- Added visible `Edit Policy` action with simulated/read-only editing.
- Added policy and dashboard tests for metadata loading.

### Verification

Focused checks:

- `tests/policy tests/demo` -> `27 passed`
- `ruff check src/industrial_ai/policy src/industrial_ai/demo tests/policy tests/demo`
  -> `All checks passed`

Full suite:

- `84 passed`
- `ruff check src tests` -> `All checks passed`

## 2026-06-02 - Dashboard Severity Rules Visibility

### Goal

Show severity policy rules in the Streamlit dashboard without duplicating rule
logic in the UI.

### What Changed

- Added production rule metadata in `src/industrial_ai/policy/severity.py`.
- Approval requirement checks now use the same policy source.
- Added an expandable `Severity Rules` section to the investigation dashboard.
- The section shows SEV criteria, approval requirements, triggered reason, and
  inputs used.
- Added policy and dashboard tests for rule metadata.

### Verification

Focused checks:

- `tests/policy tests/approvals tests/demo` -> `32 passed`
- `ruff check src/industrial_ai/policy src/industrial_ai/approvals src/industrial_ai/demo tests/policy tests/approvals tests/demo`
  -> `All checks passed`

Full suite:

- `82 passed`
- `ruff check src tests` -> `All checks passed`

## 2026-06-02 - Dashboard Evaluation Visibility

### Goal

Expose evaluation results in the Streamlit dashboard without creating a second
evaluation implementation.

### What Changed

- Added an `Evaluation` tab to `src/industrial_ai/demo/streamlit_app.py`.
- Reused `industrial_ai.evaluation.test_rig.run_rig`.
- Added dashboard helpers for pass rate, key-input formatting, and scenario
  filtering.
- Added `tests/demo/test_streamlit_app.py`.

### Dashboard Output

The Evaluation tab shows:

- data sources used
- number of scenarios
- passed count
- failed count
- pass rate
- scenario table with area, scenario, expected, actual, pass/fail, and key inputs
- filters: `All`, `Passed`, `Failed`
- `Run Evaluation` button

### Verification

Focused checks:

- `tests/demo tests/evaluation` -> `19 passed`
- `ruff check src/industrial_ai/demo src/industrial_ai/evaluation tests/demo tests/evaluation`
  -> `All checks passed`

Full suite:

- `80 passed`
- `ruff check src tests` -> `All checks passed`

## 2026-06-02 - Held-Out Demo Correctness Rig

### Goal

Add a test rig that checks the dashboard story against data not used for
training:

```text
AI4I held-out split
MVTec test images
severity rules
approval status
```

### What Changed

- Added `src/industrial_ai/evaluation/test_rig.py`.
- Added `tests/evaluation/test_test_rig.py`.
- Dashboard `auto` vision no longer silently falls back to comparison when a
  ResNet profile is missing.
- ResNet profiles now store good-reference embeddings and use nearest-neighbor
  embedding distance instead of only distance to the normal center.
- Rebuilt ignored local ResNet profiles under `models/` for the five active
  categories.

### Rig Command

```bash
TORCH_HOME=/tmp/torch-cache PYTHONPATH=src .venv/bin/python -m industrial_ai.evaluation.test_rig \
  --category cable \
  --category grid \
  --category metal_nut \
  --category screw \
  --category transistor
```

### Current Rig Result

Current result:

- total scenarios: `14`
- passed: `12`
- failed: `2`

Passing areas:

- AI4I held-out positive and negative telemetry cases
- cable good/defect image cases
- grid good image case
- metal_nut good/defect image cases
- screw good image case
- transistor good/defect image cases
- severity and approval rules

Failing areas:

- `grid_defect_image`: sampled defect image is missed
- `screw_defect_image`: sampled defect image is missed

The rig confirms the earlier dashboard issue: the old comparison fallback could
over-flag good images. The current ResNet nearest-neighbor profiles fix the
sampled good-image false positives, but grid and screw defect recall still need
model work before those categories should be treated as reliable.

### Verification

Full suite:

- `72 passed`
- `ruff check src tests` -> `All checks passed`

## 2026-06-10 - AMD MI300X LoRA + LLM-as-Judge Evaluation

### Goal

Run the first real AMD MI300X LoRA fine-tuning and LLM-as-Judge evaluation for
the factory multimodal agent fine-tuning track.

### Environment

- AMD MI300X-class `gfx942` GPU
- ROCm 7.0 / HIP 7.0
- BF16 used for LoRA training and generation

### Candidate Model

- Base model: `Qwen/Qwen3-4B-Instruct-2507`

Gemma was initially considered, but `google/gemma-3-4b-it` is gated and blocked
the smoke test without Hugging Face authentication. The AMD workflow defaults
were switched to the non-gated Qwen instruct model.

### LoRA Training

Training completed successfully on the AMD GPU.

| Metric | Value |
|---|---:|
| train_loss | `1.5399` |
| eval_loss | `1.1387` |
| elapsed_seconds | `63.39` |

Adapter path:

```text
data/amd/lora/qwen4b_adapter/
```

### Candidate Generation

Generated base and LoRA responses for the evaluation dataset.

| Metric | Value |
|---|---:|
| eval examples | `10` |
| max_new_tokens | `128` |
| base total latency | `66.902s` |
| base mean latency | `6.690s` |
| LoRA total latency | `77.759s` |
| LoRA mean latency | `7.776s` |

Generation now uses a hackathon-friendly default subset size and writes base
and LoRA outputs incrementally after each aligned example so partial progress is
preserved.

### Judge

- Judge model: `Qwen/Qwen3-14B`
- Endpoint: OpenAI-compatible vLLM at `localhost:8000`
- Judge runtime: `22.538s`
- Records: `20`
- Successes: `20/20`

`openai/gpt-oss-20b` was accessible, but vLLM failed to start it on this ROCm
image because of MXFP4 / `triton_kernels` compatibility:

```text
ModuleNotFoundError: No module named 'triton_kernels.tensor'
```

Qwen 14B was used as the strongest available non-gated fallback judge.

### LLM-as-Judge Results

Hallucination score is lower-is-better. All other metrics are higher-is-better.

| Metric | Base | LoRA | Improvement |
|---|---:|---:|---:|
| Hallucination score | `1.2` | `1.0` | `16.67%` lower is better |
| RCA quality | `3.0` | `4.1` | `36.67%` |
| Actionability | `3.8` | `4.0` | `5.26%` |
| Severity reasoning | `3.0` | `4.1` | `36.67%` |

### Key Interpretation

- LoRA improved all four judged dimensions.
- Largest gains were RCA quality and severity reasoning.
- This provides the fine-tuning track evidence for the hackathon.

### Generated Artifact Paths

```text
data/amd/lora/training_metrics.json
data/amd/lora/training_log.txt
data/evals/base_results.jsonl
data/evals/lora_results.jsonl
data/evals/judge_scores.jsonl
data/evals/summary.csv
data/evals/summary.json
data/evals/llm_judge_report.md
```

These artifacts are generated and should not be committed unless intentionally
selected later for final submission.

## 2026-06-12 - AMD Cloud Recovery Rerun

### Goal

Recover generated AMD Cloud artifacts after starting from a fresh notebook and
make model identities explicit for reviewer auditability.

### Exact Evaluation Identity

| Field | Value |
|---|---|
| Base Model | `Qwen/Qwen3-4B-Instruct-2507` |
| LoRA Model | `Qwen/Qwen3-4B-Instruct-2507` with adapter `data/amd/lora/qwen4b_adapter` |
| Judge Model | `Qwen/Qwen3-14B` |
| Judge Serving | vLLM OpenAI-compatible endpoint at `http://localhost:8000/v1/chat/completions` |
| Hardware | AMD MI300X-class `gfx942` GPU |
| ROCm / HIP | ROCm 7.0 / HIP 7.0 |
| Precision | BF16 for LoRA training, candidate generation, and vLLM judge serving |

The base model is the model that was fine-tuned. The LoRA model is the same
base checkpoint loaded with the generated adapter at
`data/amd/lora/qwen4b_adapter`. The judge model is separate and was used only
to score candidate responses.

### LoRA Training

Training completed successfully on the AMD GPU.

| Metric | Value |
|---|---:|
| train examples | `40` |
| eval examples | `10` |
| train_loss | `0.1870` |
| eval_loss | `0.02746` |
| elapsed_seconds | `290.289` |

### LLM-as-Judge

The Qwen 14B judge endpoint was started with vLLM and reached a healthy
`/v1/models` response before scoring.

| Metric | Value |
|---|---:|
| eval examples | `10` |
| judge records | `20` |
| judge successes | `20/20` |
| full experiment runtime | `502s` |

Hallucination score is lower-is-better. All other metrics are higher-is-better.

| Metric | Base | LoRA | Improvement |
|---|---:|---:|---:|
| Hallucination score | `1.0` | `1.0` | `0.0%` |
| RCA quality | `3.7` | `4.6` | `24.32%` |
| Actionability | `4.4` | `4.4` | `0.0%` |
| Severity reasoning | `3.8` | `4.6` | `21.05%` |

### Generated Artifact Paths

```text
data/amd/lora/training_metrics.json
data/amd/lora/training_log.txt
data/evals/eval_dataset.jsonl
data/evals/base_results.jsonl
data/evals/lora_results.jsonl
data/evals/judge_scores.jsonl
data/evals/summary.csv
data/evals/summary.json
data/evals/llm_judge_report.md
```

Timestamped export created:

```text
exports/amd_recovery_20260612T133257Z.zip
```

The export includes `data/evals/`, `data/amd/lora/training_metrics.json`,
`data/amd/lora/training_log.txt`, and `docs/progress.md`. It excludes Hugging
Face cache, model weights, trainer checkpoints, and safetensors files.

## 2026-06-12 - AMD Full Experiment Rerun Checkpoint

Feature freeze was preserved: no application or Streamlit logic was changed.

### Environment

| Field | Value |
|---|---|
| Base Model | `Qwen/Qwen3-4B-Instruct-2507` |
| LoRA Adapter | `data/amd/lora/qwen4b_adapter/` |
| Judge Model | `Qwen/Qwen3-14B` |
| Judge Serving | vLLM OpenAI-compatible endpoint at `http://localhost:8000/v1/chat/completions` |
| GPU | AMD `gfx942`, 191.984 GiB VRAM reported by PyTorch |
| ROCm / HIP | `7.0.51831-a3e329ad8` |
| Torch | `2.8.0+gitb2fb688` |
| Precision | BF16 |
| LIMIT | `10` |

### LoRA Training

| Metric | Value |
|---|---:|
| train examples | `40` |
| eval examples | `10` |
| train runtime seconds | `288.4736` |
| elapsed seconds | `289.336` |
| train loss | `0.1859703972` |
| eval loss | `0.0289932545` |

### LLM-as-Judge

| Metric | Base | LoRA | Improvement |
|---|---:|---:|---:|
| hallucination_score | `1.0` | `1.0` | `0.0%` |
| rca_quality | `3.7` | `4.2` | `13.51%` |
| actionability | `4.4` | `4.1` | `-6.82%` |
| severity_reasoning | `3.8` | `4.2` | `10.53%` |

Judge records: `20`, successes: `20/20`.

### Hardware Telemetry During LoRA + Candidate/Judge Run

| Metric | Value |
|---|---:|
| wall clock seconds | `493.9645` |
| peak VRAM GiB | `191.3731` |
| average VRAM GiB | `187.8833` |
| peak GPU utilization | `100.0%` |
| average GPU utilization | `75.8534%` |
| peak power W | `749.0` |
| average power W | `330.6336` |
| peak temperature C | `73.0` |
| average temperature C | `46.3793` |

### ROCm Benchmarks

`scripts/amd/rocm_fused_rerank_benchmark.py --modes fp32,fp16,bf16,fp8 --chart`
completed successfully.

| Metric | Value |
|---|---:|
| rows | `72` |
| successful runs | `54` |
| skipped runs | `18` |
| best latency | `0.3306500148 ms` |
| best mode | `fp32` |
| best workload | `batch=8`, `candidates=10000`, `dim=384`, `top_k=10` |

FP8 rows in the fused benchmark were skipped because native FP8 matmul is not
exposed cleanly by this PyTorch build for that script path.

`scripts/amd/rocm_kernel_comparison_benchmark.py` completed successfully.

| Metric | Value |
|---|---:|
| workload | `batch=32`, `candidates=1000000`, `dim=768`, `top_k=10` |
| best implementation | `rocblas_plus_triton_score` |
| best precision | `bf16` |
| best latency | `1.0388850351 ms` |
| throughput candidates/s | `30802253299.70026` |
| speedup vs PyTorch FP32 eager | `6.2041918188x` |
| top-k overlap vs FP32 | `0.996875` |

Kernel comparison also measured a real FP8 path:
`torch_scaled_mm_fp8` at `2.557 ms`, `2.521x` speedup, and `0.956` top-k
overlap. AITER and Composable Kernel full-rerank paths remained unsupported for
this workload on the current image.
