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
