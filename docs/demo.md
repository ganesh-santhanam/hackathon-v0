# Demo Commands

Run these commands from the repository root:

```bash
cd /home/aaa/Hackathon
```

## Prerequisites

Generate the local incident corpus:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.generate \
  --source-failure-rows 100
```

Index the corpus into local Qdrant:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.memory index
```

## 1. Train Telemetry

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.telemetry.train
```

## 2. Predict One Risky Machine

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

## 3. Retrieve Similar Incidents

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.memory search \
  "tool wear and torque anomaly" \
  --top-k 3 \
  --score-threshold 0.5
```

## 4. Generate RAG Answer

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.rag.answer \
  "What is the likely cause of a tool wear failure?"
```

## 5. Assign Severity

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.policy.severity \
  --failure-probability 0.82 \
  --rag-confidence high
```

## 6. Create Approval

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.approvals.approval create \
  INCIDENT-001 \
  --severity SEV1
```

## 7. Approve Or Reject

Approve:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.approvals.approval approve INCIDENT-001
```

Reject:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.approvals.approval reject INCIDENT-001
```

Show current approval status:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.approvals.approval show INCIDENT-001
```

## 8. Run Eval Harness

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.evaluation.harness
```

