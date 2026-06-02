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

## Streamlit UI

Launch the one-page demo app:

```bash
PYTHONPATH=src .venv/bin/streamlit run src/industrial_ai/demo/streamlit_app.py
```

The page runs the same workflow as the commands below:

```text
telemetry prediction
  -> optional visual inspection
  -> incident retrieval
  -> deterministic RAG answer
  -> severity assignment
  -> approval creation
```

The `Evaluation` tab runs the same held-out rig as the CLI and shows pass/fail
counts, pass rate, scenario details, key inputs, and filters for all/passed/failed
scenarios.

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

With telemetry-aware reranking:

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

## 4. Generate RAG Answer

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.rag.answer \
  "What is the likely cause of a tool wear failure?"
```

With local Ollama synthesis:

```bash
OLLAMA_MODEL=gemma3:4b PYTHONPATH=src .venv/bin/python -m industrial_ai.rag.answer \
  "What is the likely cause of a tool wear failure?" \
  --llm
```

The model name defaults to `gemma3:4b` when `OLLAMA_MODEL` is not set. If
Ollama is unavailable, the command falls back to deterministic RAG unless
`--no-fallback` is passed.

## 5. Assign Severity

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.policy.severity \
  --failure-probability 0.82 \
  --rag-confidence high
```

With a detected visual defect:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.policy.severity \
  --failure-probability 0.82 \
  --rag-confidence medium \
  --visual-defect-detected
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

## 9. Compare MVTec Image Against Good References

Run this first as the transparent baseline:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_compare \
  mvtec_anomaly_detection/cable/test/bent_wire/000.png
```

## 10. Evaluate Comparison Baseline By Category

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.evaluate comparison
```

## 11. Train Deep Learning MVTec Autoencoder

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_autoencoder train cable \
  --epochs 5 \
  --reference-limit 50
```

## 12. Predict With Deep Learning MVTec Autoencoder

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_autoencoder predict \
  mvtec_anomaly_detection/cable/test/bent_wire/000.png \
  --model-path models/mvtec_autoencoder_cable.pt
```

## 13. Train ResNet Embedding Detector

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet train cable \
  --reference-limit 50
```

## 14. Predict With ResNet Embedding Detector

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet predict \
  mvtec_anomaly_detection/cable/test/bent_wire/000.png \
  --model-path models/mvtec_resnet_cable.npz
```

## 15. Calibrate ResNet Threshold

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet calibrate \
  --model-path models/mvtec_resnet_cable.npz \
  --metric f1
```

## 16. Train And Calibrate All ResNet Categories

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet train-all \
  --reference-limit 50 \
  --metric f1
```

## 17. Evaluate ResNet Embedding Detector

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet evaluate \
  --model-path models/mvtec_resnet_cable.npz
```

## 18. Run Held-Out Correctness Rig

```bash
TORCH_HOME=/tmp/torch-cache PYTHONPATH=src .venv/bin/python -m industrial_ai.evaluation.test_rig \
  --category cable \
  --category grid \
  --category metal_nut \
  --category screw \
  --category transistor
```
