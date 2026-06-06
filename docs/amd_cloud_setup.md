# AMD Cloud Setup

This guide keeps the demo local-first while making it easier to launch on an AMD Cloud VM.

## 1. Clone Repo

```bash
git clone <repo-url>
cd Hackathon
```

## 2. Build Image

```bash
docker compose build streamlit
```

The image installs Python dependencies and project source only. It does not bake in raw datasets, MVTec data, model weights, Qdrant indexes, Ollama models, or `.venv`.

## 3. Prepare Demo Artifacts

Create local directories and generate the incident corpus:

```bash
PYTHON_BIN=.venv/bin/python scripts/demo_setup.sh
```

Inside the container, use:

```bash
docker compose run --rm streamlit bash scripts/demo_setup.sh
```

Required local artifacts still need to be present before richer demos:

- `ai4i_dataset/ai4i2020.csv` for telemetry training/corpus generation
- `models/telemetry_model.joblib` for prediction
- optional `mvtec_anomaly_detection/` images for visual inspection
- optional calibrated ResNet profiles under `models/`

## 4. Index Qdrant

The current app uses the project-local Qdrant path at `data/qdrant/`, mounted into the Streamlit container. The setup script runs:

```bash
PYTHONPATH=src python -m industrial_ai.incidents.memory index
```

The compose file also starts a Qdrant service at `http://qdrant:6333` and exposes it on host port `6333`. `QDRANT_URL` is provided for future or manual service-backed workflows; the existing app path remains local-file Qdrant to preserve the non-Docker workflow.

## 5. Start Dashboard

```bash
docker compose up streamlit
```

Open:

```text
http://localhost:8501
```

The Streamlit service mounts:

- `./data:/app/data`
- `./models:/app/models`
- `./mvtec_anomaly_detection:/app/mvtec_anomaly_detection:ro`
- `./ai4i_dataset:/app/ai4i_dataset:ro`

Runtime environment variables:

- `OLLAMA_BASE_URL`, default `http://host.docker.internal:11434`
- `OLLAMA_MODEL`, default `gemma3:4b`
- `QDRANT_URL`, default `http://qdrant:6333`

## 6. Troubleshooting

If the dashboard starts but investigations fail, confirm the telemetry model exists:

```bash
ls models/telemetry_model.joblib
```

If incident retrieval returns no results, rebuild the local index:

```bash
docker compose run --rm streamlit bash scripts/demo_setup.sh
```

If Ollama is unavailable, the RAG path should fall back to deterministic answers. To point at another Ollama host:

```bash
OLLAMA_BASE_URL=http://<host>:11434 OLLAMA_MODEL=gemma3:4b docker compose up streamlit
```

If visual inspection fails for demo scenarios, check that MVTec sample data and any required calibrated profiles exist under the mounted paths.

If Docker cannot mount missing local directories, create them first:

```bash
mkdir -p data models ai4i_dataset mvtec_anomaly_detection
```
