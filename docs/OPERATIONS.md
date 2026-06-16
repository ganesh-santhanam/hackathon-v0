# Operations

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Keep `.env` local. Do not commit it.

## Train And Prepare Demo Data

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.telemetry.train
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.generate --source-failure-rows 100
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.memory index
```

## Run The UI

```bash
make run-ui
```

Equivalent command:

```bash
PYTHONPATH=src .venv/bin/streamlit run src/industrial_ai/demo/streamlit_app.py
```

## Health Check

```bash
make health
```

This runs the deterministic evaluation harness. It does not require external LLM calls.

## Docker

Validate Compose:

```bash
make docker-config
```

Start Streamlit and Qdrant:

```bash
make docker-up
```

The Compose file mounts local `data/`, `models/`, AI4I, and MVTec paths. Model weights
and datasets are not baked into the image.

## Troubleshooting

- Missing telemetry model: run `python -m industrial_ai.telemetry.train`.
- Missing Qdrant collection: run `python -m industrial_ai.incidents.memory index`.
- Ollama unavailable: use deterministic mode, or start Ollama and set `OLLAMA_MODEL`.
- Missing ResNet profile: run the relevant `industrial_ai.vision.mvtec_resnet train` and
  `calibrate` commands.
- Missing MVTec images: use telemetry-only scenarios or restore the local dataset directory.

## Demo Mode Vs Production Mode

Demo mode is the default. It favors local defaults and deterministic fallback.

Production mode is enabled with:

```bash
INDUSTRIAL_AI_PRODUCTION=true
```

Production mode currently adds configuration validation, but it does not add auth,
RBAC, tamper-resistant audit storage, or hardened network controls. See `SECURITY.md`.
