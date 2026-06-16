# Architecture

The Industrial AI Investigation Assistant is organized as local Python components with
a Streamlit UI and a Docker Compose wrapper for demo deployment.

## Runtime Flow

```text
TelemetryReading
  -> telemetry prediction
  -> optional visual inspection
  -> incident retrieval from Qdrant
  -> deterministic or local Ollama RAG answer
  -> severity policy
  -> JSON approval record
  -> dashboard/evaluation output
```

The LangGraph path keeps the visible agent trace:

```text
START
  -> telemetry_agent
  -> vision_agent
  -> memory_agent
  -> rag_agent
  -> severity_agent
  -> approval_agent
  -> END
```

## Package Map

- `industrial_ai.telemetry`: AI4I feature loading, training, and failure prediction.
- `industrial_ai.vision`: MVTec comparison, autoencoder, ResNet, and localization.
- `industrial_ai.incidents`: generated incident corpus and local Qdrant retrieval.
- `industrial_ai.rag`: deterministic answer synthesis and local Ollama integration.
- `industrial_ai.policy`: deterministic severity policy.
- `industrial_ai.approvals`: JSON-backed approval lifecycle.
- `industrial_ai.demo`: Streamlit UI and investigation orchestration facade.
- `industrial_ai.evaluation`: deterministic scenarios, demo rig, and LLM judge utilities.
- `industrial_ai.plant`: simulated JSONL plant event stream.
- `industrial_ai.config`: environment-backed settings and logging setup.
- `industrial_ai.security`: redaction and path/upload validation helpers.

## Dependency Direction

The intended dependency direction is:

```text
UI/scripts -> demo services -> component packages -> config/security/paths
```

Core business logic should stay out of Streamlit. Streamlit should prepare user input,
call the workflow facade, and render results.

## Persistence

Current persistence remains local-first:

- generated JSON/JSONL artifacts under `data/`
- local Qdrant under `data/qdrant/`
- JSON approval records under `data/approvals/`
- trusted local model artifacts under `models/`

SQLite, Postgres, cloud object stores, and streaming infrastructure are intentionally
not implemented for this demo.

## Deferred Production Hardening

- Consolidate dataclasses into a dedicated domain package.
- Split Streamlit into smaller component modules.
- Add API/service boundaries only when a non-UI client needs them.
- Add durable auth, audit, and persistence layers only for a real multi-user deployment.
