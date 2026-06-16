# AGENTS.md

## Project

Industrial AI Investigation Assistant

A local-first hackathon project combining:

* AI4I predictive maintenance telemetry
* MVTec visual inspection
* Qdrant incident memory
* Gemma/Ollama RAG
* Severity policy engine
* Human approval workflow
* LangGraph orchestration
* Event stream simulation
* Streamlit dashboard

The goal is a demoable industrial operations copilot, not a production deployment.

---

## Current Architecture

Telemetry
→ Telemetry Agent (XGBoost)

Vision
→ Vision Agent (Comparison / Autoencoder / ResNet)

Incident Memory
→ Qdrant Retrieval
→ Telemetry-aware reranking

Investigation
→ RAG Answer Generation
→ Severity Assignment
→ Approval Workflow

UI
→ Streamlit Dashboard

Orchestration
→ LangGraph

Simulation
→ Event Stream
→ Demo Fault Injection

---

## Development Principles

* Prefer simple implementations.
* Prefer demo value over technical perfection.
* Avoid over-engineering.
* Keep changes small and isolated.
* Reuse existing modules whenever possible.
* Do not duplicate business logic in Streamlit.
* Keep deterministic fallbacks operational.

---

## Explicit Non-Goals

Do not add unless explicitly requested:

* Kafka
* Spark
* Flink
* Snowflake
* Kubernetes
* Microservices
* Postgres
* Distributed infrastructure
* Cloud services
* Audio processing
* Full reproducibility workflows

These may appear in architecture diagrams but should not be implemented.

---

## Persistence

Current persistence approach:

* JSON files
* Generated artifacts
* Local Qdrant

Do not add SQLite unless explicitly requested.

Do not add databases for demo-only functionality.

---

## LLM Policy

Current approach:

* Ollama
* Gemma
* Deterministic fallback

Requirements:

* Fallback must remain functional.
* LLM failures must not break investigations.
* Show runtime observability.
* Favor reliability over prompt complexity.

---

## Vision Policy

Current vision stack:

* Comparison detector
* Autoencoder detector
* ResNet detector
* Defect localization

Goals:

* Clear visual explainability
* Bounding boxes
* Heatmaps

Do not spend significant time optimizing benchmark accuracy unless specifically requested.

---

## LangGraph Policy

LangGraph is used for orchestration.

Current graph:

START
→ telemetry_agent
→ vision_agent
→ memory_agent
→ rag_agent
→ severity_agent
→ approval_agent
→ END

Maintain visible agent traces.

Do not redesign the graph unless explicitly requested.

---

## Current Priorities

Highest priority:

1. Similar incident explainability
2. AMD developer Cloud packaging
3. ROCm benchmarking and harness
4. Stream/investigation UX improvements

Future:

5. LoRA on AMD MI300

---

## Do Not Prioritize

* Evaluation tuning
* Vision tuning
* Local Ollama GPU debugging
* Policy tuning
* Additional datasets
* Kafka/Flink/Spark implementation
* Full reproducibility workflows


---

## Dataset Handling

Do not commit:

* AI4I raw datasets
* Full MVTec datasets
* Qdrant indexes
* Model caches
* Hugging Face caches
* Ollama model files
* .venv

Prefer:

* Demo samples
* Generated corpora
* Small example assets

---

## Testing

Prefer targeted tests.

Avoid running the full suite unless necessary.

Examples:

pytest tests/incidents/
pytest tests/vision/
pytest tests/demo/

Run full suite only before major checkpoints.

---

## Output Expectations

Before implementation:

* State which files will be modified.

After implementation:

* Brief summary only.
* Avoid long logs.
* Report pass/fail counts.
* Include commands used for verification.

Do not paste large JSON outputs unless needed for debugging.

---

## Feature Workflow

For each feature:

1. Inspect only relevant files.
2. Propose affected files.
3. Implement.
4. Run targeted tests.
5. Summarize.
6. Stop.

Avoid broad repository exploration when unnecessary.
