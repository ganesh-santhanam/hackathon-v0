# Industrial AI Assistant: Detailed Technical README

This is the complete technical reference for the repository. The root `README.md`
is intentionally concise and GitHub-facing; this document preserves setup,
operations, architecture, evaluation, AMD/ROCm, and engineering hardening detail.

## Scope

The project is a local-first industrial incident investigation assistant.

```text
Telemetry
  -> Vision
  -> Incident Memory
  -> Root Cause Analysis
  -> Severity Policy
  -> Human Approval
```

The implementation is a demoable system, not a production deployment. It uses local
files, generated artifacts, local Qdrant, optional local Ollama/vLLM-style endpoints,
deterministic fallbacks, and Streamlit.

## Primary Entry Points

| Area | Entry Point |
| --- | --- |
| Streamlit dashboard | `src/industrial_ai/demo/streamlit_app.py` |
| LangGraph workflow | `src/industrial_ai/demo/graph_workflow.py` |
| Investigation facade | `src/industrial_ai/demo/investigation.py` |
| Telemetry training | `python -m industrial_ai.telemetry.train` |
| Telemetry inference | `python -m industrial_ai.telemetry.predict` |
| Incident corpus generation | `python -m industrial_ai.incidents.generate` |
| Qdrant indexing/search | `python -m industrial_ai.incidents.memory` |
| RAG answer generation | `python -m industrial_ai.rag.answer` |
| Severity policy | `python -m industrial_ai.policy.severity` |
| Approval workflow | `python -m industrial_ai.approvals.approval` |
| Vision comparison | `python -m industrial_ai.vision.mvtec_compare` |
| Vision autoencoder | `python -m industrial_ai.vision.mvtec_autoencoder` |
| Vision ResNet | `python -m industrial_ai.vision.mvtec_resnet` |
| Vision evaluation | `python -m industrial_ai.vision.evaluate` |
| Demo correctness rig | `python -m industrial_ai.evaluation.test_rig` |
| Deterministic harness | `python -m industrial_ai.evaluation.harness` |
| LLM judge | `scripts/run_llm_judge_eval.py` |
| Unified evaluation package | `scripts/evals/run_full_evaluation.py` |
| AMD/ROCm scripts | `scripts/amd/*.py`, `scripts/amd/*.sh` |
| Docker | `Dockerfile`, `docker-compose.yml` |

## Architecture

![Current implemented architecture](diagrams/current_architecture.svg)

The current LangGraph workflow is intentionally linear and inspectable:

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

| Component | Package | Responsibility |
| --- | --- | --- |
| Telemetry | `industrial_ai.telemetry` | AI4I-style feature loading, XGBoost training, failure prediction |
| Vision | `industrial_ai.vision` | MVTec comparison, autoencoder, ResNet anomaly detection, localization |
| Incident memory | `industrial_ai.incidents` | Corpus generation, Qdrant indexing, retrieval, telemetry-aware reranking |
| RAG | `industrial_ai.rag` | Deterministic answer formatting and optional Ollama synthesis |
| Policy | `industrial_ai.policy` | Deterministic severity assignment |
| Approvals | `industrial_ai.approvals` | JSON-backed approval lifecycle |
| Demo | `industrial_ai.demo` | Streamlit UI and workflow facade |
| Evaluation | `industrial_ai.evaluation` | Scenario harnesses, correctness rig, judge utilities |
| Plant stream | `industrial_ai.plant` | Demo event stream simulation |
| Config | `industrial_ai.config` | Settings and logging setup |
| Security | `industrial_ai.security` | Secret redaction and path/upload validation |

Intended dependency direction:

```text
UI/scripts -> demo services -> component packages -> config/security/paths
```

## Persistence Model

| Artifact | Location |
| --- | --- |
| Generated incident corpus | `data/incidents/ai4i_incident_corpus.jsonl` |
| Local Qdrant | `data/qdrant/` |
| Docker Qdrant service data | `data/qdrant_service/` |
| Approval records | `data/approvals/approvals.json` |
| Plant event stream | `data/plant/events.jsonl` |
| Evaluation artifacts | `data/evals/` |
| Benchmark artifacts | `data/benchmarks/` |
| Vision examples | `data/vision_examples/` |
| Local models | `models/` |

Do not commit large datasets, Qdrant indexes, model caches, downloaded model weights,
generated exports, `.env`, or local virtual environments.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Keep `.env` local. Defaults are local-demo friendly.

## Make Commands

| Task | Command |
| --- | --- |
| Install dependencies | `make install` |
| Run tests | `make test` |
| Run lint | `make lint` |
| Format | `make format` |
| Run Streamlit | `make run-ui` |
| Run deterministic eval | `make eval` |
| Validate Docker Compose | `make docker-config` |
| Start Docker Compose | `make docker-up` |
| Health check | `make health` |

## Data Preparation

Train the baseline telemetry model:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.telemetry.train
```

Generate the local incident corpus:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.generate \
  --source-failure-rows 100
```

Index incidents into local Qdrant:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.memory index
```

## Streamlit Demo

```bash
PYTHONPATH=src .venv/bin/streamlit run src/industrial_ai/demo/streamlit_app.py
```

The dashboard supports:

| Tab / Area | Behavior |
| --- | --- |
| Investigation | Manual telemetry input, optional vision image, retrieval, RCA, severity, approval |
| Demo controls | Injected tool wear, power, cooling, visual defect, and multi-modal scenarios |
| Agent trace | Visible LangGraph-style node trace when enabled |
| Similar incidents | Match reasons, telemetry comparison, scores, evidence |
| Policy management | Active deterministic rules and current triggered decision |
| Evaluation | Held-out rig summary and scenario table |
| Plant stream | Local JSONL event replay and trigger simulation |

## Telemetry Pipeline

Train:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.telemetry.train
```

Predict:

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

The model output includes failure probability, risk level, top feature importances,
and deterministic evidence strings such as high tool wear, torque anomaly, and
rotational speed anomaly.

## Vision Pipeline

Run comparison baseline:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_compare \
  mvtec_anomaly_detection/cable/test/bent_wire/000.png
```

Evaluate comparison baseline:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.evaluate comparison
```

Train autoencoder:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_autoencoder train cable \
  --epochs 5 \
  --reference-limit 50
```

Predict with autoencoder:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_autoencoder predict \
  mvtec_anomaly_detection/cable/test/bent_wire/000.png \
  --model-path models/mvtec_autoencoder_cable.pt
```

Train ResNet embedding detector:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.vision.mvtec_resnet train cable \
  --reference-limit 50
```

Calibrate threshold:

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

## Incident Memory And Retrieval

Search similar incidents:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.incidents.memory search \
  "tool wear and torque anomaly" \
  --top-k 3 \
  --score-threshold 0.5
```

Search with telemetry-aware reranking:

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

Current retrieval characteristics:

| Retrieval Capability | Detail |
| --- | --- |
| Corpus | `300` incident documents from `100` AI4I failure rows |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector size | `384` |
| Store | Local Qdrant |
| Reranking | Vector score plus telemetry similarity |
| Explainability | Match reasons and telemetry comparison rows |
| Fallback | No-evidence response when threshold filters all results |

## Why Retrieval Matters More Than Fine-Tuning

Fine-tuning adapts style, vocabulary, response shape, RCA phrasing, and severity
explanation style. Retrieval preserves current evidence, recent incidents, plant
details, maintenance history, and changing operating conditions.

| Decision | Chosen Path | Reason |
| --- | --- | --- |
| Operational facts | Retrieval | Facts change and need auditability |
| Response quality | LoRA | Domain adaptation improves phrasing and reasoning style |
| Severity | Deterministic policy | Governance decisions must be testable |
| RCA grounding | Retrieved evidence | Reduces stale model-memory risk |

The model can write better explanations after adaptation, but retrieved incidents
remain the factual authority.

## RAG And Local LLM Mode

Deterministic answer:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.rag.answer \
  "What is the likely cause of a tool wear failure?"
```

Local Ollama synthesis:

```bash
OLLAMA_MODEL=gemma3:4b PYTHONPATH=src .venv/bin/python -m industrial_ai.rag.answer \
  "What is the likely cause of a tool wear failure?" \
  --llm
```

`OLLAMA_MODEL` defaults to `gemma3:4b`. `OLLAMA_BASE_URL` defaults to
`http://localhost:11434`. `OLLAMA_GENERATE_URL` can override the generate endpoint.

If Ollama is unavailable, deterministic fallback remains active unless
`--no-fallback` is passed.

## Severity Policy

Assign severity:

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

Current policy:

| Rule | Severity | Approval Required |
| --- | --- | --- |
| Failure probability > 80% and visual defect detected | SEV1 | Yes |
| Failure probability > 80% and RAG confidence high | SEV1 | Yes |
| Failure probability > 50% | SEV2 | No |
| Failure probability <= 50% | SEV3 | No |

## Why Human Approval Matters

Industrial AI should support operators, not bypass operational control. SEV1
recommendations can affect safety, production quality, downtime, and maintenance
cost. The demo therefore creates explicit approval records instead of treating model
output as an action.

Create approval:

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

## Evaluation

Fast validation:

```bash
make test
make eval
```

Deterministic harness:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.evaluation.harness
```

Demo correctness rig:

```bash
PYTHONPATH=src .venv/bin/python -m industrial_ai.evaluation.test_rig --category cable
```

Unified evaluation package:

```bash
PYTHONPATH=src .venv/bin/python scripts/evals/run_full_evaluation.py
```

LLM judge help:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_llm_judge_eval.py --help
```

## Supported Headline Metrics

| Area | Result |
| --- | ---: |
| Historical telemetry ROC AUC | `0.97` |
| Historical telemetry average precision | `0.70` |
| Incident corpus | `300` documents |
| Embedding size | `384` |
| Severity scenario accuracy | `12/12` |
| LLM judge records | `20` |
| LoRA RCA quality | `4.2` vs base `3.7` |
| LoRA severity reasoning | `4.2` vs base `3.8` |
| Hallucination score | `1.0` base and LoRA |
| Best ROCm speedup | `6.2x` |
| Best ROCm latency | `1.039 ms` |
| Evaluation package completeness | `70.5%` |
| Latest hardening validation | `124` tests passed |

Metric caveat: telemetry ROC AUC and average precision are historical development
metrics retained in the final technical report. The packaged full-run evaluation marks
missing artifacts explicitly instead of fabricating values.

## AMD Cloud And LoRA

AMD Cloud enabled:

| Capability | Detail |
| --- | --- |
| BF16 LoRA | Qwen3-4B adapter path |
| LLM-as-judge | Qwen3-14B through OpenAI-compatible endpoint |
| Hardware telemetry | MI300X-class runtime metrics |
| ROCm benchmark | rocBLAS and Triton reranking experiments |

Packaged LoRA metrics:

| Metric | Value |
| --- | ---: |
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| LoRA model | `Qwen/Qwen3-4B-Instruct-2507+data/amd/lora/qwen4b_adapter` |
| Adapter size | `141.22 MB` |
| Train loss | `0.19` |
| Eval loss | `0.03` |
| Train runtime | `288.47 s` |
| GPU hours | `0.08` |
| Examples | `10` |
| Successes | `10` |

Judge results:

| Metric | Base | LoRA | Improvement |
| --- | ---: | ---: | ---: |
| hallucination_score | `1.0` | `1.0` | `0.0` |
| rca_quality | `3.7` | `4.2` | `13.51` |
| actionability | `4.4` | `4.1` | `-6.82` |
| severity_reasoning | `3.8` | `4.2` | `10.53` |

![LoRA judge score change](../data/evals/full_run/charts/report_v2_lora_improvement.svg)

## ROCm Optimization

The ROCm benchmark targets telemetry-aware incident memory reranking, not LLM
inference. It accelerates the stage that scores candidate incidents before generation.

![ROCm optimization ladder](diagrams/rocm_optimization_ladder.svg)

Best kernel comparison result:

| Field | Value |
| --- | ---: |
| Implementation | `rocblas_plus_triton_score` |
| Precision | `bf16` |
| Latency | `1.039 ms` |
| Candidates/sec | `30,802,253,299.700` |
| Speedup | `6.204x` |
| Top-k overlap | `0.997` |
| Peak VRAM | `7.897 GB` |

Why it won:

| Stage | Approach |
| --- | --- |
| Similarity | rocBLAS GEMM |
| Telemetry penalty | Triton fused scoring |
| Score combine | Triton fused scoring |
| Top-k | PyTorch top-k |

FP8 was not selected because the kernel comparison had lower top-k overlap and the
fused sweep skipped FP8 where native FP8 matmul was not exposed cleanly by the PyTorch
build.

## Docker

Validate Compose:

```bash
make docker-config
```

Start Streamlit and Qdrant:

```bash
make docker-up
```

The Docker image does not bake in datasets, model weights, Qdrant indexes, or caches.
The Compose file mounts local `data/`, `models/`, `mvtec_anomaly_detection/`, and
`ai4i_dataset/` paths.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `INDUSTRIAL_AI_ENV` | `demo` | Environment label |
| `INDUSTRIAL_AI_PRODUCTION` | `false` | Enables stricter config validation |
| `INDUSTRIAL_AI_LOG_LEVEL` | `INFO` | Logging level |
| `OLLAMA_MODEL` | `gemma3:4b` | Local model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_GENERATE_URL` | derived | Full generate endpoint override |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant service URL for Docker/service contexts |
| `OPENAI_BASE_URL` | `http://localhost:8000/v1` | Optional local OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | unset | Optional local endpoint token |

## Security And Hardening

See `../SECURITY.md` for the full policy.

Current hardening:

| Area | Status |
| --- | --- |
| `.env` handling | `.env.example` tracked, `.env` ignored |
| Redaction | Secret/token redaction helper added |
| Upload paths | Shared safe upload path validation |
| Production config | Placeholder secret validation |
| Model artifacts | Documented as trusted local artifacts only |
| Tests | Config, redaction, validation tests added |

Remaining risks:

| Risk | Current Position |
| --- | --- |
| Streamlit auth/RBAC | Not implemented |
| Approval audit guarantees | JSON demo persistence only |
| Untrusted model files | Not supported |
| Dependency lock/CVE scanning | Deferred |
| Docker hardening | Demo-level |

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Missing telemetry model | Run `python -m industrial_ai.telemetry.train` |
| Missing incident corpus | Run `python -m industrial_ai.incidents.generate --source-failure-rows 100` |
| Missing Qdrant collection | Run `python -m industrial_ai.incidents.memory index` |
| Ollama unavailable | Use deterministic mode or start Ollama and set `OLLAMA_MODEL` |
| Missing ResNet profile | Run `mvtec_resnet train` and `calibrate` |
| Missing MVTec images | Use telemetry-only scenarios or restore local dataset |
| Docker command missing | Install Docker before using `make docker-config` or `make docker-up` |

## Production Readiness Validation

Latest recorded validation from `REFACTOR_SUMMARY.md`:

| Command | Result |
| --- | --- |
| `PYTHONPATH=src .venv/bin/pytest -q` | `124 passed in 3.78s` |
| `PYTHONPATH=src .venv/bin/python -m industrial_ai.evaluation.harness` | `12 passed, 0 failed` |
| `PYTHONPATH=src .venv/bin/python -m ruff check .` | passed |
| Streamlit/core import smoke check | `imports_ok=8` |
| `docker compose config` | blocked locally because Docker was unavailable |

## Repository Structure

```text
src/industrial_ai/
  approvals/      JSON-backed approval lifecycle
  config/         environment-backed settings and logging setup
  demo/           Streamlit app and investigation workflow facade
  evaluation/     deterministic harnesses and LLM judge utilities
  incidents/      incident corpus generation and Qdrant retrieval
  plant/          simulated plant event stream
  policy/         severity policy engine
  rag/            deterministic and local Ollama RAG answer generation
  security/       redaction and path validation helpers
  telemetry/      AI4I training and prediction
  vision/         MVTec anomaly detection and localization
tests/
scripts/
docs/
data/
models/
```

## Deferred Production Hardening

| Deferred Work | Reason |
| --- | --- |
| Domain package consolidation | Avoid destabilizing demo behavior before broader integration tests |
| Streamlit component split | Useful, but not needed for current demo reliability |
| Durable approval database | JSON is the current persistence policy |
| Auth/RBAC | Needed for production, not local hackathon demo |
| OpenTelemetry/LangSmith export | Roadmap item |
| Dependency lock and CVE scanner | Should be added before external release |
| Tamper-resistant audit logs | Needed for real industrial deployment |

## Roadmap

| Horizon | Work |
| --- | --- |
| 30 days | SHAP-style telemetry attribution, labeled retrieval relevance set, persisted LangGraph traces, retrieval latency measurement |
| 90 days | LangSmith/OpenTelemetry trace export, larger judge set, citation accuracy, token usage and TTFT capture |
| 180 days | Streaming ingestion pilot, shared vector memory, model gateway, feature-store consistency, audit-grade approvals |

## Reference Documents

| Document | Purpose |
| --- | --- |
| `docs/final_technical_report_final.md` | Judge-facing technical report and artifact-backed metrics |
| `docs/ARCHITECTURE.md` | Current architecture and dependency direction |
| `docs/EVALUATION.md` | Evaluation commands and metric policy |
| `docs/OPERATIONS.md` | Setup, run, health, Docker, troubleshooting |
| `SECURITY.md` | Threat model and hardening checklist |
| `REFACTOR_PLAN.md` | Production-readiness refactor plan |
| `REFACTOR_SUMMARY.md` | Hardening changes and validation results |
