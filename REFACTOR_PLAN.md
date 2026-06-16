# Refactor Plan

## Current Architecture Summary

The repository is a local-first industrial AI demo with a compact Python package under
`src/industrial_ai` and command/script entry points around it.

Primary entry points:

- Streamlit dashboard: `src/industrial_ai/demo/streamlit_app.py`
- LangGraph workflow: `src/industrial_ai/demo/graph_workflow.py`
- Investigation service facade: `src/industrial_ai/demo/investigation.py`
- Telemetry training: `python -m industrial_ai.telemetry.train`
- Telemetry inference: `python -m industrial_ai.telemetry.predict`
- Incident generation: `python -m industrial_ai.incidents.generate`
- Qdrant indexing/search: `python -m industrial_ai.incidents.memory`
- RAG answer generation: `python -m industrial_ai.rag.answer`
- Severity policy CLI: `python -m industrial_ai.policy.severity`
- Approval CLI: `python -m industrial_ai.approvals.approval`
- Vision CLIs: `python -m industrial_ai.vision.mvtec_compare`,
  `python -m industrial_ai.vision.mvtec_autoencoder`,
  `python -m industrial_ai.vision.mvtec_resnet`, and
  `python -m industrial_ai.vision.evaluate`
- Demo correctness rig: `python -m industrial_ai.evaluation.test_rig`
- Deterministic severity harness: `python -m industrial_ai.evaluation.harness`
- LLM judge harness: `scripts/run_llm_judge_eval.py` and
  `src/industrial_ai/evaluation/llm_judge.py`
- AMD/ROCm scripts: `scripts/amd/*.py` and `scripts/amd/*.sh`
- Unified evaluation packaging: `scripts/evals/*.py`
- Docker: `Dockerfile` and `docker-compose.yml`

Current flow:

```text
TelemetryReading
  -> XGBoost telemetry model
  -> optional vision detector/localization
  -> Qdrant incident retrieval with telemetry-aware reranking
  -> deterministic or local Ollama RAG answer
  -> severity policy
  -> JSON approval record
  -> Streamlit/dashboard/evaluation presentation
```

The code is already divided by component, but shared production concerns such as config,
logging, path validation, redaction, and operational docs are thin.

## Main Code Smells

- Configuration is scattered across module constants and direct `os.environ` reads.
- `src/industrial_ai/paths.py` provides paths but not path validation or environment override handling.
- Library code uses `print()` only at CLI boundaries, which is acceptable, but there is no shared logging setup.
- Multiple modules parse JSON directly without a shared helper for error messages and schema expectations.
- Some exception handling around optional vision explainability is broad and appends raw exception text to user-facing evidence.
- Boundary models are dataclasses, but they are spread across feature modules instead of collected under a domain layer.
- Streamlit contains UI state, upload validation, display formatting, and workflow orchestration in one large file.
- Dependency versions are unconstrained in `requirements.txt`.
- Docker Compose stores Qdrant data under `data/qdrant_service`, while package defaults use `data/qdrant`.
- Generated artifacts and local model files exist in the working tree; `.gitignore` mostly covers them but can be tightened.

## Security Risks

- Local model loading uses `joblib` and `.npz` artifacts. This is acceptable for trusted local demo artifacts, but it must be documented as unsafe for untrusted files.
- Ollama/OpenAI-compatible endpoints are local by default, but endpoint URLs and errors are shown in observability surfaces. Errors should be redacted before display/logging.
- No central secret redaction helper exists.
- No `.env.example` documents allowed configuration values.
- No `SECURITY.md` explains the local demo threat model, secret handling, or production hardening gaps.
- Uploaded vision filenames are sanitized and suffix-filtered, but a shared path validation helper would make the behavior reusable and testable.
- Secret scanning is manual only; docs should include reproducible commands.

## Proposed Target Structure

Use the requested architecture as a direction without destabilizing the hackathon demo.
The immediate target is incremental:

```text
src/industrial_ai/
  config/
    settings.py
    logging.py
  security/
    secrets.py
    validation.py
  utils/
    io.py
  existing component packages...
```

Defer broad moves such as relocating all domain dataclasses or moving Streamlit into
`industrial_ai/app` until after the demo path has stronger tests. The current component
package layout is serviceable and easier to preserve safely.

## Refactor Phases

### Phase 1: Inventory and plan

Completed by this document.

### Phase 2: Safety net

- Run the existing test suite or targeted subsets if dependencies/data make the full suite slow.
- Add smoke tests for config, redaction, path validation, and core importability if gaps are found.

### Phase 3: Config, paths, logging

- Add `industrial_ai.config.settings` for local-demo-friendly environment handling.
- Add `industrial_ai.config.logging` for shared logging setup.
- Add `industrial_ai.security.secrets` for redaction.
- Add `industrial_ai.security.validation` for safe path and upload validation.
- Keep existing constants in `paths.py` to avoid breaking imports, but back them with helper functions where practical.

### Phase 4: Domain models

- Do not move existing dataclasses yet.
- Add only missing typed boundary models where they reduce raw dict passing.
- Document a future consolidation path under deferred hardening.

### Phase 5: Service boundaries

- Keep `demo/investigation.py` as the workflow facade.
- Keep Streamlit behavior intact.
- Extract only safe utility behavior from Streamlit, such as upload validation, when backed by tests.

### Phase 6: DRY and consistency

- Reuse redaction and validation helpers in RAG/LLM observability and UI upload handling.
- Avoid introducing large abstractions for tiny deterministic functions.

### Phase 7: Security hardening

- Tighten `.gitignore` and `.dockerignore`.
- Add `.env.example`.
- Add `SECURITY.md`.
- Document trusted-local-artifact assumptions for `joblib`, `.pt`, and `.npz` loading.
- Add tests for redaction and path traversal prevention.

### Phase 8: Final validation

Run and record:

- `pytest -q` or targeted subsets if external datasets/models are missing.
- `python -m industrial_ai.evaluation.harness`
- Streamlit import/smoke check.
- Docker Compose config validation if Docker is available.
- `ruff check` if feasible.

## Files Likely To Change

- `REFACTOR_PLAN.md`
- `.gitignore`
- `.dockerignore`
- `.env.example`
- `README.md`
- `SECURITY.md`
- `docs/ARCHITECTURE.md`
- `docs/OPERATIONS.md`
- `docs/EVALUATION.md`
- `Makefile`
- `src/industrial_ai/config/__init__.py`
- `src/industrial_ai/config/settings.py`
- `src/industrial_ai/config/logging.py`
- `src/industrial_ai/security/__init__.py`
- `src/industrial_ai/security/secrets.py`
- `src/industrial_ai/security/validation.py`
- `src/industrial_ai/rag/answer.py`
- `src/industrial_ai/demo/streamlit_app.py`
- Targeted tests under `tests/security/`, `tests/config/`, and existing component test folders
- `REFACTOR_SUMMARY.md`

## Test Strategy

- Keep existing behavior tests as the main regression guard.
- Add small deterministic tests for new helpers:
  - secret redaction
  - placeholder secret validation
  - safe filename/path validation
  - settings defaults and environment overrides
- Add or preserve regression coverage for:
  - severity decisions
  - approval lifecycle
  - retrieval thresholding/reranking
  - evaluation scenarios
  - Streamlit helper functions
- Avoid full training or large dataset tests unless explicitly needed.
- Prefer synthetic/tempfile tests for persistence and validation behavior.

## Deferred Production Hardening

- Move dataclasses into a formal `domain` package after the demo has stable integration tests.
- Split Streamlit into small component modules after behavior is locked.
- Add OpenTelemetry or structured JSON logs only if runtime requirements justify it.
- Add durable database persistence only when explicitly requested; current policy is JSON/local Qdrant.
- Add dependency lock files and vulnerability scanning after package versions are agreed.
