# Security Policy

## Supported Threat Model

This repository is a local-first hackathon demo. It is designed for trusted operators
running local files, local models, local Qdrant, and optional local Ollama or
OpenAI-compatible endpoints.

The current code is not hardened for hostile multi-user deployments, untrusted model
artifacts, arbitrary public uploads, or internet-facing service exposure.

## Local Demo Assumptions

- AI4I, MVTec, generated incident corpora, Qdrant indexes, and model artifacts are local.
- `joblib`, `.pt`, and `.npz` model artifacts are trusted local artifacts only.
- Streamlit is intended for a trusted demo network, not unauthenticated public access.
- LLM calls should target local Ollama or a controlled OpenAI-compatible endpoint.
- Deterministic fallback must remain available when LLM calls fail.

## Secret Handling

- Do not commit `.env`, API keys, tokens, cloud credentials, model registry tokens, or private endpoints.
- Use `.env.example` to document variables without real secrets.
- User-facing error display should use the redaction helper in `industrial_ai.security.secrets`.
- Production mode validates obvious placeholder secret values via `AppSettings.validate_for_runtime()`.

## Known Limitations

- There is no authentication or RBAC in the Streamlit demo.
- JSON approval records are local files, not tamper-resistant audit logs.
- Local model artifact loading is unsafe for untrusted files.
- The project does not include a dependency lock file or automated CVE scanner.
- Docker Compose is for demo packaging and does not include TLS, auth, or hardened container policies.

## Secret Checks

Run a lightweight manual scan:

```bash
rg -n "(API[_-]?KEY|TOKEN|SECRET|PASSWORD|Bearer|sk-[A-Za-z0-9]|AKIA|hf_)" \
  -g '!**/.git/**' -g '!**/.venv/**' -g '!**/__pycache__/**'
```

For a stronger pass, install and run a dedicated scanner such as `gitleaks` or
`detect-secrets` before publishing.

## Reporting Issues

For hackathon use, report issues directly to the repository owner or project team.
Include the affected file, reproduction steps, and whether any local secret or private
artifact may have been exposed.

## Production Hardening Checklist

- Add authentication and authorization around the UI/API.
- Move approvals and audit logs to durable, access-controlled storage if persistence requirements change.
- Validate and scan all uploaded files before processing.
- Sign or checksum trusted model artifacts.
- Add dependency pinning and vulnerability scanning.
- Run containers as non-root and add resource limits.
- Add structured logs, metrics, and traces with redaction.
- Protect all model and vector-store endpoints with network policy and credentials.
