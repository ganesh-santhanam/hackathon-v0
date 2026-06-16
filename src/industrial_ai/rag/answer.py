import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from industrial_ai.config.settings import DEFAULT_OLLAMA_MODEL as SETTINGS_DEFAULT_OLLAMA_MODEL, load_settings
from industrial_ai.incidents.memory import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SCORE_THRESHOLD,
    NO_RELEVANT_INCIDENTS_MESSAGE,
    SearchResult,
    load_embedder,
    retrieve_incidents,
)
from industrial_ai.paths import QDRANT_DATA_DIR
from industrial_ai.security.secrets import redact_text

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = SETTINGS_DEFAULT_OLLAMA_MODEL
MAX_CONTEXT_CHARS = 6000


@dataclass(frozen=True)
class SupportingIncident:
    document_id: str
    document_type: str
    machine_id: str
    title: str
    score: float


@dataclass(frozen=True)
class RagMetadata:
    rag_mode: str
    llm_provider: str | None = None
    llm_model: str | None = None
    endpoint_url: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    latency_ms: int | None = None
    raw_error: str | None = None


def deterministic_metadata() -> RagMetadata:
    return RagMetadata(rag_mode="deterministic")


def ollama_metadata(
    model_name: str,
    endpoint_url: str = DEFAULT_OLLAMA_URL,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
    latency_ms: int | None = None,
    raw_error: str | None = None,
) -> RagMetadata:
    return RagMetadata(
        rag_mode="llm",
        llm_provider="Ollama",
        llm_model=model_name,
        endpoint_url=endpoint_url,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        latency_ms=latency_ms,
        raw_error=raw_error,
    )


@dataclass(frozen=True)
class OllamaConnectionCheck:
    success: bool
    provider: str
    model_name: str
    endpoint_url: str
    latency_ms: int
    error_message: str | None = None


@dataclass(frozen=True)
class RagAnswer:
    question: str
    likely_root_cause: str
    confidence: str
    supporting_incidents: list[SupportingIncident]
    evidence: list[str]
    recommended_action: str
    limitations: list[str] = field(default_factory=list)
    metadata: RagMetadata = field(default_factory=deterministic_metadata)


def infer_likely_root_cause(results: list[SearchResult]) -> str:
    modes = [
        mode
        for result in results
        for mode in result.metadata.get("failure_modes", [])
        if isinstance(mode, str)
    ]
    if not modes:
        return "Unknown from retrieved evidence"
    return Counter(modes).most_common(1)[0][0]


def confidence_from_scores(results: list[SearchResult]) -> str:
    if not results:
        return "none"
    top_score = results[0].score
    if top_score >= 0.7:
        return "high"
    if top_score >= 0.5:
        return "medium"
    return "low"


def collect_evidence(results: list[SearchResult], limit: int = 8) -> list[str]:
    evidence = []
    seen = set()
    for result in results:
        for item in result.evidence:
            if item not in seen:
                evidence.append(item)
                seen.add(item)
            if len(evidence) >= limit:
                return evidence
    return evidence


def recommended_action_for_root_cause(root_cause: str) -> str:
    if root_cause == "tool wear failure":
        return "Inspect tooling, replace worn tooling if needed, and confirm telemetry returns to normal bands."
    if root_cause == "power failure":
        return "Inspect power delivery, torque/load conditions, and validate machine operation before restart."
    if root_cause == "heat dissipation failure":
        return "Inspect cooling, airflow, and process temperature controls before returning the machine to service."
    if root_cause == "overstrain failure":
        return "Inspect mechanical load, torque, and rotational speed limits before continuing production."
    return "Review the supporting incidents and perform a targeted maintenance inspection."


def build_answer_from_results(question: str, results: list[SearchResult]) -> RagAnswer:
    if not results:
        return RagAnswer(
            question=question,
            likely_root_cause="No evidence available",
            confidence="none",
            supporting_incidents=[],
            evidence=[NO_RELEVANT_INCIDENTS_MESSAGE],
            recommended_action="Do not infer a root cause. Collect more evidence or lower the retrieval threshold.",
            limitations=["No retrieved incidents met the retrieval threshold."],
        )

    root_cause = infer_likely_root_cause(results)
    return RagAnswer(
        question=question,
        likely_root_cause=root_cause,
        confidence=confidence_from_scores(results),
        supporting_incidents=[
            SupportingIncident(
                document_id=result.document_id,
                document_type=result.document_type,
                machine_id=result.machine_id,
                title=result.title,
                score=result.score,
            )
            for result in results
        ],
        evidence=collect_evidence(results),
        recommended_action=recommended_action_for_root_cause(root_cause),
        limitations=["Answer generated by deterministic fallback from retrieved incident metadata and evidence."],
    )


def supporting_incidents_from_results(results: list[SearchResult]) -> list[SupportingIncident]:
    return [
        SupportingIncident(
            document_id=result.document_id,
            document_type=result.document_type,
            machine_id=result.machine_id,
            title=result.title,
            score=result.score,
        )
        for result in results
    ]


def compact_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3].rstrip()}..."


def build_prompt_context(results: list[SearchResult], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    sections = []
    remaining = max_chars
    for result in results:
        evidence = "\n".join(f"- {item}" for item in result.evidence)
        section = (
            f"Incident ID: {result.document_id}\n"
            f"Document type: {result.document_type}\n"
            f"Machine ID: {result.machine_id}\n"
            f"Title: {result.title}\n"
            f"Retrieval score: {result.score:.4f}\n"
            f"Body:\n{result.body}\n"
            f"Evidence:\n{evidence}"
        )
        if remaining <= 0:
            break
        section = compact_text(section, remaining)
        sections.append(section)
        remaining -= len(section) + 2
    return "\n\n".join(sections)


def build_ollama_prompt(question: str, results: list[SearchResult]) -> str:
    context = build_prompt_context(results)
    incident_ids = ", ".join(result.document_id for result in results)
    return (
        "You are an industrial incident investigation assistant.\n"
        "Answer only from the retrieved incident evidence below. Do not use outside knowledge. "
        "If the evidence is insufficient, say so in limitations and keep recommendations cautious.\n"
        "Return only valid JSON with exactly these keys: likely_root_cause, confidence, "
        "supporting_incidents, evidence, recommended_action, limitations.\n"
        "Use confidence as one of: high, medium, low, none.\n"
        "supporting_incidents must reference only these incident IDs: "
        f"{incident_ids}.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{context}"
    )


class OllamaUnavailableError(RuntimeError):
    pass


def ollama_model_from_env() -> str:
    return load_settings().ollama_model


def ollama_generate_url_from_env() -> str:
    return load_settings().ollama_generate_url


def call_ollama(
    prompt: str,
    model: str | None = None,
    url: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: float = 30.0,
) -> str:
    payload = json.dumps(
        {
            "model": model or ollama_model_from_env(),
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise OllamaUnavailableError(f"Ollama request failed: {exc}") from exc
    if "response" not in body:
        raise OllamaUnavailableError("Ollama response did not include a response field.")
    return str(body["response"])


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected Ollama to return a JSON object.")
    return parsed


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def normalize_supporting_incidents(value: Any, results: list[SearchResult]) -> list[SupportingIncident]:
    by_id = {result.document_id: result for result in results}
    ids = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                document_id = item.get("document_id") or item.get("incident_id") or item.get("id")
            else:
                document_id = item
            if document_id in by_id and document_id not in ids:
                ids.append(str(document_id))
    elif isinstance(value, str) and value in by_id:
        ids.append(value)
    if not ids:
        ids = [result.document_id for result in results]
    return supporting_incidents_from_results([by_id[document_id] for document_id in ids if document_id in by_id])


def build_answer_from_ollama_response(
    question: str,
    results: list[SearchResult],
    raw_response: str,
    model_name: str,
    endpoint_url: str = DEFAULT_OLLAMA_URL,
    latency_ms: int | None = None,
) -> RagAnswer:
    parsed = parse_json_object(raw_response)
    confidence = str(parsed.get("confidence", "low")).lower()
    if confidence not in {"high", "medium", "low", "none"}:
        confidence = "low"
    return RagAnswer(
        question=question,
        likely_root_cause=str(parsed.get("likely_root_cause", "Unknown from retrieved evidence")),
        confidence=confidence,
        supporting_incidents=normalize_supporting_incidents(parsed.get("supporting_incidents"), results),
        evidence=normalize_string_list(parsed.get("evidence")),
        recommended_action=str(
            parsed.get(
                "recommended_action",
                "Review the supporting incidents and perform a targeted maintenance inspection.",
            )
        ),
        limitations=normalize_string_list(parsed.get("limitations")),
        metadata=ollama_metadata(
            model_name=model_name,
            endpoint_url=endpoint_url,
            fallback_used=False,
            latency_ms=latency_ms,
        ),
    )


def build_llm_answer_from_results(
    question: str,
    results: list[SearchResult],
    fallback: bool = True,
) -> RagAnswer:
    model_name = ollama_model_from_env()
    endpoint_url = ollama_generate_url_from_env()
    if not results:
        answer = build_answer_from_results(question, results)
        fallback_reason = "No retrieved incidents met the threshold; Ollama was skipped because there was no evidence context."
        return RagAnswer(
            question=answer.question,
            likely_root_cause=answer.likely_root_cause,
            confidence=answer.confidence,
            supporting_incidents=answer.supporting_incidents,
            evidence=answer.evidence,
            recommended_action=answer.recommended_action,
            limitations=[
                fallback_reason,
                *answer.limitations,
            ],
            metadata=ollama_metadata(
                model_name=model_name,
                endpoint_url=endpoint_url,
                fallback_used=True,
                fallback_reason=fallback_reason,
                latency_ms=0,
            ),
        )
    started_at = time.perf_counter()
    try:
        raw_response = call_ollama(
            build_ollama_prompt(question, results),
            model=model_name,
            url=endpoint_url,
        )
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return build_answer_from_ollama_response(
            question,
            results,
            raw_response,
            model_name=model_name,
            endpoint_url=endpoint_url,
            latency_ms=latency_ms,
        )
    except (OllamaUnavailableError, ValueError, json.JSONDecodeError) as exc:
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        if not fallback:
            raise OllamaUnavailableError(f"Ollama unavailable and fallback disabled: {exc}") from exc
        error_message = redact_text(exc)
        answer = build_answer_from_results(question, results)
        return RagAnswer(
            question=answer.question,
            likely_root_cause=answer.likely_root_cause,
            confidence=answer.confidence,
            supporting_incidents=answer.supporting_incidents,
            evidence=answer.evidence,
            recommended_action=answer.recommended_action,
            limitations=[
                f"Ollama unavailable; used deterministic fallback. Error: {error_message}",
                *answer.limitations,
            ],
            metadata=ollama_metadata(
                model_name=model_name,
                endpoint_url=endpoint_url,
                fallback_used=True,
                fallback_reason="Ollama LLM generation failed; deterministic RAG fallback was used.",
                latency_ms=latency_ms,
                raw_error=error_message,
            ),
        )


def test_ollama_connection(
    model_name: str | None = None,
    endpoint_url: str | None = None,
    timeout_seconds: float = 10.0,
) -> OllamaConnectionCheck:
    model_name = model_name or ollama_model_from_env()
    endpoint_url = endpoint_url or ollama_generate_url_from_env()
    prompt = 'Return only valid JSON: {"status":"ok"}.'
    started_at = time.perf_counter()
    try:
        raw_response = call_ollama(
            prompt,
            model=model_name,
            url=endpoint_url,
            timeout_seconds=timeout_seconds,
        )
        parse_json_object(raw_response)
        return OllamaConnectionCheck(
            success=True,
            provider="Ollama",
            model_name=model_name,
            endpoint_url=endpoint_url,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
    except (OllamaUnavailableError, ValueError, json.JSONDecodeError) as exc:
        return OllamaConnectionCheck(
            success=False,
            provider="Ollama",
            model_name=model_name,
            endpoint_url=endpoint_url,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            error_message=redact_text(exc),
        )


def answer_question(
    question: str,
    top_k: int = 3,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    qdrant_path: Path = QDRANT_DATA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    llm: bool = False,
    fallback: bool = True,
) -> RagAnswer:
    embedder = load_embedder(embedding_model)
    response = retrieve_incidents(
        query=question,
        top_k=top_k,
        score_threshold=score_threshold,
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        embedder=embedder,
    )
    if llm:
        return build_llm_answer_from_results(question, response.results, fallback=fallback)
    return build_answer_from_results(question, response.results)


def answer_to_dict(answer: RagAnswer) -> dict[str, Any]:
    return asdict(answer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Answer a question using retrieved incident evidence.")
    parser.add_argument("question")
    parser.add_argument("--top-k", default=3, type=int)
    parser.add_argument("--score-threshold", default=DEFAULT_SCORE_THRESHOLD, type=float)
    parser.add_argument("--qdrant-path", default=QDRANT_DATA_DIR, type=Path)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm", action="store_true", help="Use local Ollama for answer synthesis.")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail instead of falling back to deterministic RAG when Ollama is unavailable.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        answer = answer_question(
            question=args.question,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            qdrant_path=args.qdrant_path,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            llm=args.llm,
            fallback=not args.no_fallback,
        )
    except OllamaUnavailableError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(json.dumps(answer_to_dict(answer), indent=2))


if __name__ == "__main__":
    main()
