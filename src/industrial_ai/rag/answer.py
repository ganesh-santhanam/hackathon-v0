import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class SupportingIncident:
    document_id: str
    document_type: str
    machine_id: str
    title: str
    score: float


@dataclass(frozen=True)
class RagAnswer:
    question: str
    likely_root_cause: str
    confidence: str
    supporting_incidents: list[SupportingIncident]
    evidence: list[str]
    recommended_action: str


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
    )


def answer_question(
    question: str,
    top_k: int = 3,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    qdrant_path: Path = QDRANT_DATA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    answer = answer_question(
        question=args.question,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(answer_to_dict(answer), indent=2))


if __name__ == "__main__":
    main()
