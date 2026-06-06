import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from industrial_ai.paths import INCIDENTS_DATA_DIR, QDRANT_DATA_DIR


DEFAULT_COLLECTION_NAME = "incident_documents"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CORPUS_PATH = INCIDENTS_DATA_DIR / "ai4i_incident_corpus.jsonl"
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_CANDIDATE_MULTIPLIER = 5
NO_RELEVANT_INCIDENTS_MESSAGE = "No relevant incidents found"
TELEMETRY_FIELDS = (
    "tool_wear_min",
    "torque_nm",
    "rotational_speed_rpm",
    "air_temperature_k",
    "process_temperature_k",
)
TELEMETRY_TOLERANCES = {
    "tool_wear_min": 100.0,
    "torque_nm": 20.0,
    "rotational_speed_rpm": 1000.0,
    "air_temperature_k": 10.0,
    "process_temperature_k": 10.0,
}
TELEMETRY_LABELS = {
    "tool_wear_min": ("tool wear", "min"),
    "torque_nm": ("torque", "Nm"),
    "rotational_speed_rpm": ("rotational speed", "rpm"),
    "air_temperature_k": ("air temperature", "K"),
    "process_temperature_k": ("process temperature", "K"),
}
TELEMETRY_COMPARISON_FIELDS = (
    "tool_wear_min",
    "torque_nm",
    "rotational_speed_rpm",
    "air_temperature_k",
    "process_temperature_k",
)
TELEMETRY_REASON_THRESHOLD = 0.7


class Embedder(Protocol):
    def encode(self, sentences: str | list[str], normalize_embeddings: bool = True) -> Any:
        pass


@dataclass(frozen=True)
class SearchResult:
    score: float
    document_id: str
    document_type: str
    machine_id: str
    title: str
    body: str
    metadata: dict[str, Any]
    evidence: list[str]
    vector_score: float | None = None
    telemetry_similarity_score: float | None = None
    combined_score: float | None = None
    match_reasons: list[str] = field(default_factory=list)
    telemetry_comparison: list[dict[str, Any]] = field(default_factory=list)
    failure_mode: str | None = None


@dataclass(frozen=True)
class TelemetryQuery:
    tool_wear_min: float
    torque_nm: float
    rotational_speed_rpm: float
    air_temperature_k: float
    process_temperature_k: float
    machine_type: str | None = None


@dataclass(frozen=True)
class SearchResponse:
    query: str
    top_k: int
    score_threshold: float
    top_score: float | None
    message: str
    results: list[SearchResult]


def load_incident_documents(corpus_path: Path = DEFAULT_CORPUS_PATH) -> list[dict[str, Any]]:
    with corpus_path.open(encoding="utf-8") as corpus_file:
        return [json.loads(line) for line in corpus_file if line.strip()]


def document_text(document: dict[str, Any]) -> str:
    evidence = "\n".join(document["evidence"])
    return f"{document['title']}\n\n{document['body']}\n\nEvidence:\n{evidence}"


def load_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    return [embedding.tolist() if hasattr(embedding, "tolist") else list(embedding) for embedding in embeddings]


def embed_query(embedder: Embedder, query: str) -> list[float]:
    embedding = embedder.encode(query, normalize_embeddings=True)
    if hasattr(embedding, "tolist"):
        return embedding.tolist()
    return list(embedding)


def build_qdrant_client(qdrant_path: Path = QDRANT_DATA_DIR) -> QdrantClient:
    qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(qdrant_path))


def index_incident_corpus(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    qdrant_path: Path = QDRANT_DATA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedder: Embedder | None = None,
) -> int:
    documents = load_incident_documents(corpus_path)
    if not documents:
        raise ValueError(f"No incident documents found in {corpus_path}")

    embedder = embedder or load_embedder()
    vectors = embed_texts(embedder, [document_text(document) for document in documents])
    vector_size = len(vectors[0])

    client = build_qdrant_client(qdrant_path)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=index,
                vector=vector,
                payload={
                    "document_id": document["document_id"],
                    "document_type": document["document_type"],
                    "machine_id": document["machine_id"],
                    "title": document["title"],
                    "body": document["body"],
                    "metadata": document["metadata"],
                    "evidence": document["evidence"],
                },
            )
            for index, (document, vector) in enumerate(zip(documents, vectors, strict=True))
        ],
    )
    client.close()
    return len(documents)


def build_document_type_filter(document_type: str | None) -> Filter | None:
    if document_type is None:
        return None
    return Filter(
        must=[
            FieldCondition(
                key="document_type",
                match=MatchValue(value=document_type),
            )
        ]
    )


def telemetry_from_metadata(metadata: dict[str, Any]) -> dict[str, float] | None:
    telemetry = metadata.get("telemetry")
    if not isinstance(telemetry, dict):
        return None
    if any(field_name not in telemetry for field_name in TELEMETRY_FIELDS):
        return None
    return {field_name: float(telemetry[field_name]) for field_name in TELEMETRY_FIELDS}


def partial_telemetry_from_metadata(metadata: dict[str, Any]) -> dict[str, float]:
    telemetry = metadata.get("telemetry")
    if not isinstance(telemetry, dict):
        return {}

    values = {}
    for field_name in TELEMETRY_FIELDS:
        if field_name not in telemetry:
            continue
        try:
            values[field_name] = float(telemetry[field_name])
        except (TypeError, ValueError):
            continue
    return values


def telemetry_similarity_score(
    query: TelemetryQuery,
    document_telemetry: dict[str, float] | None,
) -> float | None:
    if document_telemetry is None:
        return None
    scores = []
    for field_name in TELEMETRY_FIELDS:
        query_value = float(getattr(query, field_name))
        document_value = float(document_telemetry[field_name])
        tolerance = TELEMETRY_TOLERANCES[field_name]
        scores.append(max(0.0, 1.0 - (abs(query_value - document_value) / tolerance)))
    return float(sum(scores) / len(scores))


def format_telemetry_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


def telemetry_match_reasons(
    query: TelemetryQuery,
    document_telemetry: dict[str, float] | None,
    similarity_threshold: float = TELEMETRY_REASON_THRESHOLD,
) -> list[str]:
    if document_telemetry is None:
        return []

    reasons = []
    for field_name in TELEMETRY_FIELDS:
        if field_name not in document_telemetry:
            continue
        query_value = float(getattr(query, field_name))
        document_value = float(document_telemetry[field_name])
        tolerance = TELEMETRY_TOLERANCES[field_name]
        similarity = max(0.0, 1.0 - (abs(query_value - document_value) / tolerance))
        if similarity < similarity_threshold:
            continue

        label, unit = TELEMETRY_LABELS[field_name]
        reasons.append(
            f"Similar {label}: current {format_telemetry_value(query_value)} {unit} "
            f"vs incident {format_telemetry_value(document_value)} {unit}"
        )
    return reasons


def temperature_band_match_reason(
    query: TelemetryQuery,
    document_telemetry: dict[str, float],
    similarity_threshold: float = TELEMETRY_REASON_THRESHOLD,
) -> str | None:
    temperature_fields = ("air_temperature_k", "process_temperature_k")
    if any(field_name not in document_telemetry for field_name in temperature_fields):
        return None
    similarities = []
    for field_name in temperature_fields:
        query_value = float(getattr(query, field_name))
        document_value = float(document_telemetry[field_name])
        tolerance = TELEMETRY_TOLERANCES[field_name]
        similarities.append(max(0.0, 1.0 - (abs(query_value - document_value) / tolerance)))
    if sum(similarities) / len(similarities) < similarity_threshold:
        return None
    return (
        "Similar temperature band: current "
        f"{format_telemetry_value(query.air_temperature_k)} K air / "
        f"{format_telemetry_value(query.process_temperature_k)} K process vs incident "
        f"{format_telemetry_value(document_telemetry['air_temperature_k'])} K air / "
        f"{format_telemetry_value(document_telemetry['process_temperature_k'])} K process"
    )


def semantic_match_reason(result: SearchResult) -> str:
    if result.document_type == "rca_report":
        return "Semantically similar RCA text"
    return f"Semantically similar {result.document_type.replace('_', ' ')} text"


def failure_mode_match_reasons(result: SearchResult) -> list[str]:
    failure_modes = result.metadata.get("failure_modes", [])
    if not isinstance(failure_modes, list):
        return []
    return [
        f"Same failure mode: {failure_mode}"
        for failure_mode in failure_modes
        if isinstance(failure_mode, str)
    ]


def primary_failure_mode(metadata: dict[str, Any]) -> str | None:
    failure_modes = metadata.get("failure_modes", [])
    if not isinstance(failure_modes, list):
        return None
    return next((mode for mode in failure_modes if isinstance(mode, str)), None)


def machine_type_match_reason(result: SearchResult, telemetry_query: TelemetryQuery | None) -> str | None:
    if telemetry_query is None or telemetry_query.machine_type is None:
        return None
    historical_type = result.metadata.get("machine_type")
    if not isinstance(historical_type, str) or historical_type != telemetry_query.machine_type:
        return None
    return f"Same machine type: {historical_type}"


def telemetry_comparison_rows(
    query: TelemetryQuery,
    document_telemetry: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for field_name in TELEMETRY_COMPARISON_FIELDS:
        label, unit = TELEMETRY_LABELS[field_name]
        row = {
            "signal": label,
            "current": format_telemetry_value(float(getattr(query, field_name))),
            "incident": "n/a",
            "unit": unit,
        }
        if field_name in document_telemetry:
            row["incident"] = format_telemetry_value(document_telemetry[field_name])
            tolerance = TELEMETRY_TOLERANCES[field_name]
            row["similarity"] = round(
                max(
                    0.0,
                    1.0 - (abs(float(getattr(query, field_name)) - document_telemetry[field_name]) / tolerance),
                ),
                3,
            )
        else:
            row["similarity"] = None
        rows.append(row)
    return rows


def build_match_reasons(
    result: SearchResult,
    telemetry_query: TelemetryQuery | None = None,
) -> list[str]:
    reasons = []
    if telemetry_query is not None:
        document_telemetry = partial_telemetry_from_metadata(result.metadata)
        reasons.extend(
            telemetry_match_reasons(
                telemetry_query,
                document_telemetry or None,
            )
        )
        temperature_reason = temperature_band_match_reason(telemetry_query, document_telemetry)
        if temperature_reason and temperature_reason not in reasons:
            reasons.append(temperature_reason)
        machine_type_reason = machine_type_match_reason(result, telemetry_query)
        if machine_type_reason:
            reasons.append(machine_type_reason)
    if result.vector_score is None or result.vector_score >= DEFAULT_SCORE_THRESHOLD:
        reasons.append(semantic_match_reason(result))
    reasons.extend(failure_mode_match_reasons(result))
    return reasons


def with_match_reasons(
    result: SearchResult,
    telemetry_query: TelemetryQuery | None = None,
) -> SearchResult:
    document_telemetry = partial_telemetry_from_metadata(result.metadata)
    return replace(
        result,
        match_reasons=build_match_reasons(result, telemetry_query),
        telemetry_comparison=(
            telemetry_comparison_rows(telemetry_query, document_telemetry)
            if telemetry_query is not None
            else []
        ),
        failure_mode=primary_failure_mode(result.metadata),
    )


def combine_scores(
    vector_score: float,
    telemetry_score: float | None,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
) -> float:
    if telemetry_score is None:
        return vector_score
    return (vector_weight * vector_score) + ((1 - vector_weight) * telemetry_score)


def rerank_with_telemetry(
    results: list[SearchResult],
    telemetry_query: TelemetryQuery | None,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
) -> list[SearchResult]:
    if telemetry_query is None:
        return results

    reranked = []
    for result in results:
        vector_score = result.vector_score if result.vector_score is not None else result.score
        telemetry_score = telemetry_similarity_score(
            telemetry_query,
            telemetry_from_metadata(result.metadata),
        )
        combined_score = combine_scores(vector_score, telemetry_score, vector_weight=vector_weight)
        reranked_result = SearchResult(
            score=combined_score,
            document_id=result.document_id,
            document_type=result.document_type,
            machine_id=result.machine_id,
            title=result.title,
            body=result.body,
            metadata=result.metadata,
            evidence=result.evidence,
            vector_score=vector_score,
            telemetry_similarity_score=telemetry_score,
            combined_score=combined_score,
        )
        reranked.append(with_match_reasons(reranked_result, telemetry_query))
    return sorted(reranked, key=lambda item: item.combined_score or item.score, reverse=True)


def search_incidents(
    query: str,
    top_k: int = 5,
    document_type: str | None = None,
    qdrant_path: Path = QDRANT_DATA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedder: Embedder | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    telemetry_query: TelemetryQuery | None = None,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
) -> list[SearchResult]:
    embedder = embedder or load_embedder()
    query_vector = embed_query(embedder, query)
    vector_limit = top_k * candidate_multiplier if telemetry_query else top_k
    client = build_qdrant_client(qdrant_path)
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=build_document_type_filter(document_type),
        limit=vector_limit,
        with_payload=True,
    )
    client.close()

    results = []
    for point in response.points:
        if point.score < score_threshold:
            continue
        payload = point.payload or {}
        result = SearchResult(
            score=float(point.score),
            document_id=payload["document_id"],
            document_type=payload["document_type"],
            machine_id=payload["machine_id"],
            title=payload["title"],
            body=payload["body"],
            metadata=payload["metadata"],
            evidence=payload["evidence"],
            vector_score=float(point.score),
            telemetry_similarity_score=None,
            combined_score=float(point.score),
        )
        results.append(with_match_reasons(result))
    return rerank_with_telemetry(
        results,
        telemetry_query=telemetry_query,
        vector_weight=vector_weight,
    )[:top_k]


def retrieve_incidents(
    query: str,
    top_k: int = 5,
    document_type: str | None = None,
    qdrant_path: Path = QDRANT_DATA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedder: Embedder | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    telemetry_query: TelemetryQuery | None = None,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
) -> SearchResponse:
    raw_results = search_incidents(
        query=query,
        top_k=top_k,
        document_type=document_type,
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        embedder=embedder,
        score_threshold=float("-inf"),
        telemetry_query=telemetry_query,
        vector_weight=vector_weight,
    )
    top_score = raw_results[0].score if raw_results else None
    results = [result for result in raw_results if result.score >= score_threshold]
    return SearchResponse(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        top_score=top_score,
        message="Relevant incidents found" if results else NO_RELEVANT_INCIDENTS_MESSAGE,
        results=results,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Qdrant memory for incident documents.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index the generated incident corpus.")
    index_parser.add_argument("--corpus-path", default=DEFAULT_CORPUS_PATH, type=Path)
    index_parser.add_argument("--qdrant-path", default=QDRANT_DATA_DIR, type=Path)
    index_parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    index_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)

    search_parser = subparsers.add_parser("search", help="Search indexed incident documents.")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", default=5, type=int)
    search_parser.add_argument("--score-threshold", default=DEFAULT_SCORE_THRESHOLD, type=float)
    search_parser.add_argument("--document-type", choices=["incident_report", "rca_report", "maintenance_note"])
    search_parser.add_argument("--qdrant-path", default=QDRANT_DATA_DIR, type=Path)
    search_parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    search_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    search_parser.add_argument("--telemetry-rerank", action="store_true")
    search_parser.add_argument("--tool-wear-min", type=float)
    search_parser.add_argument("--torque-nm", type=float)
    search_parser.add_argument("--rotational-speed-rpm", type=float)
    search_parser.add_argument("--air-temperature-k", type=float)
    search_parser.add_argument("--process-temperature-k", type=float)
    search_parser.add_argument("--vector-weight", default=DEFAULT_VECTOR_WEIGHT, type=float)
    return parser


def telemetry_query_from_args(args: argparse.Namespace) -> TelemetryQuery | None:
    values = {
        "tool_wear_min": args.tool_wear_min,
        "torque_nm": args.torque_nm,
        "rotational_speed_rpm": args.rotational_speed_rpm,
        "air_temperature_k": args.air_temperature_k,
        "process_temperature_k": args.process_temperature_k,
    }
    if not args.telemetry_rerank and all(value is None for value in values.values()):
        return None
    missing = [field for field, value in values.items() if value is None]
    if missing:
        raise ValueError(f"Telemetry reranking requires all telemetry fields: {missing}")
    return TelemetryQuery(**values)


def main() -> None:
    args = build_parser().parse_args()
    embedder = load_embedder(args.embedding_model)
    if args.command == "index":
        count = index_incident_corpus(
            corpus_path=args.corpus_path,
            qdrant_path=args.qdrant_path,
            collection_name=args.collection_name,
            embedder=embedder,
        )
        print(json.dumps({"indexed_documents": count, "collection_name": args.collection_name}, indent=2))
        return

    response = retrieve_incidents(
        query=args.query,
        top_k=args.top_k,
        document_type=args.document_type,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection_name,
        embedder=embedder,
        score_threshold=args.score_threshold,
        telemetry_query=telemetry_query_from_args(args),
        vector_weight=args.vector_weight,
    )
    print(
        json.dumps(
            {
                "query": response.query,
                "top_k": response.top_k,
                "score_threshold": response.score_threshold,
                "top_score": response.top_score,
                "message": response.message,
                "results": [result.__dict__ for result in response.results],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
