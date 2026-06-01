import argparse
import json
from dataclasses import dataclass
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
NO_RELEVANT_INCIDENTS_MESSAGE = "No relevant incidents found"


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


def search_incidents(
    query: str,
    top_k: int = 5,
    document_type: str | None = None,
    qdrant_path: Path = QDRANT_DATA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedder: Embedder | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[SearchResult]:
    embedder = embedder or load_embedder()
    query_vector = embed_query(embedder, query)
    client = build_qdrant_client(qdrant_path)
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=build_document_type_filter(document_type),
        limit=top_k,
        with_payload=True,
    )
    client.close()

    results = []
    for point in response.points:
        if point.score < score_threshold:
            continue
        payload = point.payload or {}
        results.append(
            SearchResult(
                score=float(point.score),
                document_id=payload["document_id"],
                document_type=payload["document_type"],
                machine_id=payload["machine_id"],
                title=payload["title"],
                body=payload["body"],
                metadata=payload["metadata"],
                evidence=payload["evidence"],
            )
        )
    return results


def retrieve_incidents(
    query: str,
    top_k: int = 5,
    document_type: str | None = None,
    qdrant_path: Path = QDRANT_DATA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedder: Embedder | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> SearchResponse:
    raw_results = search_incidents(
        query=query,
        top_k=top_k,
        document_type=document_type,
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        embedder=embedder,
        score_threshold=float("-inf"),
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
    return parser


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
