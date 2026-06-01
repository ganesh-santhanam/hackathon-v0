import json

import numpy as np

from industrial_ai.incidents.memory import (
    NO_RELEVANT_INCIDENTS_MESSAGE,
    build_document_type_filter,
    document_text,
    index_incident_corpus,
    load_incident_documents,
    retrieve_incidents,
    search_incidents,
)


class FakeEmbedder:
    def encode(self, sentences, normalize_embeddings=True):
        if isinstance(sentences, str):
            return self._encode_one(sentences)
        return np.array([self._encode_one(sentence) for sentence in sentences])

    def _encode_one(self, sentence):
        text = sentence.lower()
        if "torque" in text or "power" in text:
            return np.array([1.0, 0.0, 0.0])
        if "tool wear" in text:
            return np.array([0.0, 1.0, 0.0])
        return np.array([0.0, 0.0, 1.0])


def write_temp_corpus(path):
    documents = [
        {
            "document_id": "doc-1",
            "document_type": "incident_report",
            "machine_id": "AI4I-00001",
            "title": "Power Incident",
            "body": "Machine showed a power failure with abnormal torque.",
            "metadata": {"failure_modes": ["power failure"]},
            "evidence": ["Torque outside normal range"],
        },
        {
            "document_id": "doc-2",
            "document_type": "maintenance_note",
            "machine_id": "AI4I-00002",
            "title": "Tool Wear Maintenance",
            "body": "Technician replaced worn tooling after tool wear alarm.",
            "metadata": {"failure_modes": ["tool wear failure"]},
            "evidence": ["Tool wear unusually high"],
        },
    ]
    with path.open("w", encoding="utf-8") as corpus_file:
        for document in documents:
            corpus_file.write(json.dumps(document) + "\n")


def test_load_incident_documents_reads_jsonl(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    write_temp_corpus(corpus_path)

    documents = load_incident_documents(corpus_path)

    assert len(documents) == 2
    assert documents[0]["document_id"] == "doc-1"


def test_document_text_combines_title_body_and_evidence():
    document = {
        "title": "Title",
        "body": "Body",
        "evidence": ["Evidence one", "Evidence two"],
    }

    text = document_text(document)

    assert "Title" in text
    assert "Body" in text
    assert "Evidence one" in text


def test_build_document_type_filter_is_optional():
    assert build_document_type_filter(None) is None
    assert build_document_type_filter("rca_report") is not None


def test_index_and_search_incidents_with_temp_qdrant(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    qdrant_path = tmp_path / "qdrant"
    write_temp_corpus(corpus_path)
    embedder = FakeEmbedder()

    indexed_count = index_incident_corpus(
        corpus_path=corpus_path,
        qdrant_path=qdrant_path,
        collection_name="test_incidents",
        embedder=embedder,
    )
    results = search_incidents(
        query="torque power issue",
        top_k=1,
        qdrant_path=qdrant_path,
        collection_name="test_incidents",
        embedder=embedder,
    )

    assert indexed_count == 2
    assert results[0].document_id == "doc-1"
    assert results[0].document_type == "incident_report"


def test_search_incidents_filters_by_document_type(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    qdrant_path = tmp_path / "qdrant"
    write_temp_corpus(corpus_path)
    embedder = FakeEmbedder()
    index_incident_corpus(
        corpus_path=corpus_path,
        qdrant_path=qdrant_path,
        collection_name="test_incidents_filtered",
        embedder=embedder,
    )

    results = search_incidents(
        query="torque power issue",
        top_k=2,
        document_type="maintenance_note",
        qdrant_path=qdrant_path,
        collection_name="test_incidents_filtered",
        embedder=embedder,
        score_threshold=0.0,
    )

    assert len(results) == 1
    assert results[0].document_id == "doc-2"


def test_retrieve_incidents_includes_top_score_and_threshold(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    qdrant_path = tmp_path / "qdrant"
    write_temp_corpus(corpus_path)
    embedder = FakeEmbedder()
    index_incident_corpus(
        corpus_path=corpus_path,
        qdrant_path=qdrant_path,
        collection_name="test_incidents_threshold",
        embedder=embedder,
    )

    response = retrieve_incidents(
        query="torque power issue",
        top_k=2,
        qdrant_path=qdrant_path,
        collection_name="test_incidents_threshold",
        embedder=embedder,
        score_threshold=0.9,
    )

    assert response.top_score == 1.0
    assert response.score_threshold == 0.9
    assert response.message == "Relevant incidents found"
    assert response.results[0].document_id == "doc-1"


def test_retrieve_incidents_returns_no_relevant_message_below_threshold(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    qdrant_path = tmp_path / "qdrant"
    write_temp_corpus(corpus_path)
    embedder = FakeEmbedder()
    index_incident_corpus(
        corpus_path=corpus_path,
        qdrant_path=qdrant_path,
        collection_name="test_incidents_no_relevant",
        embedder=embedder,
    )

    response = retrieve_incidents(
        query="torque power issue",
        top_k=2,
        qdrant_path=qdrant_path,
        collection_name="test_incidents_no_relevant",
        embedder=embedder,
        score_threshold=1.1,
    )

    assert response.top_score == 1.0
    assert response.message == NO_RELEVANT_INCIDENTS_MESSAGE
    assert response.results == []
