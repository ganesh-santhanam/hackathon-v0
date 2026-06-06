import json

import numpy as np

from industrial_ai.incidents.memory import (
    NO_RELEVANT_INCIDENTS_MESSAGE,
    SearchResult,
    TelemetryQuery,
    build_match_reasons,
    build_document_type_filter,
    document_text,
    index_incident_corpus,
    load_incident_documents,
    retrieve_incidents,
    search_incidents,
    telemetry_similarity_score,
    with_match_reasons,
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
            "metadata": {
                "machine_type": "L",
                "failure_modes": ["power failure"],
                "telemetry": {
                    "tool_wear_min": 20,
                    "torque_nm": 10.0,
                    "rotational_speed_rpm": 2800,
                    "air_temperature_k": 299.0,
                    "process_temperature_k": 309.0,
                },
            },
            "evidence": ["Torque outside normal range"],
        },
        {
            "document_id": "doc-2",
            "document_type": "maintenance_note",
            "machine_id": "AI4I-00002",
            "title": "Tool Wear Maintenance",
            "body": "Technician replaced worn tooling after tool wear alarm.",
            "metadata": {
                "machine_type": "M",
                "failure_modes": ["tool wear failure"],
                "telemetry": {
                    "tool_wear_min": 210,
                    "torque_nm": 55.5,
                    "rotational_speed_rpm": 1266,
                    "air_temperature_k": 301.1,
                    "process_temperature_k": 311.6,
                },
            },
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


def test_telemetry_similarity_scores_exact_match_highest():
    query = TelemetryQuery(
        tool_wear_min=210,
        torque_nm=55.5,
        rotational_speed_rpm=1266,
        air_temperature_k=301.1,
        process_temperature_k=311.6,
    )

    assert telemetry_similarity_score(
        query,
        {
            "tool_wear_min": 210,
            "torque_nm": 55.5,
            "rotational_speed_rpm": 1266,
            "air_temperature_k": 301.1,
            "process_temperature_k": 311.6,
        },
    ) == 1.0


def test_build_match_reasons_explains_telemetry_semantics_and_failure_mode():
    result = SearchResult(
        score=0.72,
        document_id="doc-1",
        document_type="rca_report",
        machine_id="AI4I-01997",
        title="RCA Report - AI4I-01997",
        body="Root cause analysis points to tool wear failure.",
        metadata={
            "machine_type": "M",
            "failure_modes": ["tool wear failure"],
            "telemetry": {
                "tool_wear_min": 198,
                "torque_nm": 52.1,
                "rotational_speed_rpm": 1266,
                "air_temperature_k": 301.1,
                "process_temperature_k": 311.6,
            },
        },
        evidence=["Tool wear unusually high"],
        vector_score=0.71,
        telemetry_similarity_score=0.84,
        combined_score=0.76,
    )

    reasons = build_match_reasons(
        result,
        TelemetryQuery(
            tool_wear_min=210,
            torque_nm=55.5,
            rotational_speed_rpm=1266,
            air_temperature_k=301.1,
            process_temperature_k=311.6,
            machine_type="M",
        ),
    )

    assert "Similar tool wear: current 210 min vs incident 198 min" in reasons
    assert "Similar torque: current 55.5 Nm vs incident 52.1 Nm" in reasons
    assert "Similar rotational speed: current 1266 rpm vs incident 1266 rpm" in reasons
    assert any(reason.startswith("Similar temperature band:") for reason in reasons)
    assert "Same machine type: M" in reasons
    assert "Semantically similar RCA text" in reasons
    assert "Same failure mode: tool wear failure" in reasons


def test_match_explanation_handles_missing_telemetry_fields_safely():
    result = SearchResult(
        score=0.72,
        document_id="doc-1",
        document_type="maintenance_note",
        machine_id="AI4I-01997",
        title="Maintenance Note - AI4I-01997",
        body="Maintenance note points to tool wear failure.",
        metadata={
            "machine_type": "M",
            "failure_modes": ["tool wear failure"],
            "telemetry": {
                "tool_wear_min": 198,
            },
        },
        evidence=[],
        vector_score=0.71,
        combined_score=0.71,
    )

    explained = with_match_reasons(
        result,
        TelemetryQuery(
            tool_wear_min=210,
            torque_nm=55.5,
            rotational_speed_rpm=1266,
            air_temperature_k=301.1,
            process_temperature_k=311.6,
            machine_type="M",
        ),
    )

    assert "Similar tool wear: current 210 min vs incident 198 min" in explained.match_reasons
    assert "Same machine type: M" in explained.match_reasons
    assert explained.failure_mode == "tool wear failure"
    assert explained.telemetry_comparison[0] == {
        "signal": "tool wear",
        "current": "210",
        "incident": "198",
        "unit": "min",
        "similarity": 0.88,
    }
    assert explained.telemetry_comparison[1]["incident"] == "n/a"
    assert explained.telemetry_comparison[1]["similarity"] is None


def test_retrieve_incidents_can_rerank_by_exact_telemetry_match(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    qdrant_path = tmp_path / "qdrant"
    write_temp_corpus(corpus_path)
    embedder = FakeEmbedder()
    index_incident_corpus(
        corpus_path=corpus_path,
        qdrant_path=qdrant_path,
        collection_name="test_incidents_telemetry_rerank",
        embedder=embedder,
    )

    response = retrieve_incidents(
        query="torque power issue",
        top_k=2,
        qdrant_path=qdrant_path,
        collection_name="test_incidents_telemetry_rerank",
        embedder=embedder,
        score_threshold=0.0,
        telemetry_query=TelemetryQuery(
            tool_wear_min=210,
            torque_nm=55.5,
            rotational_speed_rpm=1266,
            air_temperature_k=301.1,
            process_temperature_k=311.6,
            machine_type="M",
        ),
        vector_weight=0.2,
    )

    assert response.results[0].document_id == "doc-2"
    assert response.results[0].vector_score == 0.0
    assert response.results[0].telemetry_similarity_score == 1.0
    assert response.results[0].combined_score == 0.8
    assert response.results[0].match_reasons
    assert response.results[0].failure_mode == "tool wear failure"
    assert response.results[0].telemetry_comparison
