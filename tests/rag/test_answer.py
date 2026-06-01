from industrial_ai.incidents.memory import SearchResponse, SearchResult
from industrial_ai.rag import answer as answer_module
from industrial_ai.rag.answer import (
    build_answer_from_results,
    collect_evidence,
    confidence_from_scores,
    infer_likely_root_cause,
)


def make_result(
    document_id="doc-1",
    score=0.72,
    failure_modes=None,
    evidence=None,
):
    return SearchResult(
        score=score,
        document_id=document_id,
        document_type="rca_report",
        machine_id="AI4I-00001",
        title="RCA Report - AI4I-00001",
        body="Root cause analysis points to tool wear failure.",
        metadata={"failure_modes": failure_modes or ["tool wear failure"]},
        evidence=evidence
        or [
            "Tool wear: 220 min",
            "Torque: 55.5 Nm",
        ],
    )


def test_build_answer_from_results_uses_only_retrieved_documents():
    result = make_result()

    answer = build_answer_from_results("What happened?", [result])

    assert answer.likely_root_cause == "tool wear failure"
    assert answer.confidence == "high"
    assert answer.supporting_incidents[0].document_id == "doc-1"
    assert "Tool wear: 220 min" in answer.evidence
    assert "Inspect tooling" in answer.recommended_action


def test_build_answer_from_results_returns_no_evidence_response():
    answer = build_answer_from_results("What happened?", [])

    assert answer.likely_root_cause == "No evidence available"
    assert answer.confidence == "none"
    assert answer.supporting_incidents == []
    assert answer.evidence == ["No relevant incidents found"]
    assert answer.recommended_action.startswith("Do not infer")


def test_infer_likely_root_cause_uses_majority_failure_mode():
    results = [
        make_result("doc-1", failure_modes=["power failure"]),
        make_result("doc-2", failure_modes=["tool wear failure"]),
        make_result("doc-3", failure_modes=["tool wear failure"]),
    ]

    assert infer_likely_root_cause(results) == "tool wear failure"


def test_confidence_from_scores_uses_top_score():
    assert confidence_from_scores([make_result(score=0.71)]) == "high"
    assert confidence_from_scores([make_result(score=0.51)]) == "medium"
    assert confidence_from_scores([make_result(score=0.3)]) == "low"
    assert confidence_from_scores([]) == "none"


def test_collect_evidence_deduplicates_items():
    results = [
        make_result("doc-1", evidence=["Tool wear: 220 min", "Torque: 55.5 Nm"]),
        make_result("doc-2", evidence=["Tool wear: 220 min", "Rotational speed: 1266 rpm"]),
    ]

    assert collect_evidence(results) == [
        "Tool wear: 220 min",
        "Torque: 55.5 Nm",
        "Rotational speed: 1266 rpm",
    ]


def test_answer_question_uses_retrieval_results(monkeypatch):
    monkeypatch.setattr(answer_module, "load_embedder", lambda _: object())

    def fake_retrieve_incidents(**kwargs):
        return SearchResponse(
            query=kwargs["query"],
            top_k=kwargs["top_k"],
            score_threshold=kwargs["score_threshold"],
            top_score=0.72,
            message="Relevant incidents found",
            results=[make_result()],
        )

    monkeypatch.setattr(answer_module, "retrieve_incidents", fake_retrieve_incidents)

    answer = answer_module.answer_question("tool wear failure")

    assert answer.likely_root_cause == "tool wear failure"
    assert answer.supporting_incidents[0].score == 0.72
