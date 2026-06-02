from industrial_ai.incidents.memory import SearchResponse, SearchResult
from industrial_ai.rag import answer as answer_module
from industrial_ai.rag.answer import (
    DEFAULT_OLLAMA_MODEL,
    OllamaUnavailableError,
    build_answer_from_results,
    build_llm_answer_from_results,
    build_ollama_prompt,
    collect_evidence,
    confidence_from_scores,
    infer_likely_root_cause,
    ollama_model_from_env,
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


def test_build_ollama_prompt_uses_retrieved_incident_context_only():
    result = make_result()

    prompt = build_ollama_prompt("What happened?", [result])

    assert "Answer only from the retrieved incident evidence" in prompt
    assert "doc-1" in prompt
    assert "Root cause analysis points to tool wear failure." in prompt
    assert "supporting_incidents must reference only these incident IDs: doc-1" in prompt


def test_llm_answer_uses_mocked_ollama_response(monkeypatch):
    result = make_result()

    def fake_call_ollama(prompt):
        assert "doc-1" in prompt
        return """
        {
          "likely_root_cause": "tool wear failure",
          "confidence": "high",
          "supporting_incidents": ["doc-1"],
          "evidence": ["Tool wear: 220 min"],
          "recommended_action": "Inspect and replace tooling.",
          "limitations": ["Only one retrieved incident was available."]
        }
        """

    monkeypatch.setattr(answer_module, "call_ollama", fake_call_ollama)

    answer = build_llm_answer_from_results("What happened?", [result])

    assert answer.likely_root_cause == "tool wear failure"
    assert answer.confidence == "high"
    assert answer.supporting_incidents[0].document_id == "doc-1"
    assert answer.evidence == ["Tool wear: 220 min"]
    assert answer.limitations == ["Only one retrieved incident was available."]


def test_llm_answer_falls_back_when_ollama_unavailable(monkeypatch):
    result = make_result()

    def fake_call_ollama(prompt):
        raise OllamaUnavailableError("connection refused")

    monkeypatch.setattr(answer_module, "call_ollama", fake_call_ollama)

    answer = build_llm_answer_from_results("What happened?", [result])

    assert answer.likely_root_cause == "tool wear failure"
    assert "Ollama unavailable; used deterministic fallback" in answer.limitations[0]


def test_llm_answer_raises_when_fallback_disabled(monkeypatch):
    result = make_result()

    def fake_call_ollama(prompt):
        raise OllamaUnavailableError("connection refused")

    monkeypatch.setattr(answer_module, "call_ollama", fake_call_ollama)

    try:
        build_llm_answer_from_results("What happened?", [result], fallback=False)
    except OllamaUnavailableError as exc:
        assert "fallback disabled" in str(exc)
    else:
        raise AssertionError("Expected OllamaUnavailableError")


def test_ollama_model_uses_env_var_with_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert ollama_model_from_env() == DEFAULT_OLLAMA_MODEL

    monkeypatch.setenv("OLLAMA_MODEL", "custom-model")
    assert ollama_model_from_env() == "custom-model"
