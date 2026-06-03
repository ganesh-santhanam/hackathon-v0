from industrial_ai.demo import graph_workflow
from industrial_ai.demo.graph_workflow import NODE_ORDER, run_investigation_graph
from industrial_ai.demo.investigation import VisionCheck
from industrial_ai.incidents.memory import SearchResponse, SearchResult
from industrial_ai.telemetry.predict import FailurePrediction, RiskLevel, TelemetryReading


def make_reading():
    return TelemetryReading(
        machine_id="FAN-023",
        type="M",
        air_temperature_k=301.1,
        process_temperature_k=311.6,
        rotational_speed_rpm=1266,
        torque_nm=55.5,
        tool_wear_min=210,
    )


def make_prediction():
    return FailurePrediction(
        machine_id="FAN-023",
        failure_probability=0.82,
        failure_probability_percent=82,
        risk_level=RiskLevel.HIGH,
        top_feature_importances=[],
        evidence=["Tool wear unusually high"],
    )


def make_search_result():
    return SearchResult(
        score=0.72,
        document_id="doc-1",
        document_type="rca_report",
        machine_id="AI4I-00001",
        title="RCA Report - AI4I-00001",
        body="Root cause analysis points to tool wear failure.",
        metadata={"failure_modes": ["tool wear failure"]},
        evidence=["Tool wear: 210 min", "Torque: 55.5 Nm"],
    )


def make_vision():
    return VisionCheck(
        method="resnet",
        category="cable",
        defect_detected=True,
        defect_type="cut_outer_insulation",
        confidence=0.91,
        anomaly_score=0.31,
        threshold=0.25,
        evidence=["Embedding distance anomaly score: 0.3100"],
    )


def install_graph_stubs(monkeypatch):
    monkeypatch.setattr(graph_workflow, "predict_failure", lambda _: make_prediction())
    monkeypatch.setattr(graph_workflow, "load_embedder", lambda _: object())
    monkeypatch.setattr(graph_workflow, "run_vision_check", lambda **_: make_vision())

    def fake_retrieve_incidents(**kwargs):
        assert kwargs["telemetry_query"].tool_wear_min == 210
        return SearchResponse(
            query=kwargs["query"],
            top_k=kwargs["top_k"],
            score_threshold=kwargs["score_threshold"],
            top_score=0.72,
            message="Relevant incidents found",
            results=[make_search_result()],
        )

    monkeypatch.setattr(graph_workflow, "retrieve_incidents", fake_retrieve_incidents)


def test_graph_workflow_runs_requested_agent_nodes(monkeypatch, tmp_path):
    install_graph_stubs(monkeypatch)

    result = run_investigation_graph(
        reading=make_reading(),
        vision_image_path=tmp_path / "image.png",
        vision_category="cable",
        approvals_store_path=tmp_path / "approvals.json",
    )

    assert result.prediction.risk_level == RiskLevel.HIGH
    assert result.vision is not None
    assert result.similar_incidents[0].document_id == "doc-1"
    assert result.rag_answer.likely_root_cause == "tool wear failure"
    assert result.severity.severity.value == "SEV1"
    assert result.approval.status.value == "pending"
    assert [item.split(":")[0] for item in result.agent_trace] == list(NODE_ORDER)


def test_graph_workflow_skips_vision_node_when_no_image(monkeypatch, tmp_path):
    install_graph_stubs(monkeypatch)

    result = run_investigation_graph(
        reading=make_reading(),
        approvals_store_path=tmp_path / "approvals.json",
    )

    assert result.vision is None
    assert "vision_agent: skipped" in result.agent_trace[1]
