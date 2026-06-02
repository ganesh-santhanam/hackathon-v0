from industrial_ai.demo import investigation as investigation_module
from industrial_ai.demo.investigation import (
    VisionCheck,
    build_evidence_items,
    build_incident_id,
    build_retrieval_query,
    run_investigation,
    run_vision_check,
)
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


def make_prediction(
    failure_probability=0.82,
    risk_level=RiskLevel.HIGH,
):
    return FailurePrediction(
        machine_id="FAN-023",
        failure_probability=failure_probability,
        failure_probability_percent=round(failure_probability * 100),
        risk_level=risk_level,
        top_feature_importances=[],
        evidence=[
            "Tool wear unusually high",
            "Torque outside normal range",
            "Rotational speed anomaly",
        ],
    )


def defect_vision():
    return VisionCheck(
        method="resnet",
        category="cable",
        defect_detected=True,
        defect_type="bent_wire",
        confidence=0.91,
        anomaly_score=0.31,
        threshold=0.25,
        evidence=["Embedding distance anomaly score: 0.3100"],
    )


def no_defect_vision():
    return VisionCheck(
        method="resnet",
        category="cable",
        defect_detected=False,
        defect_type=None,
        confidence=0.74,
        anomaly_score=0.12,
        threshold=0.2,
        evidence=["Embedding distance below threshold"],
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


def install_investigation_stubs(monkeypatch, prediction, vision=None, retrieval_results=None):
    monkeypatch.setattr(investigation_module, "predict_failure", lambda _: prediction)
    monkeypatch.setattr(investigation_module, "load_embedder", lambda _: object())
    if vision is not None:
        monkeypatch.setattr(investigation_module, "run_vision_check", lambda **_: vision)

    def fake_retrieve_incidents(**kwargs):
        assert kwargs["telemetry_query"].tool_wear_min == 210
        assert kwargs["telemetry_query"].torque_nm == 55.5
        return SearchResponse(
            query=kwargs["query"],
            top_k=kwargs["top_k"],
            score_threshold=kwargs["score_threshold"],
            top_score=0.72 if retrieval_results is None else None,
            message="Relevant incidents found" if retrieval_results is None else "No relevant incidents found",
            results=[make_search_result()] if retrieval_results is None else retrieval_results,
        )

    monkeypatch.setattr(investigation_module, "retrieve_incidents", fake_retrieve_incidents)


def test_build_retrieval_query_uses_normalized_telemetry_evidence():
    assert build_retrieval_query(make_prediction()) == (
        "Telemetry failure probability is 82% (HIGH)"
    )


def test_build_retrieval_query_adds_vision_evidence_when_available():
    query = build_retrieval_query(make_prediction(), defect_vision())

    assert "Telemetry failure probability is 82% (HIGH)" in query
    assert "Visual inspection detected bent_wire in cable" in query


def test_build_evidence_items_normalizes_telemetry_and_vision():
    evidence = build_evidence_items(make_prediction(), defect_vision())

    assert evidence[0].source == "telemetry"
    assert evidence[0].status == "HIGH"
    assert evidence[1].source == "vision"
    assert evidence[1].status == "defect_detected"
    assert "Embedding distance anomaly score: 0.3100" in evidence[1].details


def test_build_incident_id_is_stable_for_machine():
    assert build_incident_id("FAN-023") == "FAN-023-INVESTIGATION"


def test_high_telemetry_plus_defect_becomes_sev1_and_requires_approval(monkeypatch, tmp_path):
    install_investigation_stubs(monkeypatch, make_prediction(), defect_vision())

    result = run_investigation(
        reading=make_reading(),
        vision_image_path=tmp_path / "image.png",
        vision_category="cable",
        approvals_store_path=tmp_path / "approvals.json",
    )

    assert result.vision is not None
    assert result.evidence[1].status == "defect_detected"
    assert result.severity.severity.value == "SEV1"
    assert result.approval.approval_required is True
    assert result.approval.status.value == "pending"


def test_low_telemetry_plus_defect_stays_sev3(monkeypatch, tmp_path):
    install_investigation_stubs(
        monkeypatch,
        make_prediction(failure_probability=0.2, risk_level=RiskLevel.LOW),
        defect_vision(),
    )

    result = run_investigation(
        reading=make_reading(),
        vision_image_path=tmp_path / "image.png",
        vision_category="cable",
        approvals_store_path=tmp_path / "approvals.json",
    )

    assert result.severity.severity.value == "SEV3"
    assert result.approval.approval_required is False
    assert result.approval.status.value == "not_required"


def test_high_telemetry_plus_no_defect_stays_sev2_without_high_rag(monkeypatch, tmp_path):
    install_investigation_stubs(
        monkeypatch,
        make_prediction(),
        no_defect_vision(),
        retrieval_results=[],
    )

    result = run_investigation(
        reading=make_reading(),
        vision_image_path=tmp_path / "image.png",
        vision_category="cable",
        approvals_store_path=tmp_path / "approvals.json",
    )

    assert result.vision is not None
    assert result.vision.defect_detected is False
    assert result.evidence[1].status == "no_defect_detected"
    assert result.severity.severity.value == "SEV2"
    assert result.approval.approval_required is False


def test_no_vision_input_uses_telemetry_only(monkeypatch, tmp_path):
    install_investigation_stubs(
        monkeypatch,
        make_prediction(failure_probability=0.2, risk_level=RiskLevel.LOW),
    )

    result = run_investigation(
        reading=make_reading(),
        approvals_store_path=tmp_path / "approvals.json",
    )

    assert result.vision is None
    assert [item.source for item in result.evidence] == ["telemetry"]
    assert result.severity.severity.value == "SEV3"
    assert result.approval.approval_required is False


def test_auto_vision_requires_resnet_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(
        investigation_module,
        "model_path_for_category",
        lambda category: tmp_path / f"mvtec_resnet_{category}.npz",
    )

    try:
        run_vision_check(tmp_path / "image.png", "cable", method="auto")
    except FileNotFoundError as exc:
        assert "No calibrated ResNet profile found" in str(exc)
    else:
        raise AssertionError("Expected missing ResNet profile to fail auto vision")
