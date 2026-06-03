from industrial_ai.demo.streamlit_app import (
    DEMO_SCENARIOS,
    build_vision_upload_path,
    evaluation_rows,
    format_key_inputs,
    pass_rate,
    policy_summary,
    rag_metadata_display,
    severity_rule_rows,
)
from industrial_ai.evaluation.test_rig import RigCaseResult, RigReport


def make_report() -> RigReport:
    return RigReport(
        total=2,
        passed=1,
        failed=1,
        results=[
            RigCaseResult(
                area="telemetry",
                scenario="heldout_positive_high_risk",
                passed=True,
                expected="failure",
                actual="HIGH 0.996",
                details={"probability": 0.995837, "risk_level": "HIGH", "heldout_target": 1},
            ),
            RigCaseResult(
                area="vision",
                scenario="grid_defect_image",
                passed=False,
                expected="defect_detected=True",
                actual="defect_detected=False",
                details={
                    "image_path": "/tmp/grid/test/bent/000.png",
                    "score": 0.1767,
                    "threshold": 0.3568,
                    "confidence": 0.95,
                    "defect_type": "bent",
                },
            ),
        ],
    )


def test_pass_rate_handles_report_counts():
    assert pass_rate(make_report()) == 0.5
    assert pass_rate(RigReport(total=0, passed=0, failed=0, results=[])) == 0.0


def test_severity_rule_rows_are_loaded_from_policy_source():
    rows = severity_rule_rows()

    assert rows[0] == {
        "severity": "SEV1",
        "criteria": "Failure probability > 80% and visual defect detected",
        "approval_required": "Yes",
    }
    assert rows[-1] == {
        "severity": "SEV3",
        "criteria": "Failure probability <= 50%",
        "approval_required": "No",
    }


def test_policy_summary_is_loaded_from_policy_source():
    assert policy_summary() == {
        "name": "Tier 0 Severity Policy",
        "version": "1.0.0",
        "last_modified": "2026-06-02T00:00:00Z",
    }


def test_format_key_inputs_prioritizes_scenario_details():
    formatted = format_key_inputs(
        {
            "image_path": "/tmp/grid/test/bent/000.png",
            "score": 0.17674,
            "threshold": 0.35675,
            "confidence": 0.95,
            "defect_type": "bent",
        }
    )

    assert formatted == (
        "image_path=000.png, score=0.1767, threshold=0.3568, "
        "confidence=0.95, defect_type=bent"
    )


def test_evaluation_rows_filters_all_passed_and_failed():
    report = make_report()

    assert len(evaluation_rows(report, "All")) == 2
    assert [row["scenario"] for row in evaluation_rows(report, "Passed")] == [
        "heldout_positive_high_risk"
    ]
    assert [row["scenario"] for row in evaluation_rows(report, "Failed")] == [
        "grid_defect_image"
    ]


def test_build_vision_upload_path_sanitizes_user_controlled_machine_id(tmp_path):
    path = build_vision_upload_path(
        machine_id="../../tmp/pwn",
        vision_category="cable",
        uploaded_filename="inspection.png",
        upload_dir=tmp_path,
    )

    assert path.parent == tmp_path
    assert path.name == "tmp_pwn-cable.png"


def test_build_vision_upload_path_sanitizes_suffix_and_category(tmp_path):
    path = build_vision_upload_path(
        machine_id="/tmp/pwn",
        vision_category="../grid",
        uploaded_filename="../../payload.exe",
        upload_dir=tmp_path,
    )

    assert path.parent == tmp_path
    assert path.name == "tmp_pwn-grid.png"


def test_rag_metadata_display_supports_new_and_stale_metadata_shapes():
    class OldMetadata:
        mode = "llm"
        provider = "Ollama"
        model_name = "gemma3:4b"
        endpoint_url = "http://localhost:11434/api/generate"
        fallback_used = True
        error_message = "connection refused"

    assert rag_metadata_display(OldMetadata()) == {
        "rag_mode": "llm",
        "llm_provider": "Ollama",
        "llm_model": "gemma3:4b",
        "endpoint_url": "http://localhost:11434/api/generate",
        "fallback_used": True,
        "fallback_reason": None,
        "latency_ms": None,
        "raw_error": "connection refused",
    }


def test_demo_scenarios_define_required_buttons_and_telemetry_inputs():
    assert set(DEMO_SCENARIOS) == {
        "tool_wear_failure",
        "power_failure",
        "cooling_failure",
        "visual_defect",
        "multi_modal_sev1",
    }
    for scenario in DEMO_SCENARIOS.values():
        assert scenario["label"].startswith("Injected")
        assert scenario["machine_id"]
        assert scenario["machine_type"] in {"L", "M", "H"}
        assert scenario["air_temperature_k"] > 0
        assert scenario["process_temperature_k"] > 0
        assert scenario["rotational_speed_rpm"] > 0
        assert scenario["torque_nm"] > 0
        assert scenario["tool_wear_min"] >= 0


def test_visual_demo_scenarios_point_to_available_local_images():
    for scenario_key in ["visual_defect", "multi_modal_sev1"]:
        image_path = DEMO_SCENARIOS[scenario_key]["vision_image_path"]

        assert image_path.exists()
        assert image_path.suffix == ".png"
        assert DEMO_SCENARIOS[scenario_key]["vision_enabled"] is True
