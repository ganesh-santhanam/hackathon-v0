from industrial_ai.demo.streamlit_app import (
    build_vision_upload_path,
    evaluation_rows,
    format_key_inputs,
    pass_rate,
    policy_summary,
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
