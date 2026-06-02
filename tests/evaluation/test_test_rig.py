from industrial_ai.evaluation import test_rig
from industrial_ai.evaluation.test_rig import RigCaseResult, policy_cases, run_rig


def make_case(area: str, scenario: str, passed: bool) -> RigCaseResult:
    return RigCaseResult(
        area=area,
        scenario=scenario,
        passed=passed,
        expected="expected",
        actual="actual",
        details={},
    )


def test_run_rig_aggregates_results(monkeypatch, tmp_path):
    monkeypatch.setattr(test_rig, "telemetry_cases", lambda: [make_case("telemetry", "a", True)])
    monkeypatch.setattr(
        test_rig,
        "vision_cases",
        lambda categories, models_dir: [make_case("vision", categories[0], False)],
    )
    monkeypatch.setattr(test_rig, "policy_cases", lambda: [make_case("rules", "c", True)])

    report = run_rig(categories=["cable"], models_dir=tmp_path)

    assert report.total == 3
    assert report.passed == 2
    assert report.failed == 1


def test_policy_cases_match_expected_rule_matrix():
    cases = policy_cases()

    assert [(case.scenario, case.expected, case.actual, case.passed) for case in cases] == [
        ("High telemetry + defect", "SEV1", "SEV1", True),
        ("High telemetry + no defect", "SEV2", "SEV2", True),
        ("Low telemetry + defect", "SEV3", "SEV3", True),
        ("Low telemetry + no defect", "Normal", "Normal", True),
    ]
