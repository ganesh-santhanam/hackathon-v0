import json

from industrial_ai.evaluation.harness import (
    EvaluationScenario,
    evaluate_scenario,
    evaluate_scenarios,
    load_scenarios,
    main,
)
from industrial_ai.policy.severity import Severity


def test_load_scenarios_reads_json_file(tmp_path):
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps(
            [
                {
                    "scenario_id": "example",
                    "description": "Example scenario",
                    "failure_probability": 0.82,
                    "rag_confidence": "high",
                    "expected_severity": "SEV1",
                }
            ]
        ),
        encoding="utf-8",
    )

    scenarios = load_scenarios(scenarios_path)

    assert len(scenarios) == 1
    assert scenarios[0].expected_severity == Severity.SEV1


def test_evaluate_scenario_passes_when_expected_matches_actual():
    scenario = EvaluationScenario(
        scenario_id="sev1_case",
        description="SEV1 case",
        failure_probability=0.82,
        rag_confidence="high",
        expected_severity=Severity.SEV1,
    )

    result = evaluate_scenario(scenario)

    assert result.actual_severity == Severity.SEV1
    assert result.passed is True


def test_evaluate_scenario_fails_when_expected_differs_from_actual():
    scenario = EvaluationScenario(
        scenario_id="wrong_expectation",
        description="Wrong expectation",
        failure_probability=0.82,
        rag_confidence="medium",
        expected_severity=Severity.SEV1,
    )

    result = evaluate_scenario(scenario)

    assert result.actual_severity == Severity.SEV2
    assert result.passed is False


def test_evaluate_scenarios_returns_summary_counts():
    scenarios = [
        EvaluationScenario(
            scenario_id="pass",
            description="Pass",
            failure_probability=0.82,
            rag_confidence="high",
            expected_severity=Severity.SEV1,
        ),
        EvaluationScenario(
            scenario_id="fail",
            description="Fail",
            failure_probability=0.2,
            rag_confidence="none",
            expected_severity=Severity.SEV1,
        ),
    ]

    summary = evaluate_scenarios(scenarios)

    assert summary.total == 2
    assert summary.passed == 1
    assert summary.failed == 1
    assert summary.pass_rate == 0.5


def test_main_returns_zero_when_all_scenarios_pass(tmp_path, monkeypatch, capsys):
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps(
            [
                {
                    "scenario_id": "pass",
                    "description": "Pass",
                    "failure_probability": 0.82,
                    "rag_confidence": "high",
                    "expected_severity": "SEV1",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["harness", "--scenarios-path", str(scenarios_path)])

    exit_code = main()

    assert exit_code == 0
    assert '"failed": 0' in capsys.readouterr().out


def test_main_returns_nonzero_when_any_scenario_fails(tmp_path, monkeypatch, capsys):
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps(
            [
                {
                    "scenario_id": "fail",
                    "description": "Fail",
                    "failure_probability": 0.2,
                    "rag_confidence": "none",
                    "expected_severity": "SEV1",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["harness", "--scenarios-path", str(scenarios_path)])

    exit_code = main()

    assert exit_code == 1
    assert '"failed": 1' in capsys.readouterr().out
