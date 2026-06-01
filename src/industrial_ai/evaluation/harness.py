import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from industrial_ai.paths import EVALUATION_SCENARIOS_PATH
from industrial_ai.policy.severity import Severity, assign_severity


@dataclass(frozen=True)
class EvaluationScenario:
    scenario_id: str
    description: str
    failure_probability: float
    rag_confidence: str
    expected_severity: Severity


@dataclass(frozen=True)
class EvaluationResult:
    scenario_id: str
    description: str
    expected_severity: Severity
    actual_severity: Severity
    passed: bool
    reason: str


@dataclass(frozen=True)
class EvaluationSummary:
    total: int
    passed: int
    failed: int
    pass_rate: float
    results: list[EvaluationResult]


def load_scenarios(path: Path = EVALUATION_SCENARIOS_PATH) -> list[EvaluationScenario]:
    raw_scenarios = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvaluationScenario(
            scenario_id=item["scenario_id"],
            description=item["description"],
            failure_probability=float(item["failure_probability"]),
            rag_confidence=item["rag_confidence"],
            expected_severity=Severity(item["expected_severity"]),
        )
        for item in raw_scenarios
    ]


def evaluate_scenario(scenario: EvaluationScenario) -> EvaluationResult:
    decision = assign_severity(
        failure_probability=scenario.failure_probability,
        rag_confidence=scenario.rag_confidence,
    )
    passed = decision.severity == scenario.expected_severity
    return EvaluationResult(
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        expected_severity=scenario.expected_severity,
        actual_severity=decision.severity,
        passed=passed,
        reason=decision.reason,
    )


def evaluate_scenarios(scenarios: list[EvaluationScenario]) -> EvaluationSummary:
    results = [evaluate_scenario(scenario) for scenario in scenarios]
    passed = sum(result.passed for result in results)
    total = len(results)
    return EvaluationSummary(
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=passed / total if total else 0.0,
        results=results,
    )


def summary_to_dict(summary: EvaluationSummary) -> dict[str, Any]:
    return asdict(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic evaluation scenarios.")
    parser.add_argument("--scenarios-path", default=EVALUATION_SCENARIOS_PATH, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = evaluate_scenarios(load_scenarios(args.scenarios_path))
    print(json.dumps(summary_to_dict(summary), indent=2))
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
