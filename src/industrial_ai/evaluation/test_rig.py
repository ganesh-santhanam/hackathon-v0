import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

from industrial_ai.approvals.approval import create_approval
from industrial_ai.paths import MODELS_DIR, MVTEC_DATASET_DIR
from industrial_ai.policy.severity import assign_severity
from industrial_ai.telemetry.ai4i import load_ai4i_dataset
from industrial_ai.telemetry.predict import TelemetryReading, classify_risk, load_telemetry_model
from industrial_ai.vision.mvtec_resnet import model_path_for_category, predict_resnet


@dataclass(frozen=True)
class RigCaseResult:
    area: str
    scenario: str
    passed: bool
    expected: str
    actual: str
    details: dict


@dataclass(frozen=True)
class RigReport:
    total: int
    passed: int
    failed: int
    results: list[RigCaseResult]


def result(area: str, scenario: str, passed: bool, expected: str, actual: str, **details):
    return RigCaseResult(
        area=area,
        scenario=scenario,
        passed=passed,
        expected=expected,
        actual=actual,
        details=details,
    )


def telemetry_reading_from_row(machine_id: str, row) -> TelemetryReading:
    return TelemetryReading(
        machine_id=machine_id,
        type=str(row["type"]),
        air_temperature_k=float(row["air_temperature_k"]),
        process_temperature_k=float(row["process_temperature_k"]),
        rotational_speed_rpm=float(row["rotational_speed_rpm"]),
        torque_nm=float(row["torque_nm"]),
        tool_wear_min=float(row["tool_wear_min"]),
    )


def telemetry_cases() -> list[RigCaseResult]:
    dataset = load_ai4i_dataset()
    _x_train, x_test, _y_train, y_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )
    model = load_telemetry_model()
    probabilities = model.predict_proba(x_test)[:, 1]
    positive_positions = [index for index, target in enumerate(y_test.tolist()) if target == 1]
    negative_positions = [index for index, target in enumerate(y_test.tolist()) if target == 0]
    high_positive_position = max(positive_positions, key=lambda index: probabilities[index])
    low_negative_position = min(negative_positions, key=lambda index: probabilities[index])

    cases = []
    for name, position, expected_label in [
        ("heldout_positive_high_risk", high_positive_position, 1),
        ("heldout_negative_low_risk", low_negative_position, 0),
    ]:
        row = x_test.iloc[position]
        probability = float(probabilities[position])
        risk = classify_risk(probability)
        expected = "failure" if expected_label else "normal"
        actual = f"{risk.value} {probability:.3f}"
        passed = probability >= 0.7 if expected_label else probability < 0.5
        cases.append(
            result(
                area="telemetry",
                scenario=name,
                passed=passed,
                expected=expected,
                actual=actual,
                machine_id=telemetry_reading_from_row(name, row).machine_id,
                probability=probability,
                risk_level=risk.value,
                heldout_target=int(y_test.iloc[position]),
            )
        )
    return cases


def first_defect_image(category: str, dataset_dir: Path = MVTEC_DATASET_DIR) -> Path | None:
    for path in sorted((dataset_dir / category / "test").glob("*/*.png")):
        if path.parent.name != "good":
            return path
    return None


def first_good_image(category: str, dataset_dir: Path = MVTEC_DATASET_DIR) -> Path | None:
    images = sorted((dataset_dir / category / "test" / "good").glob("*.png"))
    return images[0] if images else None


def vision_cases(categories: list[str], models_dir: Path = MODELS_DIR) -> list[RigCaseResult]:
    cases = []
    for category in categories:
        model_path = model_path_for_category(category, models_dir=models_dir)
        if not model_path.exists():
            cases.append(
                result(
                    area="vision",
                    scenario=f"{category}_profile_exists",
                    passed=False,
                    expected="calibrated ResNet profile exists",
                    actual=f"missing {model_path}",
                )
            )
            continue

        for label, image_path, expected_defect in [
            ("good_image", first_good_image(category), False),
            ("defect_image", first_defect_image(category), True),
        ]:
            if image_path is None:
                cases.append(
                    result(
                        area="vision",
                        scenario=f"{category}_{label}",
                        passed=False,
                        expected="test image exists",
                        actual="missing",
                    )
                )
                continue
            prediction = predict_resnet(image_path=image_path, model_path=model_path)
            cases.append(
                result(
                    area="vision",
                    scenario=f"{category}_{label}",
                    passed=prediction.defect_detected is expected_defect,
                    expected=f"defect_detected={expected_defect}",
                    actual=f"defect_detected={prediction.defect_detected}",
                    image_path=str(image_path),
                    score=prediction.anomaly_score,
                    threshold=prediction.threshold,
                    confidence=prediction.confidence,
                    defect_type=prediction.defect_type,
                )
            )
    return cases


def policy_cases() -> list[RigCaseResult]:
    cases = []
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "approvals.json"
        scenarios = [
            ("High telemetry + defect", 0.81, True, "SEV1"),
            ("High telemetry + no defect", 0.81, False, "SEV2"),
            ("Low telemetry + defect", 0.2, True, "SEV3"),
            ("Low telemetry + no defect", 0.2, False, "Normal"),
        ]
        for scenario, probability, visual_defect_detected, expected in scenarios:
            decision = assign_severity(
                failure_probability=probability,
                rag_confidence="medium",
                visual_defect_detected=visual_defect_detected,
            )
            approval = create_approval(
                f"RIG-{scenario.upper().replace(' ', '-')}",
                severity=decision.severity.value,
                store_path=store_path,
            )
            actual = "Normal" if decision.severity.value == "SEV3" and not visual_defect_detected else (
                decision.severity.value
            )
            cases.append(
                result(
                    area="rules",
                    scenario=scenario,
                    passed=actual == expected,
                    expected=expected,
                    actual=actual,
                    failure_probability=probability,
                    visual_defect_detected=visual_defect_detected,
                    severity=decision.severity.value,
                    approval_required=approval.approval_required,
                    approval_status=approval.status.value,
                    reason=decision.reason,
                )
            )
    return cases


def run_rig(categories: list[str], models_dir: Path = MODELS_DIR) -> RigReport:
    results = [
        *telemetry_cases(),
        *vision_cases(categories=categories, models_dir=models_dir),
        *policy_cases(),
    ]
    passed = sum(1 for item in results if item.passed)
    return RigReport(
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        results=results,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run held-out demo correctness scenarios.")
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="MVTec category to test. Repeat for multiple categories. Defaults to cable.",
    )
    parser.add_argument("--models-dir", default=MODELS_DIR, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_rig(categories=args.categories or ["cable"], models_dir=args.models_dir)
    print(json.dumps(asdict(report), indent=2))
    if report.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
