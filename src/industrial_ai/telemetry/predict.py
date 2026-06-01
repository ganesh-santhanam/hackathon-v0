import argparse
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import joblib
import pandas as pd

from industrial_ai.paths import TELEMETRY_MODEL_PATH
from industrial_ai.telemetry.ai4i import FEATURE_COLUMNS


HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.5


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class TelemetryReading:
    machine_id: str
    type: str
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{column: getattr(self, column) for column in FEATURE_COLUMNS}])


@dataclass(frozen=True)
class FailurePrediction:
    machine_id: str
    failure_probability: float
    failure_probability_percent: int
    risk_level: RiskLevel
    top_feature_importances: list[dict[str, float | str]]
    evidence: list[str]


def classify_risk(probability: float) -> RiskLevel:
    if probability >= HIGH_RISK_THRESHOLD:
        return RiskLevel.HIGH
    if probability >= MEDIUM_RISK_THRESHOLD:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def build_evidence(reading: TelemetryReading) -> list[str]:
    evidence = []
    if reading.tool_wear_min >= 195:
        evidence.append("Tool wear unusually high")
    if reading.torque_nm >= 52.6 or reading.torque_nm <= 27.2:
        evidence.append("Torque outside normal range")
    if reading.rotational_speed_rpm <= 1364 or reading.rotational_speed_rpm >= 1746:
        evidence.append("Rotational speed anomaly")
    if reading.air_temperature_k >= 302.7:
        evidence.append("Air temperature elevated")
    if reading.process_temperature_k >= 311.9:
        evidence.append("Process temperature elevated")
    if not evidence:
        evidence.append("Telemetry values within learned baseline ranges")
    return evidence


def load_telemetry_model(model_path: Path = TELEMETRY_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Telemetry model not found at {model_path}. "
            "Run: PYTHONPATH=src .venv/bin/python -m industrial_ai.telemetry.train"
        )
    return joblib.load(model_path)


def get_top_feature_importances(model, limit: int = 3) -> list[dict[str, float | str]]:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    ranked = sorted(
        zip(feature_names, importances, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {
            "feature": str(feature).replace("categorical__", "").replace("numeric__", ""),
            "importance": float(importance),
        }
        for feature, importance in ranked[:limit]
    ]


def predict_failure(
    reading: TelemetryReading,
    model_path: Path = TELEMETRY_MODEL_PATH,
) -> FailurePrediction:
    model = load_telemetry_model(model_path)
    probability = float(model.predict_proba(reading.to_frame())[0, 1])
    return FailurePrediction(
        machine_id=reading.machine_id,
        failure_probability=probability,
        failure_probability_percent=round(probability * 100),
        risk_level=classify_risk(probability),
        top_feature_importances=get_top_feature_importances(model),
        evidence=build_evidence(reading),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict AI4I machine failure risk.")
    parser.add_argument("--machine-id", default="MACHINE-001")
    parser.add_argument("--type", required=True, choices=["L", "M", "H"])
    parser.add_argument("--air-temperature-k", required=True, type=float)
    parser.add_argument("--process-temperature-k", required=True, type=float)
    parser.add_argument("--rotational-speed-rpm", required=True, type=float)
    parser.add_argument("--torque-nm", required=True, type=float)
    parser.add_argument("--tool-wear-min", required=True, type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reading = TelemetryReading(
        machine_id=args.machine_id,
        type=args.type,
        air_temperature_k=args.air_temperature_k,
        process_temperature_k=args.process_temperature_k,
        rotational_speed_rpm=args.rotational_speed_rpm,
        torque_nm=args.torque_nm,
        tool_wear_min=args.tool_wear_min,
    )
    prediction = predict_failure(reading=reading)
    print(json.dumps(asdict(prediction), indent=2))


if __name__ == "__main__":
    main()
