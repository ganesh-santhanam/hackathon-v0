import numpy as np

from industrial_ai.telemetry.predict import (
    RiskLevel,
    TelemetryReading,
    build_evidence,
    classify_risk,
    get_top_feature_importances,
)


def test_telemetry_reading_converts_to_model_frame():
    reading = TelemetryReading(
        machine_id="FAN-023",
        type="M",
        air_temperature_k=298.1,
        process_temperature_k=308.6,
        rotational_speed_rpm=1551,
        torque_nm=42.8,
        tool_wear_min=0,
    )

    frame = reading.to_frame()

    assert list(frame.columns) == [
        "type",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ]
    assert frame.iloc[0]["type"] == "M"


def test_classify_risk_uses_demo_bands():
    assert classify_risk(0.7) == RiskLevel.HIGH
    assert classify_risk(0.5) == RiskLevel.MEDIUM
    assert classify_risk(0.49) == RiskLevel.LOW


def test_build_evidence_flags_high_wear_torque_and_speed():
    reading = TelemetryReading(
        machine_id="FAN-023",
        type="L",
        air_temperature_k=302.0,
        process_temperature_k=310.0,
        rotational_speed_rpm=1200,
        torque_nm=65.0,
        tool_wear_min=220,
    )

    assert build_evidence(reading) == [
        "Tool wear unusually high",
        "Torque outside normal range",
        "Rotational speed anomaly",
    ]


def test_get_top_feature_importances_returns_ranked_features():
    class FakePreprocessor:
        def get_feature_names_out(self):
            return np.array(["categorical__type_L", "numeric__torque_nm", "numeric__tool_wear_min"])

    class FakeClassifier:
        feature_importances_ = np.array([0.2, 0.5, 0.3])

    class FakeModel:
        named_steps = {
            "preprocessor": FakePreprocessor(),
            "classifier": FakeClassifier(),
        }

    assert get_top_feature_importances(FakeModel(), limit=2) == [
        {"feature": "torque_nm", "importance": 0.5},
        {"feature": "tool_wear_min", "importance": 0.3},
    ]
