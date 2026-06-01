import numpy as np
import pandas as pd

from industrial_ai.telemetry.train import evaluate_thresholds


def test_evaluate_thresholds_reports_confusion_matrix_counts():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.4, 0.9])

    metrics = evaluate_thresholds(y_true, probabilities, thresholds=(0.5,))

    assert metrics["0.5"]["confusion_matrix"] == {
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "tp": 1,
    }
    assert metrics["0.5"]["precision"] == 0.5
    assert metrics["0.5"]["recall"] == 0.5
    assert metrics["0.5"]["f1"] == 0.5

