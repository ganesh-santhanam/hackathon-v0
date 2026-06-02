from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from industrial_ai.vision.mvtec_resnet import (  # noqa: E402
    anomaly_score,
    calibrate_resnet_threshold,
    load_resnet_profile,
    metric_value,
    model_path_for_category,
    profile_anomaly_score,
)
from industrial_ai.vision.evaluate import compute_metrics  # noqa: E402


def test_model_path_for_category_uses_models_dir(tmp_path):
    assert model_path_for_category("cable", models_dir=tmp_path) == tmp_path / "mvtec_resnet_cable.npz"


def test_anomaly_score_is_embedding_distance():
    score = anomaly_score(
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    )

    assert score == 1.0


def test_metric_value_supports_balanced_accuracy():
    metrics = compute_metrics(
        "cable",
        [
            (True, True),
            (True, False),
            (False, False),
            (False, False),
        ],
    )

    assert metric_value(metrics, "balanced_accuracy") == 0.75


def test_load_resnet_profile_reads_saved_npz(tmp_path: Path):
    model_path = tmp_path / "profile.npz"
    np.savez(
        model_path,
        category="cable",
        image_size=128,
        threshold=0.25,
        threshold_percentile=95.0,
        pretrained=False,
        normal_center=np.array([1.0, 0.0], dtype=np.float32),
    )

    profile = load_resnet_profile(model_path)

    assert profile["category"] == "cable"
    assert profile["image_size"] == 128
    assert profile["threshold"] == 0.25
    assert profile["pretrained"] is False
    assert profile["normal_center"].tolist() == [1.0, 0.0]
    assert profile["normal_embeddings"] is None


def test_profile_anomaly_score_uses_nearest_normal_embedding():
    profile = {
        "normal_center": np.array([0.0, 0.0], dtype=np.float32),
        "normal_embeddings": np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }

    score = profile_anomaly_score(np.array([0.9, 0.0], dtype=np.float32), profile)

    assert round(score, 3) == 0.1


def test_calibrate_resnet_threshold_selects_and_saves_best_threshold(monkeypatch, tmp_path):
    model_path = tmp_path / "profile.npz"
    np.savez(
        model_path,
        category="cable",
        image_size=128,
        threshold=0.9,
        threshold_percentile=95.0,
        pretrained=False,
        normal_center=np.array([1.0, 0.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "industrial_ai.vision.mvtec_resnet.resnet_test_scores",
        lambda *_args, **_kwargs: [
            (True, 0.8),
            (True, 0.7),
            (False, 0.1),
            (False, 0.2),
        ],
    )

    result = calibrate_resnet_threshold(model_path, metric="f1")
    profile = load_resnet_profile(model_path)

    assert result.threshold == 0.7
    assert result.metrics["f1"] == 1.0
    assert profile["threshold"] == 0.7
