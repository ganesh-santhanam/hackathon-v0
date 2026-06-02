from pathlib import Path

from PIL import Image

from industrial_ai.vision.evaluate import active_categories, compute_metrics, evaluate_comparison


def save_gray_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=value).save(path)


def test_compute_metrics_reports_binary_classification_scores():
    metrics = compute_metrics(
        "cable",
        [
            (True, True),
            (True, False),
            (False, False),
            (False, True),
        ],
    )

    assert metrics.total == 4
    assert metrics.tp == 1
    assert metrics.tn == 1
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5


def test_active_categories_ignores_to_avoid_folder(tmp_path):
    dataset_dir = tmp_path / "mvtec"
    save_gray_image(dataset_dir / "cable" / "train" / "good" / "000.png", 0)
    save_gray_image(dataset_dir / "cable" / "test" / "good" / "000.png", 0)
    save_gray_image(dataset_dir / "To Avoid" / "bottle" / "train" / "good" / "000.png", 0)

    assert active_categories(dataset_dir) == ["cable"]


def test_evaluate_comparison_reports_per_category_and_overall(tmp_path):
    dataset_dir = tmp_path / "mvtec"
    save_gray_image(dataset_dir / "cable" / "train" / "good" / "000.png", 0)
    save_gray_image(dataset_dir / "cable" / "test" / "good" / "000.png", 0)
    save_gray_image(dataset_dir / "cable" / "test" / "bent_wire" / "000.png", 255)

    report = evaluate_comparison(
        dataset_dir=dataset_dir,
        categories=["cable"],
        threshold=0.5,
        image_size=16,
        reference_limit=1,
    )

    assert report.detector == "comparison"
    assert report.categories[0].category == "cable"
    assert report.categories[0].accuracy == 1.0
    assert report.overall.total == 2
    assert report.overall.f1 == 1.0
