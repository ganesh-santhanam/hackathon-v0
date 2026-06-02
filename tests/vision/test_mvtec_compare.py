from pathlib import Path

from PIL import Image

from industrial_ai.vision.mvtec_compare import (
    compare_mvtec_image,
    confidence_from_score,
    infer_category,
    infer_defect_type,
    reference_image_paths,
    threshold_for_category,
)


def save_gray_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=value).save(path)


def test_infer_category_from_mvtec_path(tmp_path):
    dataset_dir = tmp_path / "mvtec"
    image_path = dataset_dir / "bottle" / "test" / "good" / "000.png"
    save_gray_image(image_path, 0)

    assert infer_category(image_path, dataset_dir) == "bottle"


def test_infer_defect_type_from_test_path(tmp_path):
    defect_path = tmp_path / "bottle" / "test" / "scratch" / "000.png"
    good_path = tmp_path / "bottle" / "test" / "good" / "000.png"

    assert infer_defect_type(defect_path) == "scratch"
    assert infer_defect_type(good_path) is None


def test_reference_image_paths_finds_good_training_images(tmp_path):
    dataset_dir = tmp_path / "mvtec"
    save_gray_image(dataset_dir / "bottle" / "train" / "good" / "001.png", 0)

    references = reference_image_paths("bottle", dataset_dir)

    assert len(references) == 1
    assert references[0].name == "001.png"


def test_threshold_for_category_uses_calibrated_industrial_threshold():
    assert threshold_for_category("cable") == 0.168
    assert threshold_for_category("unknown") == 0.10
    assert threshold_for_category("cable", threshold=0.25) == 0.25


def test_confidence_is_not_maxed_out_near_threshold():
    assert confidence_from_score(0.17, 0.168, defect_detected=True) < 0.6
    assert confidence_from_score(0.30, 0.168, defect_detected=True) <= 0.95


def test_compare_mvtec_image_detects_bright_defect(tmp_path):
    dataset_dir = tmp_path / "mvtec"
    save_gray_image(dataset_dir / "bottle" / "train" / "good" / "000.png", 0)
    image_path = dataset_dir / "bottle" / "test" / "scratch" / "000.png"
    save_gray_image(image_path, 255)

    result = compare_mvtec_image(
        image_path=image_path,
        dataset_dir=dataset_dir,
        threshold=0.5,
        image_size=16,
    )

    assert result.defect_detected is True
    assert result.defect_type == "scratch"
    assert result.confidence == 0.95
    assert result.anomaly_score == 1.0


def test_compare_mvtec_image_marks_matching_good_image_as_not_defective(tmp_path):
    dataset_dir = tmp_path / "mvtec"
    save_gray_image(dataset_dir / "bottle" / "train" / "good" / "000.png", 10)
    image_path = dataset_dir / "bottle" / "test" / "good" / "000.png"
    save_gray_image(image_path, 10)

    result = compare_mvtec_image(
        image_path=image_path,
        dataset_dir=dataset_dir,
        threshold=0.5,
        image_size=16,
    )

    assert result.defect_detected is False
    assert result.defect_type is None
    assert result.confidence == 0.95
    assert result.anomaly_score == 0.0
