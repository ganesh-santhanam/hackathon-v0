import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from industrial_ai.paths import MVTEC_DATASET_DIR
from industrial_ai.vision.mvtec_compare import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PATCH_SIZE,
    DEFAULT_REFERENCE_LIMIT,
    load_image_vector,
    patch_anomaly_score,
    reference_image_paths,
    reference_vectors,
    threshold_for_category,
)


SKIP_DATASET_DIRS = {"To Avoid", "__pycache__"}


@dataclass(frozen=True)
class CategoryMetrics:
    category: str
    total: int
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class EvaluationReport:
    detector: str
    categories: list[CategoryMetrics]
    overall: CategoryMetrics


def active_categories(dataset_dir: Path = MVTEC_DATASET_DIR) -> list[str]:
    return [
        path.name
        for path in sorted(dataset_dir.iterdir())
        if path.is_dir()
        and path.name not in SKIP_DATASET_DIRS
        and (path / "train" / "good").exists()
        and (path / "test").exists()
    ]


def test_image_paths(category: str, dataset_dir: Path = MVTEC_DATASET_DIR) -> list[Path]:
    return sorted((dataset_dir / category / "test").glob("*/*.png"))


def compute_metrics(category: str, predictions: list[tuple[bool, bool]]) -> CategoryMetrics:
    tp = sum(1 for actual, predicted in predictions if actual and predicted)
    tn = sum(1 for actual, predicted in predictions if not actual and not predicted)
    fp = sum(1 for actual, predicted in predictions if not actual and predicted)
    fn = sum(1 for actual, predicted in predictions if actual and not predicted)
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return CategoryMetrics(
        category=category,
        total=total,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )


def evaluate_detector(
    detector_name: str,
    predict_defect: Callable[[Path, str], bool],
    dataset_dir: Path = MVTEC_DATASET_DIR,
    categories: list[str] | None = None,
) -> EvaluationReport:
    categories = categories or active_categories(dataset_dir)
    category_metrics = []
    overall_predictions = []

    for category in categories:
        predictions = []
        for image_path in test_image_paths(category, dataset_dir=dataset_dir):
            actual_defect = image_path.parent.name != "good"
            predicted_defect = predict_defect(image_path, category)
            predictions.append((actual_defect, predicted_defect))

        category_metrics.append(compute_metrics(category, predictions))
        overall_predictions.extend(predictions)

    return EvaluationReport(
        detector=detector_name,
        categories=category_metrics,
        overall=compute_metrics("overall", overall_predictions),
    )


def evaluate_comparison(
    dataset_dir: Path = MVTEC_DATASET_DIR,
    categories: list[str] | None = None,
    threshold: float | None = None,
    reference_limit: int = DEFAULT_REFERENCE_LIMIT,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> EvaluationReport:
    categories = categories or active_categories(dataset_dir)
    category_metrics = []
    overall_predictions = []

    for category in categories:
        references = reference_image_paths(category, dataset_dir=dataset_dir, limit=reference_limit)
        cached_reference_vectors = reference_vectors(references, image_size=image_size)
        predictions = []
        for image_path in test_image_paths(category, dataset_dir=dataset_dir):
            image_vector = load_image_vector(image_path, image_size=image_size)
            score = patch_anomaly_score(
                image_vector,
                cached_reference_vectors,
                image_size=image_size,
                patch_size=DEFAULT_PATCH_SIZE,
            )
            category_threshold = threshold_for_category(category, threshold)
            predictions.append((image_path.parent.name != "good", score >= category_threshold))

        category_metrics.append(compute_metrics(category, predictions))
        overall_predictions.extend(predictions)

    return EvaluationReport(
        detector="comparison",
        categories=category_metrics,
        overall=compute_metrics("overall", overall_predictions),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MVTec vision detectors by category.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    comparison_parser = subparsers.add_parser("comparison")
    comparison_parser.add_argument("--dataset-dir", default=MVTEC_DATASET_DIR, type=Path)
    comparison_parser.add_argument("--category", action="append", dest="categories")
    comparison_parser.add_argument("--threshold", type=float)
    comparison_parser.add_argument("--reference-limit", default=DEFAULT_REFERENCE_LIMIT, type=int)
    comparison_parser.add_argument("--image-size", default=DEFAULT_IMAGE_SIZE, type=int)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "comparison":
        report = evaluate_comparison(
            dataset_dir=args.dataset_dir,
            categories=args.categories,
            threshold=args.threshold,
            reference_limit=args.reference_limit,
            image_size=args.image_size,
        )
        print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
