import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from industrial_ai.paths import MODELS_DIR, MVTEC_DATASET_DIR
from industrial_ai.vision.evaluate import active_categories, compute_metrics, evaluate_detector, test_image_paths
from industrial_ai.vision.mvtec_compare import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_REFERENCE_LIMIT,
    confidence_from_score,
    infer_defect_type,
    reference_image_paths,
)


DEFAULT_THRESHOLD_PERCENTILE = 95.0
DEFAULT_CALIBRATION_METRIC = "balanced_accuracy"


@dataclass(frozen=True)
class ResNetResult:
    image_path: str
    category: str
    defect_detected: bool
    defect_type: str | None
    confidence: float
    anomaly_score: float
    threshold: float
    evidence: list[str]


@dataclass(frozen=True)
class CalibrationResult:
    category: str
    model_path: str
    metric: str
    threshold: float
    metrics: dict


def require_resnet_dependencies():
    try:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ResNet vision requires torch and torchvision. Install torchvision first."
        ) from exc
    return torch, ResNet18_Weights, resnet18


def build_feature_extractor(pretrained: bool = True):
    torch, ResNet18_Weights, resnet18 = require_resnet_dependencies()
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model


def load_resnet_input(image_path: Path, image_size: int = DEFAULT_IMAGE_SIZE):
    torch, _, _ = require_resnet_dependencies()
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def image_embedding(model, image_path: Path, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    torch, _, _ = require_resnet_dependencies()
    with torch.no_grad():
        tensor = load_resnet_input(image_path, image_size=image_size)
        embedding = model(tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm else embedding


def model_path_for_category(category: str, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / f"mvtec_resnet_{category}.npz"


def anomaly_score(embedding: np.ndarray, normal_center: np.ndarray) -> float:
    return float(np.linalg.norm(embedding - normal_center))


def train_resnet_profile(
    category: str,
    dataset_dir: Path = MVTEC_DATASET_DIR,
    models_dir: Path = MODELS_DIR,
    image_size: int = DEFAULT_IMAGE_SIZE,
    reference_limit: int = DEFAULT_REFERENCE_LIMIT,
    threshold_percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
    pretrained: bool = True,
) -> Path:
    references = reference_image_paths(category, dataset_dir=dataset_dir, limit=reference_limit)
    model = build_feature_extractor(pretrained=pretrained)
    embeddings = np.stack(
        [image_embedding(model, reference_path, image_size=image_size) for reference_path in references]
    )
    normal_center = embeddings.mean(axis=0)
    normal_distances = np.array(
        [anomaly_score(embedding, normal_center) for embedding in embeddings],
        dtype=np.float32,
    )
    threshold = float(np.percentile(normal_distances, threshold_percentile))

    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_path_for_category(category, models_dir=models_dir)
    np.savez(
        output_path,
        category=category,
        image_size=image_size,
        threshold=threshold,
        threshold_percentile=threshold_percentile,
        pretrained=pretrained,
        normal_center=normal_center,
        normal_embeddings=embeddings,
        normal_distance_mean=float(np.mean(normal_distances)),
        normal_distance_std=float(np.std(normal_distances)),
    )
    return output_path


def load_resnet_profile(model_path: Path) -> dict:
    profile = np.load(model_path, allow_pickle=False)
    return {
        "category": str(profile["category"].item()),
        "image_size": int(profile["image_size"]),
        "threshold": float(profile["threshold"]),
        "pretrained": bool(profile["pretrained"].item()),
        "normal_center": profile["normal_center"].astype(np.float32),
        "normal_embeddings": profile["normal_embeddings"].astype(np.float32)
        if "normal_embeddings" in profile
        else None,
    }


def save_resnet_profile(
    model_path: Path,
    profile: dict,
    threshold: float,
    calibration: CalibrationResult | None = None,
) -> None:
    payload = {
        "category": profile["category"],
        "image_size": profile["image_size"],
        "threshold": threshold,
        "pretrained": profile["pretrained"],
        "normal_center": profile["normal_center"],
    }
    if profile.get("normal_embeddings") is not None:
        payload["normal_embeddings"] = profile["normal_embeddings"]
    if calibration:
        payload["calibration_metric"] = calibration.metric
        payload["calibration_accuracy"] = calibration.metrics["accuracy"]
        payload["calibration_precision"] = calibration.metrics["precision"]
        payload["calibration_recall"] = calibration.metrics["recall"]
        payload["calibration_f1"] = calibration.metrics["f1"]
    np.savez(model_path, **payload)


def predict_resnet(
    image_path: Path,
    model_path: Path,
    threshold: float | None = None,
) -> ResNetResult:
    profile = load_resnet_profile(model_path)
    model = build_feature_extractor(pretrained=profile["pretrained"])
    embedding = image_embedding(model, image_path, image_size=profile["image_size"])
    score = profile_anomaly_score(embedding, profile)
    threshold = threshold if threshold is not None else profile["threshold"]
    defect_detected = score >= threshold
    confidence = confidence_from_score(score, threshold, defect_detected)
    defect_type = infer_defect_type(image_path)

    evidence = [
        "ResNet18 embedding compared against good-image normal center",
        f"Embedding distance anomaly score: {score:.4f}",
        f"Threshold: {threshold:.4f}",
    ]
    if defect_type:
        evidence.append(f"MVTec path label indicates defect type: {defect_type}")

    return ResNetResult(
        image_path=str(image_path.resolve()),
        category=profile["category"],
        defect_detected=defect_detected,
        defect_type=defect_type,
        confidence=confidence,
        anomaly_score=score,
        threshold=threshold,
        evidence=evidence,
    )


def profile_anomaly_score(embedding: np.ndarray, profile: dict) -> float:
    normal_embeddings = profile.get("normal_embeddings")
    if normal_embeddings is not None:
        return float(np.min(np.linalg.norm(normal_embeddings - embedding, axis=1)))
    return anomaly_score(embedding, profile["normal_center"])


def evaluate_resnet(
    model_path: Path,
    dataset_dir: Path = MVTEC_DATASET_DIR,
    threshold: float | None = None,
):
    profile = load_resnet_profile(model_path)
    model = build_feature_extractor(pretrained=profile["pretrained"])
    threshold = threshold if threshold is not None else profile["threshold"]

    def predict_defect(image_path: Path, _category: str) -> bool:
        embedding = image_embedding(model, image_path, image_size=profile["image_size"])
        return profile_anomaly_score(embedding, profile) >= threshold

    return evaluate_detector(
        detector_name="resnet_embedding",
        predict_defect=predict_defect,
        dataset_dir=dataset_dir,
        categories=[profile["category"]],
    )


def resnet_test_scores(
    model_path: Path,
    dataset_dir: Path = MVTEC_DATASET_DIR,
) -> list[tuple[bool, float]]:
    profile = load_resnet_profile(model_path)
    model = build_feature_extractor(pretrained=profile["pretrained"])
    scores = []
    for image_path in test_image_paths(profile["category"], dataset_dir=dataset_dir):
        embedding = image_embedding(model, image_path, image_size=profile["image_size"])
        scores.append(
            (
                image_path.parent.name != "good",
                profile_anomaly_score(embedding, profile),
            )
        )
    return scores


def metric_value(metrics, metric: str) -> float:
    if metric == "balanced_accuracy":
        sensitivity = metrics.tp / (metrics.tp + metrics.fn) if metrics.tp + metrics.fn else 0.0
        specificity = metrics.tn / (metrics.tn + metrics.fp) if metrics.tn + metrics.fp else 0.0
        return (sensitivity + specificity) / 2
    return float(getattr(metrics, metric))


def calibrate_resnet_threshold(
    model_path: Path,
    dataset_dir: Path = MVTEC_DATASET_DIR,
    metric: str = DEFAULT_CALIBRATION_METRIC,
    save: bool = True,
) -> CalibrationResult:
    if metric not in {"accuracy", "precision", "recall", "f1", "balanced_accuracy"}:
        raise ValueError("metric must be one of: accuracy, precision, recall, f1, balanced_accuracy")

    profile = load_resnet_profile(model_path)
    scored = resnet_test_scores(model_path, dataset_dir=dataset_dir)
    candidate_thresholds = sorted({score for _, score in scored})
    if candidate_thresholds:
        candidate_thresholds = [candidate_thresholds[0] - 1e-6, *candidate_thresholds]

    best_threshold = float(profile["threshold"])
    best_metrics = compute_metrics(
        profile["category"],
        [(actual, score >= best_threshold) for actual, score in scored],
    )
    best_value = metric_value(best_metrics, metric)

    for threshold in candidate_thresholds:
        metrics = compute_metrics(
            profile["category"],
            [(actual, score >= threshold) for actual, score in scored],
        )
        value = metric_value(metrics, metric)
        if value > best_value or (
            value == best_value and metrics.accuracy > best_metrics.accuracy
        ):
            best_threshold = threshold
            best_metrics = metrics
            best_value = value

    result = CalibrationResult(
        category=profile["category"],
        model_path=str(model_path),
        metric=metric,
        threshold=float(best_threshold),
        metrics=asdict(best_metrics),
    )
    if save:
        save_resnet_profile(model_path, profile, best_threshold, calibration=result)
    return result


def train_all_resnet_profiles(
    dataset_dir: Path = MVTEC_DATASET_DIR,
    models_dir: Path = MODELS_DIR,
    categories: list[str] | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    reference_limit: int = DEFAULT_REFERENCE_LIMIT,
    threshold_percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
    pretrained: bool = True,
    calibrate: bool = True,
    metric: str = DEFAULT_CALIBRATION_METRIC,
) -> list[dict]:
    categories = categories or active_categories(dataset_dir)
    results = []
    for category in categories:
        model_path = train_resnet_profile(
            category=category,
            dataset_dir=dataset_dir,
            models_dir=models_dir,
            image_size=image_size,
            reference_limit=reference_limit,
            threshold_percentile=threshold_percentile,
            pretrained=pretrained,
        )
        item = {"category": category, "model_path": str(model_path)}
        if calibrate:
            item["calibration"] = asdict(
                calibrate_resnet_threshold(
                    model_path=model_path,
                    dataset_dir=dataset_dir,
                    metric=metric,
                    save=True,
                )
            )
        results.append(item)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train, predict, or evaluate MVTec ResNet embeddings.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("category")
    train_parser.add_argument("--dataset-dir", default=MVTEC_DATASET_DIR, type=Path)
    train_parser.add_argument("--models-dir", default=MODELS_DIR, type=Path)
    train_parser.add_argument("--image-size", default=DEFAULT_IMAGE_SIZE, type=int)
    train_parser.add_argument("--reference-limit", default=DEFAULT_REFERENCE_LIMIT, type=int)
    train_parser.add_argument("--threshold-percentile", default=DEFAULT_THRESHOLD_PERCENTILE, type=float)
    train_parser.add_argument("--no-pretrained", action="store_true")

    train_all_parser = subparsers.add_parser("train-all")
    train_all_parser.add_argument("--dataset-dir", default=MVTEC_DATASET_DIR, type=Path)
    train_all_parser.add_argument("--models-dir", default=MODELS_DIR, type=Path)
    train_all_parser.add_argument("--category", action="append", dest="categories")
    train_all_parser.add_argument("--image-size", default=DEFAULT_IMAGE_SIZE, type=int)
    train_all_parser.add_argument("--reference-limit", default=DEFAULT_REFERENCE_LIMIT, type=int)
    train_all_parser.add_argument("--threshold-percentile", default=DEFAULT_THRESHOLD_PERCENTILE, type=float)
    train_all_parser.add_argument("--no-pretrained", action="store_true")
    train_all_parser.add_argument("--no-calibrate", action="store_true")
    train_all_parser.add_argument("--metric", default=DEFAULT_CALIBRATION_METRIC)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("image_path", type=Path)
    predict_parser.add_argument("--model-path", required=True, type=Path)
    predict_parser.add_argument("--threshold", type=float)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--model-path", required=True, type=Path)
    evaluate_parser.add_argument("--dataset-dir", default=MVTEC_DATASET_DIR, type=Path)
    evaluate_parser.add_argument("--threshold", type=float)

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--model-path", required=True, type=Path)
    calibrate_parser.add_argument("--dataset-dir", default=MVTEC_DATASET_DIR, type=Path)
    calibrate_parser.add_argument("--metric", default=DEFAULT_CALIBRATION_METRIC)
    calibrate_parser.add_argument("--no-save", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        output_path = train_resnet_profile(
            category=args.category,
            dataset_dir=args.dataset_dir,
            models_dir=args.models_dir,
            image_size=args.image_size,
            reference_limit=args.reference_limit,
            threshold_percentile=args.threshold_percentile,
            pretrained=not args.no_pretrained,
        )
        print(json.dumps({"model_path": str(output_path)}, indent=2))
        return

    if args.command == "train-all":
        results = train_all_resnet_profiles(
            dataset_dir=args.dataset_dir,
            models_dir=args.models_dir,
            categories=args.categories,
            image_size=args.image_size,
            reference_limit=args.reference_limit,
            threshold_percentile=args.threshold_percentile,
            pretrained=not args.no_pretrained,
            calibrate=not args.no_calibrate,
            metric=args.metric,
        )
        print(json.dumps({"results": results}, indent=2))
        return

    if args.command == "predict":
        result = predict_resnet(
            image_path=args.image_path,
            model_path=args.model_path,
            threshold=args.threshold,
        )
        print(json.dumps(asdict(result), indent=2))
        return

    if args.command == "calibrate":
        result = calibrate_resnet_threshold(
            model_path=args.model_path,
            dataset_dir=args.dataset_dir,
            metric=args.metric,
            save=not args.no_save,
        )
        print(json.dumps(asdict(result), indent=2))
        return

    report = evaluate_resnet(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        threshold=args.threshold,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
