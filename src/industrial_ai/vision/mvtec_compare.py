import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from industrial_ai.paths import MVTEC_DATASET_DIR


DEFAULT_IMAGE_SIZE = 128
DEFAULT_REFERENCE_LIMIT = 50
DEFAULT_ANOMALY_THRESHOLD = 0.10
DEFAULT_PATCH_SIZE = 16
CALIBRATED_CATEGORY_THRESHOLDS = {
    "cable": 0.168,
    "grid": 0.122,
    "metal_nut": 0.133,
    "screw": 0.218,
    "transistor": 0.142,
}


@dataclass(frozen=True)
class VisionResult:
    image_path: str
    category: str
    defect_detected: bool
    defect_type: str | None
    confidence: float
    anomaly_score: float
    threshold: float
    nearest_reference: str
    evidence: list[str]


def infer_category(image_path: Path, dataset_dir: Path = MVTEC_DATASET_DIR) -> str:
    relative = image_path.resolve().relative_to(dataset_dir.resolve())
    return relative.parts[0]


def infer_defect_type(image_path: Path) -> str | None:
    parts = image_path.parts
    if "test" not in parts:
        return None
    test_index = parts.index("test")
    if test_index + 1 >= len(parts):
        return None
    defect_type = parts[test_index + 1]
    return None if defect_type == "good" else defect_type


def load_image_vector(image_path: Path, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    image = Image.open(image_path).convert("L").resize((image_size, image_size))
    return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0


def reference_image_paths(
    category: str,
    dataset_dir: Path = MVTEC_DATASET_DIR,
    limit: int = DEFAULT_REFERENCE_LIMIT,
) -> list[Path]:
    reference_dir = dataset_dir / category / "train" / "good"
    references = sorted(reference_dir.glob("*.png"))[:limit]
    if not references:
        raise FileNotFoundError(f"No MVTec good reference images found at {reference_dir}")
    return references


def reference_vectors(
    reference_paths: list[Path],
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    return np.stack(
        [load_image_vector(reference_path, image_size=image_size) for reference_path in reference_paths]
    )


def nearest_reference_distance(
    image_vector: np.ndarray,
    reference_paths: list[Path],
    reference_vectors_array: np.ndarray,
) -> tuple[float, Path]:
    distances = [
        (float(np.mean(np.abs(image_vector - reference_vector))), reference_path)
        for reference_path, reference_vector in zip(reference_paths, reference_vectors_array, strict=True)
    ]
    return min(distances, key=lambda item: item[0])


def patch_anomaly_score(
    image_vector: np.ndarray,
    reference_vectors_array: np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> float:
    image = image_vector.reshape(image_size, image_size)
    mean_reference = reference_vectors_array.mean(axis=0).reshape(image_size, image_size)
    difference = np.abs(image - mean_reference)

    patch_scores = [
        float(difference[row : row + patch_size, col : col + patch_size].mean())
        for row in range(0, image_size, patch_size)
        for col in range(0, image_size, patch_size)
    ]
    return max(patch_scores)


def confidence_from_score(score: float, threshold: float, defect_detected: bool) -> float:
    if threshold <= 0:
        return 0.99 if defect_detected else 0.5
    margin = abs(score - threshold) / threshold
    if defect_detected:
        return round(min(0.95, max(0.5, 0.55 + margin)), 3)
    return round(min(0.95, max(0.5, 0.55 + margin)), 3)


def threshold_for_category(category: str, threshold: float | None = None) -> float:
    if threshold is not None:
        return threshold
    return CALIBRATED_CATEGORY_THRESHOLDS.get(category, DEFAULT_ANOMALY_THRESHOLD)


def compare_mvtec_image(
    image_path: Path,
    category: str | None = None,
    dataset_dir: Path = MVTEC_DATASET_DIR,
    threshold: float | None = None,
    reference_limit: int = DEFAULT_REFERENCE_LIMIT,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> VisionResult:
    image_path = image_path.resolve()
    category = category or infer_category(image_path, dataset_dir)
    threshold = threshold_for_category(category, threshold)
    image_vector = load_image_vector(image_path, image_size=image_size)
    references = reference_image_paths(category, dataset_dir, limit=reference_limit)
    reference_vectors_array = reference_vectors(references, image_size=image_size)
    nearest_distance, nearest_reference = nearest_reference_distance(
        image_vector,
        references,
        reference_vectors_array,
    )
    anomaly_score = patch_anomaly_score(
        image_vector,
        reference_vectors_array,
        image_size=image_size,
    )
    defect_detected = anomaly_score >= threshold
    defect_type = infer_defect_type(image_path)
    confidence = confidence_from_score(anomaly_score, threshold, defect_detected)

    evidence = [
        f"Compared against {len(references)} good {category} reference images",
        f"Nearest-reference average distance: {nearest_distance:.3f}",
        f"Max local patch anomaly score: {anomaly_score:.3f}",
        f"Threshold: {threshold:.3f}",
    ]
    if defect_type:
        evidence.append(f"MVTec path label indicates defect type: {defect_type}")

    return VisionResult(
        image_path=str(image_path),
        category=category,
        defect_detected=defect_detected,
        defect_type=defect_type,
        confidence=confidence,
        anomaly_score=anomaly_score,
        threshold=threshold,
        nearest_reference=str(nearest_reference),
        evidence=evidence,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare an MVTec image against good references.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--category")
    parser.add_argument("--dataset-dir", default=MVTEC_DATASET_DIR, type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--reference-limit", default=DEFAULT_REFERENCE_LIMIT, type=int)
    parser.add_argument("--image-size", default=DEFAULT_IMAGE_SIZE, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = compare_mvtec_image(
        image_path=args.image_path,
        category=args.category,
        dataset_dir=args.dataset_dir,
        threshold=args.threshold,
        reference_limit=args.reference_limit,
        image_size=args.image_size,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
