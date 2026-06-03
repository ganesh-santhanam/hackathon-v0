import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from industrial_ai.paths import MVTEC_DATASET_DIR, VISION_OUTPUT_DIR
from industrial_ai.vision.mvtec_compare import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_REFERENCE_LIMIT,
    reference_image_paths,
    reference_vectors,
)
from industrial_ai.vision.mvtec_resnet import (
    build_feature_extractor,
    image_embedding,
    load_resnet_profile,
    profile_anomaly_score,
)


DEFAULT_LOCALIZATION_PATCH_SIZE = 32
DEFAULT_LOCALIZATION_STRIDE = 16


@dataclass(frozen=True)
class VisionLocalization:
    available: bool
    bounding_box: list[int]
    confidence: float
    method: str
    heatmap_path: str
    annotated_image_path: str
    top_anomaly_score: float
    patch_scores: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class PatchScore:
    row: int
    col: int
    height: int
    width: int
    score: float


def patch_origins(size: int, patch_size: int, stride: int) -> list[int]:
    if size <= patch_size:
        return [0]
    origins = list(range(0, size - patch_size + 1, stride))
    final_origin = size - patch_size
    if origins[-1] != final_origin:
        origins.append(final_origin)
    return origins


def compare_patch_scores(
    image_vector: np.ndarray,
    reference_vectors_array: np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    patch_size: int = DEFAULT_LOCALIZATION_PATCH_SIZE,
    stride: int = DEFAULT_LOCALIZATION_STRIDE,
) -> list[PatchScore]:
    image = image_vector.reshape(image_size, image_size)
    references = reference_vectors_array.reshape(reference_vectors_array.shape[0], image_size, image_size)
    scores = []
    for row in patch_origins(image_size, patch_size, stride):
        for col in patch_origins(image_size, patch_size, stride):
            image_patch = image[row : row + patch_size, col : col + patch_size]
            reference_patches = references[:, row : row + patch_size, col : col + patch_size]
            distances = np.mean(np.abs(reference_patches - image_patch), axis=(1, 2))
            scores.append(
                PatchScore(
                    row=row,
                    col=col,
                    height=patch_size,
                    width=patch_size,
                    score=float(np.min(distances)),
                )
            )
    return scores


def patch_scores_to_heatmap(
    patch_scores: list[PatchScore],
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    heatmap = np.zeros((image_size, image_size), dtype=np.float32)
    counts = np.zeros((image_size, image_size), dtype=np.float32)
    for patch in patch_scores:
        heatmap[patch.row : patch.row + patch.height, patch.col : patch.col + patch.width] += patch.score
        counts[patch.row : patch.row + patch.height, patch.col : patch.col + patch.width] += 1.0
    heatmap = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts > 0)
    peak = float(heatmap.max())
    return heatmap / peak if peak > 0 else heatmap


def localization_confidence(scores: list[float]) -> float:
    if not scores:
        return 0.0
    top_score = max(scores)
    if top_score <= 0:
        return 0.0
    mean_score = float(np.mean(scores))
    confidence = top_score / (top_score + mean_score)
    return round(float(min(0.95, max(0.35, confidence))), 3)


def scaled_bounding_box(
    patch: PatchScore,
    original_size: tuple[int, int],
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> list[int]:
    original_width, original_height = original_size
    x_scale = original_width / image_size
    y_scale = original_height / image_size
    x1 = int(round(patch.col * x_scale))
    y1 = int(round(patch.row * y_scale))
    x2 = int(round((patch.col + patch.width) * x_scale))
    y2 = int(round((patch.row + patch.height) * y_scale))
    return [
        max(0, min(original_width, x1)),
        max(0, min(original_height, y1)),
        max(0, min(original_width, x2)),
        max(0, min(original_height, y2)),
    ]


def output_stem(image_path: Path, category: str, method: str) -> str:
    key = f"{image_path.resolve()}:{image_path.stat().st_mtime_ns}:{method}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()[:10]
    return f"{category}-{image_path.stem}-{method}-{digest}"


def save_visualization_outputs(
    image_path: Path,
    category: str,
    method: str,
    heatmap: np.ndarray,
    bounding_box: list[int],
    confidence: float,
    output_dir: Path = VISION_OUTPUT_DIR,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(image_path, category, method)
    heatmap_path = output_dir / f"{stem}-heatmap.png"
    annotated_path = output_dir / f"{stem}-annotated.png"

    original = Image.open(image_path).convert("RGB")
    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").resize(
        original.size,
        Image.Resampling.BILINEAR,
    )
    red = Image.new("RGB", original.size, (255, 0, 0))
    heatmap_overlay = Image.blend(original, red, 0.45)
    heatmap_result = Image.composite(heatmap_overlay, original, heatmap_image)
    heatmap_result.save(heatmap_path)

    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)
    x1, y1, x2, y2 = bounding_box
    draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=max(3, min(original.size) // 80))
    label = "Approximate anomaly region" if confidence < 0.7 else "Anomaly region"
    text_position = (x1, max(0, y1 - 18))
    draw.rectangle(
        (text_position[0], text_position[1], text_position[0] + 220, text_position[1] + 18),
        fill=(255, 0, 0),
    )
    draw.text((text_position[0] + 4, text_position[1] + 2), label, fill=(255, 255, 255))
    annotated.save(annotated_path)

    return heatmap_path, annotated_path


def localize_by_comparison(
    image_path: Path,
    category: str,
    dataset_dir: Path = MVTEC_DATASET_DIR,
    output_dir: Path = VISION_OUTPUT_DIR,
    reference_limit: int = DEFAULT_REFERENCE_LIMIT,
    image_size: int = DEFAULT_IMAGE_SIZE,
    patch_size: int = DEFAULT_LOCALIZATION_PATCH_SIZE,
    stride: int = DEFAULT_LOCALIZATION_STRIDE,
) -> VisionLocalization:
    image_path = image_path.resolve()
    original_size = Image.open(image_path).size
    image_vector = np.asarray(
        Image.open(image_path).convert("L").resize((image_size, image_size)),
        dtype=np.float32,
    ).reshape(-1) / 255.0
    references = reference_image_paths(category, dataset_dir=dataset_dir, limit=reference_limit)
    reference_vectors_array = reference_vectors(references, image_size=image_size)
    patch_scores = compare_patch_scores(
        image_vector=image_vector,
        reference_vectors_array=reference_vectors_array,
        image_size=image_size,
        patch_size=patch_size,
        stride=stride,
    )
    top_patch = max(patch_scores, key=lambda patch: patch.score)
    heatmap = patch_scores_to_heatmap(patch_scores, image_size=image_size)
    scores = [patch.score for patch in patch_scores]
    confidence = localization_confidence(scores)
    bounding_box = scaled_bounding_box(top_patch, original_size, image_size=image_size)
    heatmap_path, annotated_path = save_visualization_outputs(
        image_path=image_path,
        category=category,
        method="patch_distance",
        heatmap=heatmap,
        bounding_box=bounding_box,
        confidence=confidence,
        output_dir=output_dir,
    )
    return VisionLocalization(
        available=True,
        bounding_box=bounding_box,
        confidence=confidence,
        method="patch_distance",
        heatmap_path=str(heatmap_path),
        annotated_image_path=str(annotated_path),
        top_anomaly_score=float(top_patch.score),
        patch_scores=[float(score) for score in scores],
    )


def crop_resnet_patch(image_path: Path, patch: PatchScore, image_size: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    crop = image.crop((patch.col, patch.row, patch.col + patch.width, patch.row + patch.height))
    patch_path = output_dir / f"{output_stem(image_path, 'patch', 'resnet')}-{patch.row}-{patch.col}.png"
    crop.resize((image_size, image_size)).save(patch_path)
    return patch_path


def localize_by_resnet(
    image_path: Path,
    category: str,
    model_path: Path,
    output_dir: Path = VISION_OUTPUT_DIR,
    patch_size: int = DEFAULT_LOCALIZATION_PATCH_SIZE,
    stride: int = DEFAULT_LOCALIZATION_STRIDE,
) -> VisionLocalization:
    image_path = image_path.resolve()
    profile = load_resnet_profile(model_path)
    image_size = int(profile["image_size"])
    original_size = Image.open(image_path).size
    model = build_feature_extractor(pretrained=profile["pretrained"])
    candidate_patches = [
        PatchScore(row=row, col=col, height=patch_size, width=patch_size, score=0.0)
        for row in patch_origins(image_size, patch_size, stride)
        for col in patch_origins(image_size, patch_size, stride)
    ]
    temp_dir = output_dir / "_patches"
    scored_patches = []
    for patch in candidate_patches:
        patch_path = crop_resnet_patch(image_path, patch, image_size=image_size, output_dir=temp_dir)
        embedding = image_embedding(model, patch_path, image_size=image_size)
        scored_patches.append(
            PatchScore(
                row=patch.row,
                col=patch.col,
                height=patch.height,
                width=patch.width,
                score=profile_anomaly_score(embedding, profile),
            )
        )

    top_patch = max(scored_patches, key=lambda patch: patch.score)
    heatmap = patch_scores_to_heatmap(scored_patches, image_size=image_size)
    scores = [patch.score for patch in scored_patches]
    confidence = localization_confidence(scores)
    bounding_box = scaled_bounding_box(top_patch, original_size, image_size=image_size)
    heatmap_path, annotated_path = save_visualization_outputs(
        image_path=image_path,
        category=category,
        method="embedding_distance",
        heatmap=heatmap,
        bounding_box=bounding_box,
        confidence=confidence,
        output_dir=output_dir,
    )
    return VisionLocalization(
        available=True,
        bounding_box=bounding_box,
        confidence=confidence,
        method="embedding_distance",
        heatmap_path=str(heatmap_path),
        annotated_image_path=str(annotated_path),
        top_anomaly_score=float(top_patch.score),
        patch_scores=[float(score) for score in scores],
    )
