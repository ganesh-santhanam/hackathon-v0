from pathlib import Path

import numpy as np
from PIL import Image

from industrial_ai.vision.localization import (
    PatchScore,
    compare_patch_scores,
    localize_by_comparison,
    patch_scores_to_heatmap,
)
from industrial_ai.vision.mvtec_compare import compare_mvtec_image


def save_gray_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def test_patch_heatmap_highlights_scored_region():
    scores = [
        PatchScore(row=0, col=0, height=8, width=8, score=0.1),
        PatchScore(row=8, col=8, height=8, width=8, score=0.9),
    ]

    heatmap = patch_scores_to_heatmap(scores, image_size=16)

    assert heatmap.shape == (16, 16)
    assert heatmap[10, 10] == 1.0
    assert heatmap[2, 2] < heatmap[10, 10]


def test_compare_patch_scores_uses_local_reference_distance():
    image = np.zeros((16, 16), dtype=np.float32)
    image[8:16, 8:16] = 1.0
    references = np.zeros((1, 16 * 16), dtype=np.float32)

    scores = compare_patch_scores(
        image_vector=image.reshape(-1),
        reference_vectors_array=references,
        image_size=16,
        patch_size=8,
        stride=8,
    )

    top_patch = max(scores, key=lambda patch: patch.score)
    assert top_patch.row == 8
    assert top_patch.col == 8
    assert top_patch.score == 1.0


def test_localization_outputs_bounding_box_for_detected_defect(tmp_path):
    dataset_dir = tmp_path / "mvtec"
    save_gray_image(dataset_dir / "cable" / "train" / "good" / "000.png", np.zeros((64, 64)))
    defect = np.zeros((64, 64), dtype=np.uint8)
    defect[40:56, 40:56] = 255
    image_path = dataset_dir / "cable" / "test" / "cut_outer_insulation" / "000.png"
    save_gray_image(image_path, defect)

    result = compare_mvtec_image(
        image_path=image_path,
        category="cable",
        dataset_dir=dataset_dir,
        threshold=0.1,
        image_size=64,
    )
    localization = localize_by_comparison(
        image_path=image_path,
        category="cable",
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "outputs",
        image_size=64,
        patch_size=16,
        stride=8,
    )

    assert result.defect_detected is True
    assert localization.available is True
    assert localization.bounding_box[0] >= 32
    assert localization.bounding_box[1] >= 32
    assert localization.top_anomaly_score > 0
    assert Path(localization.heatmap_path).exists()
    assert Path(localization.annotated_image_path).exists()
