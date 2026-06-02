from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from industrial_ai.vision.mvtec_autoencoder import (  # noqa: E402
    build_autoencoder,
    load_image_tensor,
    model_path_for_category,
    predict_autoencoder,
)


def save_gray_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=value).save(path)


def test_model_path_for_category_uses_models_dir(tmp_path):
    assert model_path_for_category("bottle", models_dir=tmp_path) == tmp_path / (
        "mvtec_autoencoder_bottle.pt"
    )


def test_load_image_tensor_returns_single_channel_tensor(tmp_path):
    image_path = tmp_path / "image.png"
    save_gray_image(image_path, 128)

    tensor = load_image_tensor(image_path, image_size=16)

    assert tuple(tensor.shape) == (1, 16, 16)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_predict_autoencoder_returns_reconstruction_result(tmp_path):
    image_path = tmp_path / "bottle" / "test" / "scratch" / "000.png"
    model_path = tmp_path / "model.pt"
    save_gray_image(image_path, 255)

    model = build_autoencoder(image_size=16)
    torch.save(
        {
            "category": "bottle",
            "image_size": 16,
            "threshold": 0.001,
            "state_dict": model.state_dict(),
            "train_error_mean": 0.0,
            "train_error_std": 0.0,
        },
        model_path,
    )

    result = predict_autoencoder(image_path, model_path)

    assert result.category == "bottle"
    assert result.defect_detected is True
    assert result.defect_type == "scratch"
    assert result.reconstruction_error >= 0.0
    assert result.evidence[0] == "Autoencoder trained on good MVTec reference images"
