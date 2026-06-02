import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from industrial_ai.paths import MODELS_DIR, MVTEC_DATASET_DIR
from industrial_ai.vision.mvtec_compare import (
    DEFAULT_ANOMALY_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_REFERENCE_LIMIT,
    confidence_from_score,
    infer_defect_type,
    reference_image_paths,
)


DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 5


@dataclass(frozen=True)
class AutoencoderResult:
    image_path: str
    category: str
    defect_detected: bool
    defect_type: str | None
    confidence: float
    reconstruction_error: float
    threshold: float
    evidence: list[str]


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for MVTec autoencoder training. Install torch first."
        ) from exc
    return torch, nn, DataLoader, Dataset


def load_image_tensor(image_path: Path, image_size: int = DEFAULT_IMAGE_SIZE):
    torch, _, _, _ = require_torch()
    image = Image.open(image_path).convert("L").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def build_autoencoder(image_size: int = DEFAULT_IMAGE_SIZE):
    _, nn, _, _ = require_torch()
    if image_size % 4 != 0:
        raise ValueError("image_size must be divisible by 4")

    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2),
        nn.Sigmoid(),
    )


def model_path_for_category(category: str, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / f"mvtec_autoencoder_{category}.pt"


def image_dataset(image_paths: list[Path], image_size: int = DEFAULT_IMAGE_SIZE):
    _, _, _, Dataset = require_torch()

    class ImageDataset(Dataset):
        def __len__(self) -> int:
            return len(image_paths)

        def __getitem__(self, index: int):
            return load_image_tensor(image_paths[index], image_size=image_size)

    return ImageDataset()


def reconstruction_errors(model, image_paths: list[Path], image_size: int = DEFAULT_IMAGE_SIZE) -> list[float]:
    torch, _, _, _ = require_torch()
    model.eval()
    errors = []
    with torch.no_grad():
        for image_path in image_paths:
            image = load_image_tensor(image_path, image_size=image_size).unsqueeze(0)
            reconstruction = model(image)
            errors.append(float(torch.mean((reconstruction - image) ** 2).item()))
    return errors


def train_autoencoder(
    category: str,
    dataset_dir: Path = MVTEC_DATASET_DIR,
    models_dir: Path = MODELS_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    reference_limit: int = DEFAULT_REFERENCE_LIMIT,
) -> Path:
    torch, nn, DataLoader, _ = require_torch()
    references = reference_image_paths(category, dataset_dir=dataset_dir, limit=reference_limit)
    dataset = image_dataset(references, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = build_autoencoder(image_size=image_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()

    train_errors = reconstruction_errors(model, references, image_size=image_size)
    threshold = max(
        DEFAULT_ANOMALY_THRESHOLD,
        float(np.percentile(train_errors, 95) + np.std(train_errors)),
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_path_for_category(category, models_dir=models_dir)
    torch.save(
        {
            "category": category,
            "image_size": image_size,
            "threshold": threshold,
            "state_dict": model.state_dict(),
            "train_error_mean": float(np.mean(train_errors)),
            "train_error_std": float(np.std(train_errors)),
        },
        output_path,
    )
    return output_path


def load_autoencoder(model_path: Path):
    torch, _, _, _ = require_torch()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_autoencoder(image_size=checkpoint["image_size"])
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def predict_autoencoder(
    image_path: Path,
    model_path: Path,
    threshold: float | None = None,
) -> AutoencoderResult:
    model, checkpoint = load_autoencoder(model_path)
    image_size = int(checkpoint["image_size"])
    category = str(checkpoint["category"])
    reconstruction_error = reconstruction_errors(model, [image_path], image_size=image_size)[0]
    threshold = threshold if threshold is not None else float(checkpoint["threshold"])
    defect_detected = reconstruction_error >= threshold
    confidence = confidence_from_score(reconstruction_error, threshold, defect_detected)
    defect_type = infer_defect_type(image_path)

    evidence = [
        "Autoencoder trained on good MVTec reference images",
        f"Reconstruction error: {reconstruction_error:.4f}",
        f"Threshold: {threshold:.4f}",
    ]
    if defect_type:
        evidence.append(f"MVTec path label indicates defect type: {defect_type}")

    return AutoencoderResult(
        image_path=str(image_path.resolve()),
        category=category,
        defect_detected=defect_detected,
        defect_type=defect_type,
        confidence=confidence,
        reconstruction_error=reconstruction_error,
        threshold=threshold,
        evidence=evidence,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or run a small MVTec autoencoder.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("category")
    train_parser.add_argument("--dataset-dir", default=MVTEC_DATASET_DIR, type=Path)
    train_parser.add_argument("--models-dir", default=MODELS_DIR, type=Path)
    train_parser.add_argument("--epochs", default=DEFAULT_EPOCHS, type=int)
    train_parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    train_parser.add_argument("--image-size", default=DEFAULT_IMAGE_SIZE, type=int)
    train_parser.add_argument("--reference-limit", default=DEFAULT_REFERENCE_LIMIT, type=int)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("image_path", type=Path)
    predict_parser.add_argument("--model-path", required=True, type=Path)
    predict_parser.add_argument("--threshold", type=float)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        output_path = train_autoencoder(
            category=args.category,
            dataset_dir=args.dataset_dir,
            models_dir=args.models_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            reference_limit=args.reference_limit,
        )
        print(json.dumps({"model_path": str(output_path)}, indent=2))
        return

    result = predict_autoencoder(
        image_path=args.image_path,
        model_path=args.model_path,
        threshold=args.threshold,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
