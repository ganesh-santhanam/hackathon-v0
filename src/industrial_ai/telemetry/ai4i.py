from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from industrial_ai.paths import AI4I_DATASET_PATH


TARGET_COLUMN = "machine_failure"
ID_COLUMNS = ("udi", "product_id")
FAILURE_MODE_COLUMNS = ("twf", "hdf", "pwf", "osf", "rnf")
CATEGORICAL_FEATURES = ("type",)
NUMERIC_FEATURES = (
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass(frozen=True)
class AI4IDataset:
    features: pd.DataFrame
    target: pd.Series
    raw: pd.DataFrame


def normalize_column_name(column: str) -> str:
    return (
        column.strip()
        .lower()
        .replace("[", "")
        .replace("]", "")
        .replace("/", "_")
        .replace(" ", "_")
    )


def load_ai4i_dataset(path: Path = AI4I_DATASET_PATH) -> AI4IDataset:
    raw = pd.read_csv(path)
    normalized = raw.rename(columns=normalize_column_name)

    missing = [column for column in FEATURE_COLUMNS + (TARGET_COLUMN,) if column not in normalized]
    if missing:
        raise ValueError(f"AI4I dataset is missing expected columns: {missing}")

    features = normalized.loc[:, FEATURE_COLUMNS].copy()
    target = normalized.loc[:, TARGET_COLUMN].astype(int).copy()
    return AI4IDataset(features=features, target=target, raw=normalized)

