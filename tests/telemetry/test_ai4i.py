from industrial_ai.telemetry.ai4i import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_ai4i_dataset,
    normalize_column_name,
)


def test_normalize_column_name_handles_units_and_spaces():
    assert normalize_column_name("Air temperature [K]") == "air_temperature_k"
    assert normalize_column_name("Torque [Nm]") == "torque_nm"


def test_load_ai4i_dataset_splits_features_and_target():
    dataset = load_ai4i_dataset()

    assert dataset.features.shape == (10000, len(FEATURE_COLUMNS))
    assert dataset.target.shape == (10000,)
    assert TARGET_COLUMN in dataset.raw.columns
    assert set(dataset.target.unique()) == {0, 1}
    assert "twf" in dataset.raw.columns
    assert "twf" not in dataset.features.columns

