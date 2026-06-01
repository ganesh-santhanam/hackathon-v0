import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from industrial_ai.paths import TELEMETRY_METRICS_PATH, TELEMETRY_MODEL_PATH
from industrial_ai.telemetry.ai4i import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_ai4i_dataset,
)


DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7)


@dataclass(frozen=True)
class TrainingResult:
    model_path: str
    metrics_path: str
    rows: int
    positives: int
    test_roc_auc: float
    test_average_precision: float


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
            ("numeric", "passthrough", list(NUMERIC_FEATURES)),
        ]
    )
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=150,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate_thresholds(y_true, probabilities, thresholds=DEFAULT_THRESHOLDS) -> dict[str, dict]:
    evaluations = {}
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        evaluations[str(threshold)] = {
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return evaluations


def train_telemetry_model(
    model_path: Path = TELEMETRY_MODEL_PATH,
    metrics_path: Path = TELEMETRY_METRICS_PATH,
) -> TrainingResult:
    dataset = load_ai4i_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / positives

    pipeline = build_pipeline(scale_pos_weight=scale_pos_weight)
    pipeline.fit(x_train, y_train)

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    threshold_metrics = evaluate_thresholds(y_test, probabilities)

    result = TrainingResult(
        model_path=str(model_path),
        metrics_path=str(metrics_path),
        rows=len(dataset.target),
        positives=int(dataset.target.sum()),
        test_roc_auc=float(roc_auc_score(y_test, probabilities)),
        test_average_precision=float(average_precision_score(y_test, probabilities)),
    )
    metrics = asdict(result) | {
        "threshold_metrics": threshold_metrics,
        "feature_columns": list(dataset.features.columns),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return result


def main() -> None:
    result = train_telemetry_model()
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
