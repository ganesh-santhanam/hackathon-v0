from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INCIDENTS_DATA_DIR = DATA_DIR / "incidents"
QDRANT_DATA_DIR = DATA_DIR / "qdrant"
APPROVALS_DATA_DIR = DATA_DIR / "approvals"
MODELS_DIR = PROJECT_ROOT / "models"

AI4I_DATASET_PATH = PROJECT_ROOT / "ai4i_dataset" / "ai4i2020.csv"
TELEMETRY_MODEL_PATH = MODELS_DIR / "telemetry_model.joblib"
TELEMETRY_METRICS_PATH = MODELS_DIR / "telemetry_metrics.json"
APPROVALS_STORE_PATH = APPROVALS_DATA_DIR / "approvals.json"
