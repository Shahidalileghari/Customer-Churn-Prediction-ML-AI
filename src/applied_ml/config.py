"""Central configuration and project paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customers.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "churn_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
RANDOM_STATE = 42
MODEL_VERSION = "1.0.0"
