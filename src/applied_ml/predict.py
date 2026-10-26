"""Load a serialized pipeline and produce a validated churn prediction."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.applied_ml.config import MODEL_PATH
from src.applied_ml.data import FEATURE_COLUMNS


def load_model(model_path: str | Path = MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Train it first with "
            "'python -m src.applied_ml.train'."
        )
    return joblib.load(path)


def predict_one(model, features: dict) -> dict:
    missing = sorted(set(FEATURE_COLUMNS) - set(features))
    if missing:
        raise ValueError(f"Missing features: {missing}")
    frame = pd.DataFrame([{column: features[column] for column in FEATURE_COLUMNS}])
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "churn_probability": round(probability, 4),
        "prediction": prediction,
        "label": "likely_to_churn" if prediction else "unlikely_to_churn",
    }
