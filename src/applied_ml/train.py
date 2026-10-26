"""Train, evaluate, and serialize the customer churn model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.applied_ml.config import ARTIFACT_DIR, METRICS_PATH, MODEL_PATH, RANDOM_STATE
from src.applied_ml.data import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset
from src.applied_ml.features import build_pipeline


def calculate_metrics(model, x_test, y_test) -> dict:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def train(data_path: str | Path, model_path: str | Path = MODEL_PATH,
          metrics_path: str | Path = METRICS_PATH) -> dict:
    frame = load_dataset(data_path)
    x = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_metrics = calculate_metrics(baseline, x_test, y_test)

    model = build_pipeline()
    model.fit(x_train, y_train)
    model_metrics = calculate_metrics(model, x_test, y_test)

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    report = {
        "project": "applied-ml-project",
        "model": "LogisticRegression",
        "random_state": RANDOM_STATE,
        "rows_total": int(len(frame)),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "target_positive_rate": round(float(y.mean()), 4),
        "baseline": baseline_metrics,
        "model_metrics": model_metrics,
        "artifacts": {"model": str(model_path), "metrics": str(metrics_path)},
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/customers.csv")
    parser.add_argument("--model-output", default=str(MODEL_PATH))
    parser.add_argument("--metrics-output", default=str(METRICS_PATH))
    args = parser.parse_args()

    report = train(args.data, args.model_output, args.metrics_output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
