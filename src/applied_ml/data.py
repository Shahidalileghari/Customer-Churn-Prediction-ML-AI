"""Generate a reproducible synthetic customer churn dataset or validate a CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = ["tenure_months", "monthly_charges", "support_tickets"]
CATEGORICAL_COLUMNS = ["contract_type", "payment_method"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
TARGET_COLUMN = "churn"


def generate_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """Create a synthetic dataset with a learnable but imperfect churn signal."""
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100 for a useful demo dataset.")

    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 73, size=n_samples)
    monthly = np.clip(rng.normal(68, 22, size=n_samples), 18, 150).round(2)
    tickets = np.clip(rng.poisson(1.6, size=n_samples), 0, 10)
    contract = rng.choice(
        ["month_to_month", "one_year", "two_year"],
        size=n_samples,
        p=[0.56, 0.27, 0.17],
    )
    payment = rng.choice(
        ["electronic", "bank_transfer", "credit_card"],
        size=n_samples,
        p=[0.48, 0.27, 0.25],
    )

    logit = (
        1.1
        - 0.045 * tenure
        + 0.018 * (monthly - 65)
        + 0.42 * tickets
        + 1.05 * (contract == "month_to_month")
        - 0.55 * (contract == "two_year")
        + 0.25 * (payment == "electronic")
        + rng.normal(0, 0.75, size=n_samples)
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "contract_type": contract,
            "support_tickets": tickets,
            "payment_method": payment,
            "churn": churn,
        }
    )

    # Include a small number of realistic missing values so cleaning is exercised.
    for column in ("monthly_charges", "payment_method"):
        indices = rng.choice(frame.index, size=max(1, n_samples // 100), replace=False)
        frame.loc[indices, column] = np.nan

    return frame


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and target values; return a defensive copy."""
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    result = frame[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    if result.empty:
        raise ValueError("Dataset is empty.")

    result[TARGET_COLUMN] = pd.to_numeric(result[TARGET_COLUMN], errors="coerce")
    result = result.dropna(subset=[TARGET_COLUMN])
    if not result[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Target column 'churn' must contain only 0 or 1.")
    if result[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Target column must contain both classes: 0 and 1.")
    return result


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read and validate a CSV dataset."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Generate one with: "
            "python -m src.applied_ml.data --output data/raw/customers.csv"
        )
    return validate_dataset(pd.read_csv(csv_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/customers.csv")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_dataset(args.samples, args.seed)
    frame.to_csv(output_path, index=False)
    print(f"Generated {len(frame)} rows at {output_path.resolve()}")
    print(f"Churn rate: {frame[TARGET_COLUMN].mean():.1%}")


if __name__ == "__main__":
    main()
