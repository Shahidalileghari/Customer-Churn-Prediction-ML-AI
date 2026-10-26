import pandas as pd
import pytest

from src.applied_ml.data import generate_dataset, validate_dataset


def test_generated_dataset_has_expected_schema_and_classes():
    frame = generate_dataset(n_samples=300, random_state=11)
    assert len(frame) == 300
    assert {"tenure_months", "monthly_charges", "contract_type",
            "support_tickets", "payment_method", "churn"} <= set(frame.columns)
    assert set(frame["churn"].dropna().unique()) == {0, 1}


def test_validation_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(pd.DataFrame({"churn": [0, 1]}))
