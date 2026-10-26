from src.applied_ml.data import FEATURE_COLUMNS, TARGET_COLUMN, generate_dataset
from src.applied_ml.features import build_pipeline


def test_pipeline_fits_and_predicts_with_missing_values():
    frame = generate_dataset(n_samples=300, random_state=7)
    x = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN]
    model = build_pipeline()
    model.fit(x, y)
    predictions = model.predict(x.iloc[:8])
    probabilities = model.predict_proba(x.iloc[:8])[:, 1]
    assert len(predictions) == 8
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
