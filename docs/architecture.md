# Architecture

## Flow

1. `data.py` creates or loads a CSV and checks the required schema.
2. `train.py` separates features and target, performs a stratified train/test split, fits a baseline and a logistic-regression pipeline, then evaluates the holdout set.
3. `features.py` owns preprocessing: numeric median imputation and scaling; categorical most-frequent imputation and one-hot encoding.
4. The entire fitted pipeline is serialized with joblib so inference applies the exact preprocessing learned during training.
5. `api.py` validates incoming JSON with Pydantic and delegates inference to `predict.py`.

## Why use a pipeline?

Fitting preprocessing only on the training split avoids leakage from the test set. Saving preprocessing and model together keeps training and serving transformations consistent.

## API routes

- `GET /health`: lightweight process health check; it does not guarantee the model artifact is available.
- `POST /predict`: validates feature values and returns a probability and binary decision.
- `GET /docs`: interactive OpenAPI documentation provided by FastAPI.
