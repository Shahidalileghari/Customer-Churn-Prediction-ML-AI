# Applied ML Project — Customer Churn Prediction

An end-to-end machine learning project demonstrating data acquisition, cleaning, exploratory analysis, model training, evaluation, reproducible artifacts, and a REST API for predictions.

The project uses a small, deterministic synthetic customer dataset generated locally. This keeps the repository runnable without downloading a dataset and avoids exposing personal customer information. The same pipeline can be adapted to a real CSV dataset.

## What this project demonstrates

- Data generation/loading and schema validation
- Data cleaning and preprocessing with scikit-learn pipelines
- Train/test split with stratification
- Baseline comparison and model evaluation
- Classification metrics: precision, recall, F1, ROC-AUC, confusion matrix
- Reproducible training with a fixed random seed
- Saving/loading a trained model artifact
- FastAPI prediction endpoint with input validation
- Unit tests for data preparation and API behavior
- Configuration through environment variables and clear documentation

## Tech stack

Python 3.11+, pandas, NumPy, scikit-learn, joblib, FastAPI, Pydantic, Uvicorn, pytest, HTTPX.

## Quick start

### 1. Create an environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Generate data and train the model

```bash
python -m src.applied_ml.data --output data/raw/customers.csv
python -m src.applied_ml.train --data data/raw/customers.csv
```

Training writes the model to `artifacts/churn_pipeline.joblib` and metrics to `artifacts/metrics.json`.

### 3. Start the API

```bash
uvicorn src.applied_ml.api:app --reload
```

Open:
- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

The API expects a trained artifact. Run training before starting the service.

### 4. Make a prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 8,
    "monthly_charges": 79.5,
    "contract_type": "month_to_month",
    "support_tickets": 3,
    "payment_method": "electronic"
  }'
```

Example response:

```json
{
  "churn_probability": 0.73,
  "prediction": 1,
  "label": "likely_to_churn",
  "model_version": "1.0.0"
}
```

Exact probability depends on the installed scikit-learn version and trained model.

## Run tests

```bash
pytest
```

## Repository layout

```text
applied-ml-project/
├── artifacts/                 # Generated model and metrics (gitignored except .gitkeep)
├── data/
│   ├── raw/                   # Generated input CSV
│   └── processed/             # Optional cleaned dataset output
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   ├── evaluation.md
│   └── deployment.md
├── notebooks/
│   └── README.md
├── src/applied_ml/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── api.py
├── tests/
├── .gitignore
├── LICENSE
└── pyproject.toml
```

## Important notes

- The generated dataset is synthetic and is for learning/demo purposes only.
- Model performance on synthetic data does not imply real-world business performance.
- The API is a local development example, not a hardened production service.
- Before production use, add authentication/authorization, rate limiting, monitoring, model/data drift checks, secrets management, and a deployment strategy.
- Avoid committing real customer data, credentials, or generated model artifacts containing sensitive information.

## Suggested extensions

1. Replace synthetic data with a documented public dataset.
2. Add cross-validation and hyperparameter tuning.
3. Track experiments with MLflow.
4. Add a data validation layer with Pandera or Great Expectations.
5. Add Docker packaging and CI.
6. Add model explainability with permutation importance or SHAP.
7. Add monitoring for drift and prediction quality.
