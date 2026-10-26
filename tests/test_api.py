from fastapi.testclient import TestClient

from src.applied_ml import api
from src.applied_ml.data import generate_dataset, FEATURE_COLUMNS, TARGET_COLUMN
from src.applied_ml.features import build_pipeline

client = TestClient(api.app)


def setup_module():
    frame = generate_dataset(n_samples=400, random_state=4)
    model = build_pipeline().fit(frame[FEATURE_COLUMNS], frame[TARGET_COLUMN])
    api.get_model.cache_clear()
    api.get_model = lambda: model


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_probability():
    response = client.post("/predict", json={
        "tenure_months": 8,
        "monthly_charges": 79.5,
        "contract_type": "month_to_month",
        "support_tickets": 3,
        "payment_method": "electronic",
    })
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("likely_to_churn", "unlikely_to_churn")


def test_predict_endpoint_validates_input():
    response = client.post("/predict", json={
        "tenure_months": -2,
        "monthly_charges": 79.5,
        "contract_type": "invalid",
        "support_tickets": 3,
        "payment_method": "electronic",
    })
    assert response.status_code == 422
