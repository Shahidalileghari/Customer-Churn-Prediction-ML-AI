"""FastAPI application for customer churn inference."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.applied_ml.config import MODEL_PATH, MODEL_VERSION
from src.applied_ml.predict import load_model, predict_one

app = FastAPI(
    title="Customer Churn Prediction API",
    version=MODEL_VERSION,
    description="Demo API for predicting customer churn from structured features.",
)


class PredictionRequest(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=1000)
    contract_type: str = Field(pattern=r"^(month_to_month|one_year|two_year)$")
    support_tickets: int = Field(ge=0, le=100)
    payment_method: str = Field(pattern=r"^(electronic|bank_transfer|credit_card)$")


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: int
    label: str
    model_version: str


@lru_cache(maxsize=1)
def get_model():
    path = Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))
    return load_model(path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "customer-churn-api", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict:
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result = predict_one(model, request.model_dump())
    result["model_version"] = MODEL_VERSION
    return result
