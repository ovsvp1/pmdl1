"""Stage 3: Model API — serves predictions from the latest trained model."""
import os
import threading
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/model.pkl")

app = FastAPI(title="Titanic Survival Prediction API")

_lock = threading.Lock()
_model = None
_model_mtime = None


class PassengerFeatures(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Ticket class (1, 2, or 3)")
    Sex: str = Field(..., description="'male' or 'female'")
    Age: float = Field(..., ge=0, le=100)
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Fare: float = Field(..., ge=0)
    Embarked: str = Field(..., description="Port of embarkation: 'C', 'Q', or 'S'")


class PredictionResponse(BaseModel):
    survived: bool
    survival_probability: float


def get_model():
    """Load the model, reloading it if the file on disk has changed."""
    global _model, _model_mtime
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model not available yet. Try again shortly.")

    mtime = os.path.getmtime(MODEL_PATH)
    with _lock:
        if _model is None or mtime != _model_mtime:
            _model = joblib.load(MODEL_PATH)
            _model_mtime = mtime
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_available": os.path.exists(MODEL_PATH)}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PassengerFeatures):
    model = get_model()
    row = pd.DataFrame([features.model_dump()])
    probability = float(model.predict_proba(row)[0][1])
    return PredictionResponse(survived=probability >= 0.5, survival_probability=probability)


@app.post("/predict_batch", response_model=List[PredictionResponse])
def predict_batch(passengers: List[PassengerFeatures]):
    model = get_model()
    rows = pd.DataFrame([p.model_dump() for p in passengers])
    probabilities = model.predict_proba(rows)[:, 1]
    return [
        PredictionResponse(survived=bool(p >= 0.5), survival_probability=float(p))
        for p in probabilities
    ]
