from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "trained_model.pkl"
RESULTS_PATH = ARTIFACTS_DIR / "results.json"


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., example=7.4)
    volatile_acidity: float = Field(..., example=0.7)
    citric_acid: float = Field(..., example=0.0)
    residual_sugar: float = Field(..., example=1.9)
    chlorides: float = Field(..., example=0.076)
    free_sulfur_dioxide: float = Field(..., example=11.0)
    total_sulfur_dioxide: float = Field(..., example=34.0)
    density: float = Field(..., example=0.9978)
    pH: float = Field(..., example=3.51)
    sulphates: float = Field(..., example=0.56)
    alcohol: float = Field(..., example=9.4)


class PredictionResponse(BaseModel):
    name: str
    roll_no: str
    wine_quality: int


def load_metadata() -> dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "results.json not found. Download it from the best Lab 2 workflow run and "
            "place it inside the artifacts folder."
        )
    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model() -> object:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "trained_model.pkl not found. Download it from the best Lab 2 workflow run "
            "and place it inside the artifacts folder."
        )
    return joblib.load(MODEL_PATH)


app = FastAPI(title="Wine Quality Inference API", version="1.0.0")


@app.get("/")
def root() -> dict:
    return {"message": "Wine Quality Inference API is running"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def run_prediction(features: WineFeatures) -> PredictionResponse:
    try:
        model = load_model()
        metadata = load_metadata()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    row = [[
        features.fixed_acidity,
        features.volatile_acidity,
        features.citric_acid,
        features.residual_sugar,
        features.chlorides,
        features.free_sulfur_dioxide,
        features.total_sulfur_dioxide,
        features.density,
        features.pH,
        features.sulphates,
        features.alcohol,
    ]]

    prediction = model.predict(row)[0]
    return PredictionResponse(
        name=metadata.get("student_name", "Your Name"),
        roll_no=metadata.get("roll_no", "Your Roll Number"),
        wine_quality=int(round(float(prediction))),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_post(features: WineFeatures) -> PredictionResponse:
    return run_prediction(features)


@app.get("/predict", response_model=PredictionResponse)
def predict_get(
    fixed_acidity: Annotated[float, Query(...)] ,
    volatile_acidity: Annotated[float, Query(...)] ,
    citric_acid: Annotated[float, Query(...)] ,
    residual_sugar: Annotated[float, Query(...)] ,
    chlorides: Annotated[float, Query(...)] ,
    free_sulfur_dioxide: Annotated[float, Query(...)] ,
    total_sulfur_dioxide: Annotated[float, Query(...)] ,
    density: Annotated[float, Query(...)] ,
    pH: Annotated[float, Query(...)] ,
    sulphates: Annotated[float, Query(...)] ,
    alcohol: Annotated[float, Query(...)] ,
) -> PredictionResponse:
    features = WineFeatures(
        fixed_acidity=fixed_acidity,
        volatile_acidity=volatile_acidity,
        citric_acid=citric_acid,
        residual_sugar=residual_sugar,
        chlorides=chlorides,
        free_sulfur_dioxide=free_sulfur_dioxide,
        total_sulfur_dioxide=total_sulfur_dioxide,
        density=density,
        pH=pH,
        sulphates=sulphates,
        alcohol=alcohol,
    )
    return run_prediction(features)

