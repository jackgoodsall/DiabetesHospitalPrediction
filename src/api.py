"""
api.py
------
FastAPI application for the diabetes readmission prediction service.

Model loading strategy is controlled by environment variables:

  Registry (default):
    MODEL_NAME   — registered model name in MLflow (default: xgboost)
    MODEL_STAGE  — registry stage to load (default: Production)

  Local files (set both to bypass MLflow):
    MODEL_PATH       — path to a .joblib model file
    TRANSFORMER_PATH — path to a .joblib transformer file

The model and transformer are loaded once at server startup via the
lifespan context manager and reused for every request.

Endpoints:
  GET  /health          — liveness / readiness check
  POST /predict         — single-record prediction
  POST /predict/batch   — batch prediction with aggregate stats
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from mlflow import MlflowClient

from schemas import (
    BatchPredictionResponse,
    HealthResponse,
    PatientRecord,
    PredictionResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "xgboost")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MODEL_PATH = os.getenv("MODEL_PATH")
TRANSFORMER_PATH = os.getenv("TRANSFORMER_PATH")

# Module-level store — populated at startup, read on every request
model_store: dict = {}


# ---------------------------------------------------------------------------
# Lifespan — load once at startup, release at shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    model_store.clear()
    logger.info("Model store cleared on shutdown")


def _load_model() -> None:
    """Populate model_store with model and transformer."""
    using_local = MODEL_PATH and TRANSFORMER_PATH

    if using_local:
        logger.info(f"Loading model from local path: {MODEL_PATH}")
        model_store["model"] = joblib.load(MODEL_PATH)
        model_store["transformer"] = joblib.load(TRANSFORMER_PATH)
        model_store["model_name"] = os.path.basename(MODEL_PATH)
        model_store["model_stage"] = "local"
        model_store["using_pyfunc"] = False
    else:
        logger.info(f"Loading '{MODEL_NAME}' at stage '{MODEL_STAGE}' from MLflow registry")
        client = MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not versions:
            raise RuntimeError(
                f"No model version found for '{MODEL_NAME}' at stage '{MODEL_STAGE}'. "
                "Train a model and promote it, or set MODEL_PATH / TRANSFORMER_PATH."
            )

        version = versions[0]
        run_id = version.run_id
        logger.info(f"Found v{version.version} (run_id={run_id})")

        model_store["model"] = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=f"preprocessor/{MODEL_NAME}_transformer.joblib",
        )
        model_store["transformer"] = joblib.load(local_path)
        model_store["model_name"] = MODEL_NAME
        model_store["model_stage"] = MODEL_STAGE
        model_store["using_pyfunc"] = True

    logger.info("Model and transformer loaded successfully")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Diabetes Readmission Prediction API",
    description=(
        "Predicts 30-day hospital readmission risk for diabetic patients. "
        "Uses an XGBoost model trained on the UCI Diabetes 130-US Hospitals dataset "
        "with an sklearn ColumnTransformer preprocessing pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _patient_to_dataframe(patient: PatientRecord) -> pd.DataFrame:
    """Convert a PatientRecord to a single-row DataFrame with original column names."""
    return pd.DataFrame([patient.model_dump(by_alias=True)])


def _patients_to_dataframe(patients: list[PatientRecord]) -> pd.DataFrame:
    """Convert a list of PatientRecords to a multi-row DataFrame."""
    return pd.DataFrame([p.model_dump(by_alias=True) for p in patients])


def _predict_probabilities(df: pd.DataFrame) -> np.ndarray:
    """Apply transformer then model; return 1-D array of readmission probabilities."""
    transformer = model_store["transformer"]
    model = model_store["model"]

    X = transformer.transform(df).to_numpy()

    if model_store["using_pyfunc"]:
        prob_df = model.predict(pd.DataFrame(X))
        return np.array(prob_df).ravel()
    else:
        return model.predict_proba(X)[:, 1]


def _check_model_loaded() -> None:
    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness and readiness check. Returns 200 when the model is loaded."""
    loaded = "model" in model_store
    return HealthResponse(
        status="healthy" if loaded else "unavailable",
        model_loaded=loaded,
        model_name=model_store.get("model_name"),
        model_stage=model_store.get("model_stage"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientRecord, threshold: float = 0.5) -> PredictionResponse:
    """
    Predict 30-day readmission risk for a single patient record.

    - **threshold**: decision boundary for the binary label (default 0.5).
      Lower values increase sensitivity (more patients flagged as at-risk).
    """
    _check_model_loaded()

    try:
        df = _patient_to_dataframe(patient)
        probabilities = _predict_probabilities(df)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    probability = float(probabilities[0])
    return PredictionResponse(
        readmission_probability=probability,
        readmitted=probability >= threshold,
        threshold_used=threshold,
        model_name=model_store["model_name"],
        model_stage=model_store["model_stage"],
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(
    patients: list[PatientRecord],
    threshold: float = 0.5,
) -> BatchPredictionResponse:
    """
    Predict 30-day readmission risk for multiple patient records in one call.

    More efficient than calling `/predict` N times — preprocessing and
    inference are vectorised across the entire batch.

    - **threshold**: decision boundary applied to every record (default 0.5).
    """
    _check_model_loaded()

    if not patients:
        raise HTTPException(status_code=400, detail="Request body must contain at least one record.")

    try:
        df = _patients_to_dataframe(patients)
        probabilities = _predict_probabilities(df)
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    predictions = [
        PredictionResponse(
            readmission_probability=float(prob),
            readmitted=float(prob) >= threshold,
            threshold_used=threshold,
            model_name=model_store["model_name"],
            model_stage=model_store["model_stage"],
        )
        for prob in probabilities
    ]

    readmitted_count = sum(1 for p in predictions if p.readmitted)
    return BatchPredictionResponse(
        predictions=predictions,
        total_records=len(predictions),
        predicted_readmitted=readmitted_count,
        readmission_rate=readmitted_count / len(predictions),
    )
