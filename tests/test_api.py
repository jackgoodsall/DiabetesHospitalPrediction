"""
Tests for the FastAPI prediction endpoints.

The model and transformer are mocked so no MLflow server or trained
artefact is needed.  The mock transformer returns a DataFrame of zeros
and the mock model always returns probability 0.8 for every record.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from api import app, model_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_PATIENT = {
    "time_in_hospital": 5,
    "num_lab_procedures": 42,
    "num_medications": 12,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_diagnoses": 7,
    "number_inpatient": 0,
    "race": "Caucasian",
    "gender": "Female",
    "age": "[50-60)",
    "weight": None,
    "change": "Ch",
    "diabetesMed": "Yes",
    "metformin": "Steady",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "glipizide": "No",
    "glyburide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "insulin": "Up",
    "glipizide-metformin": "No",
    "glyburide-metformin": "No",
    "medical_specialty": "InternalMedicine",
    "diag_1": "250.01",
    "diag_2": "401",
    "diag_3": "272",
    "A1Cresult": ">8",
}


def _make_mock_transformer(n_cols: int = 10):
    """Return a mock ColumnTransformer that produces a zero DataFrame."""
    mock = MagicMock()
    mock.transform.side_effect = lambda df: pd.DataFrame(
        np.zeros((len(df), n_cols))
    )
    return mock


def _make_mock_model(probability: float = 0.8):
    """Return a mock pyfunc model that always predicts a fixed probability."""
    mock = MagicMock()
    mock.predict.side_effect = lambda df: pd.DataFrame(
        {"prob": [probability] * len(df)}
    )
    return mock


@pytest.fixture(autouse=True)
def populate_model_store():
    """Inject mock model and transformer into the module-level store before each test."""
    model_store.clear()
    model_store["model"] = _make_mock_model(probability=0.8)
    model_store["transformer"] = _make_mock_transformer()
    model_store["model_name"] = "xgboost"
    model_store["model_stage"] = "Production"
    model_store["using_pyfunc"] = True
    yield
    model_store.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200_when_model_loaded(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_model_loaded_true(self, client):
        data = resp = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_reports_model_name_and_stage(self, client):
        data = client.get("/health").json()
        assert data["model_name"] == "xgboost"
        assert data["model_stage"] == "Production"

    def test_returns_unavailable_when_store_empty(self, client):
        model_store.clear()
        data = client.get("/health").json()
        assert data["status"] == "unavailable"
        assert data["model_loaded"] is False


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_returns_200_with_valid_payload(self, client):
        resp = client.post("/predict", json=MINIMAL_PATIENT)
        assert resp.status_code == 200

    def test_response_schema(self, client):
        data = client.post("/predict", json=MINIMAL_PATIENT).json()
        assert "readmission_probability" in data
        assert "readmitted" in data
        assert "threshold_used" in data
        assert "model_name" in data
        assert "model_stage" in data

    def test_probability_is_float_between_0_and_1(self, client):
        data = client.post("/predict", json=MINIMAL_PATIENT).json()
        prob = data["readmission_probability"]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_default_threshold_is_0_5(self, client):
        data = client.post("/predict", json=MINIMAL_PATIENT).json()
        assert data["threshold_used"] == 0.5

    def test_custom_threshold_via_query_param(self, client):
        # Mock returns 0.8; threshold 0.9 → not readmitted
        data = client.post("/predict?threshold=0.9", json=MINIMAL_PATIENT).json()
        assert data["threshold_used"] == 0.9
        assert data["readmitted"] is False

    def test_readmitted_true_when_prob_exceeds_threshold(self, client):
        # Mock returns 0.8; default threshold 0.5 → readmitted
        data = client.post("/predict", json=MINIMAL_PATIENT).json()
        assert data["readmitted"] is True

    def test_missing_required_field_returns_422(self, client):
        bad_payload = {k: v for k, v in MINIMAL_PATIENT.items() if k != "gender"}
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_wrong_type_for_numerical_field_returns_422(self, client):
        bad_payload = {**MINIMAL_PATIENT, "time_in_hospital": "not-a-number"}
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_out_of_range_time_in_hospital_returns_422(self, client):
        bad_payload = {**MINIMAL_PATIENT, "time_in_hospital": 0}
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_503_when_model_not_loaded(self, client):
        model_store.clear()
        resp = client.post("/predict", json=MINIMAL_PATIENT)
        assert resp.status_code == 503

    def test_hyphenated_alias_columns_accepted(self, client):
        payload = {**MINIMAL_PATIENT, "glipizide-metformin": "Steady"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_optional_fields_can_be_null(self, client):
        payload = {
            **MINIMAL_PATIENT,
            "race": None,
            "weight": None,
            "medical_specialty": None,
            "diag_1": None,
            "diag_2": None,
            "diag_3": None,
            "A1Cresult": None,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_returns_200_with_single_record(self, client):
        resp = client.post("/predict/batch", json=[MINIMAL_PATIENT])
        assert resp.status_code == 200

    def test_response_schema(self, client):
        data = client.post("/predict/batch", json=[MINIMAL_PATIENT]).json()
        assert "predictions" in data
        assert "total_records" in data
        assert "predicted_readmitted" in data
        assert "readmission_rate" in data

    def test_total_records_matches_input_count(self, client):
        payload = [MINIMAL_PATIENT, MINIMAL_PATIENT, MINIMAL_PATIENT]
        data = client.post("/predict/batch", json=payload).json()
        assert data["total_records"] == 3

    def test_predictions_list_length_matches_input(self, client):
        payload = [MINIMAL_PATIENT, MINIMAL_PATIENT]
        data = client.post("/predict/batch", json=payload).json()
        assert len(data["predictions"]) == 2

    def test_readmission_rate_is_fraction(self, client):
        data = client.post("/predict/batch", json=[MINIMAL_PATIENT]).json()
        rate = data["readmission_rate"]
        assert 0.0 <= rate <= 1.0

    def test_aggregate_stats_are_consistent(self, client):
        # Mock returns 0.8 → all readmitted at default threshold 0.5
        payload = [MINIMAL_PATIENT, MINIMAL_PATIENT]
        data = client.post("/predict/batch", json=payload).json()
        assert data["predicted_readmitted"] == 2
        assert data["readmission_rate"] == 1.0

    def test_empty_list_returns_400(self, client):
        resp = client.post("/predict/batch", json=[])
        assert resp.status_code == 400

    def test_503_when_model_not_loaded(self, client):
        model_store.clear()
        resp = client.post("/predict/batch", json=[MINIMAL_PATIENT])
        assert resp.status_code == 503

    def test_custom_threshold_applies_to_all_records(self, client):
        # Mock returns 0.8; threshold 0.9 → none readmitted
        payload = [MINIMAL_PATIENT, MINIMAL_PATIENT]
        data = client.post("/predict/batch?threshold=0.9", json=payload).json()
        assert data["predicted_readmitted"] == 0
        assert data["readmission_rate"] == 0.0
