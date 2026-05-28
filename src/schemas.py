"""
schemas.py
----------
Pydantic request and response models for the prediction API.

The feature names and types mirror the transformer schema defined in
configs/run_config.yaml. Any field change here must also be reflected there.
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class PatientRecord(BaseModel):
    """
    A single patient record submitted for readmission prediction.

    Numerical features are hospital-stay statistics; categorical features
    cover demographics, diagnoses, and medication indicators.
    Hyphenated column names (e.g. glipizide-metformin) are accepted via
    Pydantic aliases so the JSON key matches the original dataset column name
    while remaining a valid Python attribute.
    """

    model_config = ConfigDict(populate_by_name=True)

    # ------------------------------------------------------------------
    # Numerical features
    # ------------------------------------------------------------------
    time_in_hospital: int = Field(..., ge=1, le=14, description="Days in hospital (1–14)")
    num_lab_procedures: int = Field(..., ge=0, description="Number of lab tests performed")
    num_medications: int = Field(..., ge=0, description="Number of distinct medications")
    number_outpatient: int = Field(..., ge=0, description="Prior outpatient visits in past year")
    number_emergency: int = Field(..., ge=0, description="Prior emergency visits in past year")
    number_diagnoses: int = Field(..., ge=1, description="Number of diagnoses entered to the system")
    number_inpatient: int = Field(..., ge=0, description="Prior inpatient visits in past year")

    # ------------------------------------------------------------------
    # Categorical features — demographics & admin
    # ------------------------------------------------------------------
    race: Optional[str] = Field(None, description="Patient race (e.g. Caucasian, AfricanAmerican)")
    gender: str = Field(..., description="Patient gender (Male / Female)")
    age: str = Field(..., description="Age bracket, e.g. '[50-60)'")
    weight: Optional[str] = Field(None, description="Weight bracket; frequently missing")
    change: str = Field(..., description="Whether diabetes medication was changed (Ch / No)")
    diabetesMed: str = Field(..., description="Whether any diabetes medication was prescribed (Yes / No)")

    # ------------------------------------------------------------------
    # Categorical features — medication dosage indicators
    # Typical values: No / Steady / Up / Down
    # ------------------------------------------------------------------
    metformin: str = Field(..., description="Metformin dosage change indicator")
    repaglinide: str = Field(..., description="Repaglinide dosage change indicator")
    nateglinide: str = Field(..., description="Nateglinide dosage change indicator")
    chlorpropamide: str = Field(..., description="Chlorpropamide dosage change indicator")
    glimepiride: str = Field(..., description="Glimepiride dosage change indicator")
    glipizide: str = Field(..., description="Glipizide dosage change indicator")
    glyburide: str = Field(..., description="Glyburide dosage change indicator")
    pioglitazone: str = Field(..., description="Pioglitazone dosage change indicator")
    rosiglitazone: str = Field(..., description="Rosiglitazone dosage change indicator")
    acarbose: str = Field(..., description="Acarbose dosage change indicator")
    miglitol: str = Field(..., description="Miglitol dosage change indicator")
    troglitazone: str = Field(..., description="Troglitazone dosage change indicator")
    tolazamide: str = Field(..., description="Tolazamide dosage change indicator")
    insulin: str = Field(..., description="Insulin dosage change indicator")

    # ------------------------------------------------------------------
    # Categorical features — combination medications (hyphenated names)
    # ------------------------------------------------------------------
    glipizide_metformin: Optional[str] = Field(
        None, alias="glipizide-metformin",
        description="Glipizide-metformin combination indicator"
    )
    glyburide_metformin: Optional[str] = Field(
        None, alias="glyburide-metformin",
        description="Glyburide-metformin combination indicator"
    )

    # ------------------------------------------------------------------
    # Categorical features — clinical
    # ------------------------------------------------------------------
    medical_specialty: Optional[str] = Field(None, description="Admitting physician specialty")
    diag_1: Optional[str] = Field(None, description="Primary diagnosis ICD-9 code")
    diag_2: Optional[str] = Field(None, description="Secondary diagnosis ICD-9 code")
    diag_3: Optional[str] = Field(None, description="Additional diagnosis ICD-9 code")
    A1Cresult: Optional[str] = Field(None, description="HbA1c test result (>8, >7, Norm, None)")


class PredictionResponse(BaseModel):
    """Prediction result for a single patient record."""

    readmission_probability: float = Field(..., ge=0.0, le=1.0)
    readmitted: bool
    threshold_used: float
    model_name: str
    model_stage: str


class BatchPredictionResponse(BaseModel):
    """Aggregated results for a batch prediction request."""

    predictions: list[PredictionResponse]
    total_records: int
    predicted_readmitted: int
    readmission_rate: float


class HealthResponse(BaseModel):
    """Server and model readiness information."""

    status: str
    model_loaded: bool
    model_name: Optional[str]
    model_stage: Optional[str]
