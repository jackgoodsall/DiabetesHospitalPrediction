"""
data_validation.py
------------------
Pandera schema definitions for the two data states in the ingestion pipeline:

  RawDataSchema     — validates the CSV as loaded from disk, before any cleaning.
  CleanedDataSchema — validates after column drops and target binarisation.

Both schemas use strict=False so extra columns in the source data do not cause
failures — only the columns we actually care about are checked.

Typical values for categorical medication columns are:
  "No" | "Steady" | "Up" | "Down"
These are declared with isin checks where the cardinality is well-known.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series
from typing import Optional

# ---------------------------------------------------------------------------
# Shared value sets
# ---------------------------------------------------------------------------
_MEDICATION_VALUES = {"No", "Steady", "Up", "Down"}
_YES_NO = {"Yes", "No"}
_CHANGE_VALUES = {"Ch", "No"}
_GENDER_VALUES = {"Male", "Female", "Unknown/Invalid"}
_READMITTED_RAW = {"NO", ">30", "<30"}
_A1C_VALUES = {"None", "Norm", ">7", ">8"}


# ---------------------------------------------------------------------------
# Raw data schema — validates the CSV before any cleaning
# ---------------------------------------------------------------------------
class RawDataSchema(pa.DataFrameModel):
    """
    Expected shape of the raw diabetic_data.csv.

    Numerical range checks catch data loading errors (wrong file, corrupted
    rows). Categorical isin checks catch encoding drift or upstream changes to
    the source dataset.  nullable=True is used wherever the UCI dataset
    legitimately uses '?' or empty values.
    """

    # ------------------------------------------------------------------
    # Numerical features — ranges derived from the UCI dataset description
    # ------------------------------------------------------------------
    time_in_hospital: Series[int] = pa.Field(ge=1, le=14, coerce=True)
    num_lab_procedures: Series[int] = pa.Field(ge=0, coerce=True)
    num_medications: Series[int] = pa.Field(ge=0, coerce=True)
    number_outpatient: Series[int] = pa.Field(ge=0, coerce=True)
    number_emergency: Series[int] = pa.Field(ge=0, coerce=True)
    number_diagnoses: Series[int] = pa.Field(ge=1, coerce=True)
    number_inpatient: Series[int] = pa.Field(ge=0, coerce=True)

    # ------------------------------------------------------------------
    # Target — must be one of the three original readmission categories
    # ------------------------------------------------------------------
    readmitted: Series[str] = pa.Field(isin=_READMITTED_RAW)

    # ------------------------------------------------------------------
    # Categorical features — well-defined small cardinality sets
    # ------------------------------------------------------------------
    gender: Series[str] = pa.Field(isin=_GENDER_VALUES)
    change: Series[str] = pa.Field(isin=_CHANGE_VALUES)
    diabetesMed: Series[str] = pa.Field(isin=_YES_NO)

    # Medication dosage indicators (No / Steady / Up / Down)
    metformin: Series[str] = pa.Field(isin=_MEDICATION_VALUES)
    repaglinide: Series[str] = pa.Field(isin=_MEDICATION_VALUES)
    nateglinide: Series[str] = pa.Field(isin=_MEDICATION_VALUES)
    chlorpropamide: Series[str] = pa.Field(isin=_MEDICATION_VALUES)
    glimepiride: Series[str] = pa.Field(isin=_MEDICATION_VALUES)
    glipizide: Series[str] = pa.Field(isin=_MEDICATION_VALUES)
    glyburide: Series[str] = pa.Field(isin=_MEDICATION_VALUES)
    pioglitazone: Series[str] = pa.Field(isin=_MEDICATION_VALUES)
    rosiglitazone: Series[str] = pa.Field(isin=_MEDICATION_VALUES)
    acarbose: Series[str] = pa.Field(isin=_MEDICATION_VALUES)
    miglitol: Series[str] = pa.Field(isin=_MEDICATION_VALUES)
    troglitazone: Series[str] = pa.Field(isin=_MEDICATION_VALUES)
    tolazamide: Series[str] = pa.Field(isin=_MEDICATION_VALUES)
    insulin: Series[str] = pa.Field(isin=_MEDICATION_VALUES)

    # ------------------------------------------------------------------
    # Categorical features — nullable (UCI uses '?' for missing values)
    # ------------------------------------------------------------------
    race: Optional[Series[str]] = pa.Field(nullable=True)
    age: Series[str]  # bracket format e.g. [50-60)
    weight: Optional[Series[str]] = pa.Field(nullable=True)
    medical_specialty: Optional[Series[str]] = pa.Field(nullable=True)
    diag_1: Optional[Series[str]] = pa.Field(nullable=True)
    diag_2: Optional[Series[str]] = pa.Field(nullable=True)
    diag_3: Optional[Series[str]] = pa.Field(nullable=True)
    A1Cresult: Optional[Series[str]] = pa.Field(nullable=True)

    class Config:
        strict = False   # extra columns (IDs, dropped cols) are allowed
        coerce = False   # only coerce where explicitly declared per-field


# ---------------------------------------------------------------------------
# Cleaned data schema — validates after column drops + target binarisation
# ---------------------------------------------------------------------------
class CleanedDataSchema(pa.DataFrameModel):
    """
    Expected shape after BinaryReadmissionInputCleaningPipeline.run_pipeline().

    The drop columns are gone, and the target has been converted to binary int.
    """

    # Numerical features — same ranges as raw
    time_in_hospital: Series[int] = pa.Field(ge=1, le=14, coerce=True)
    num_lab_procedures: Series[int] = pa.Field(ge=0, coerce=True)
    num_medications: Series[int] = pa.Field(ge=0, coerce=True)
    number_outpatient: Series[int] = pa.Field(ge=0, coerce=True)
    number_emergency: Series[int] = pa.Field(ge=0, coerce=True)
    number_diagnoses: Series[int] = pa.Field(ge=1, coerce=True)
    number_inpatient: Series[int] = pa.Field(ge=0, coerce=True)

    # Target is now binary 0 / 1
    readmitted: Series[int] = pa.Field(isin=[0, 1], coerce=True)

    # Key categoricals still present after cleaning
    gender: Series[str] = pa.Field(isin=_GENDER_VALUES)
    change: Series[str] = pa.Field(isin=_CHANGE_VALUES)
    diabetesMed: Series[str] = pa.Field(isin=_YES_NO)
    insulin: Series[str] = pa.Field(isin=_MEDICATION_VALUES)

    # Columns that must have been removed by the cleaning step
    @pa.dataframe_check
    @classmethod
    def drop_columns_removed(cls, df: pa.typing.DataFrame) -> bool:
        """Fail if any of the expected drop columns are still present.

        Note: patient_nbr is deliberately NOT forbidden here. It is retained
        through cleaning so the pipeline can perform a group-aware (leak-free)
        train/test split on it, and is dropped only after splitting.
        """
        forbidden = {
            "encounter_id", "admission_type_id",
            "discharge_disposition_id", "admission_source_id",
            "max_glu_serum", "acetohexamide", "tolbutamide", "examide",
            "citoglipton", "glimepiride-pioglitazone",
            "metformin-rosiglitazone", "metformin-pioglitazone",
        }
        remaining = forbidden & set(df.columns)
        if remaining:
            raise ValueError(f"Expected drop columns still present: {remaining}")
        return True

    class Config:
        strict = False
        coerce = False
