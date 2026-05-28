"""
Tests for Pandera data validation schemas.

Builds minimal synthetic DataFrames that mimic the raw and cleaned shapes,
then checks that valid data passes and specific violations raise ValueError.
"""

import numpy as np
import pandas as pd
import pandera.errors
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from components.data_validation import CleanedDataSchema, RawDataSchema

# Pandera raises SchemaErrors when calling .validate() directly;
# our pipeline wraps these into ValueError via _validate().
_SCHEMA_ERROR = pandera.errors.SchemaErrors


# ---------------------------------------------------------------------------
# Helpers — minimal valid DataFrames
# ---------------------------------------------------------------------------

def _raw_row(**overrides) -> dict:
    """Return a single valid raw-data row, with optional field overrides."""
    base = {
        "time_in_hospital": 5,
        "num_lab_procedures": 42,
        "num_medications": 12,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_diagnoses": 7,
        "number_inpatient": 0,
        "readmitted": "NO",
        "gender": "Female",
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
        "race": "Caucasian",
        "age": "[50-60)",
        "weight": None,
        "medical_specialty": "InternalMedicine",
        "diag_1": "250.01",
        "diag_2": "401",
        "diag_3": "272",
        "A1Cresult": ">8",
    }
    base.update(overrides)
    return base


def _cleaned_row(**overrides) -> dict:
    """Return a single valid cleaned-data row."""
    base = {
        "time_in_hospital": 5,
        "num_lab_procedures": 42,
        "num_medications": 12,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_diagnoses": 7,
        "number_inpatient": 0,
        "readmitted": 1,
        "gender": "Female",
        "change": "Ch",
        "diabetesMed": "Yes",
        "insulin": "Up",
        "race": "Caucasian",
        "age": "[50-60)",
        "weight": None,
        "medical_specialty": "InternalMedicine",
        "diag_1": "250.01",
        "diag_2": "401",
        "diag_3": "272",
        "A1Cresult": ">8",
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
    }
    base.update(overrides)
    return base


def _df(row_fn, n=3, **overrides) -> pd.DataFrame:
    return pd.DataFrame([row_fn(**overrides) for _ in range(n)])


# ---------------------------------------------------------------------------
# RawDataSchema
# ---------------------------------------------------------------------------

class TestRawDataSchema:
    def test_valid_data_passes(self):
        RawDataSchema.validate(_df(_raw_row))

    def test_extra_columns_are_allowed(self):
        df = _df(_raw_row)
        df["encounter_id"] = 999
        df["patient_nbr"] = 12345
        RawDataSchema.validate(df)

    def test_time_in_hospital_below_1_fails(self):
        with pytest.raises(_SCHEMA_ERROR):
            RawDataSchema.validate(_df(_raw_row, time_in_hospital=0), lazy=True)

    def test_time_in_hospital_above_14_fails(self):
        with pytest.raises(_SCHEMA_ERROR):
            RawDataSchema.validate(_df(_raw_row, time_in_hospital=15), lazy=True)

    def test_negative_num_lab_procedures_fails(self):
        with pytest.raises(_SCHEMA_ERROR):
            RawDataSchema.validate(_df(_raw_row, num_lab_procedures=-1), lazy=True)

    def test_negative_number_diagnoses_fails(self):
        with pytest.raises(_SCHEMA_ERROR):
            RawDataSchema.validate(_df(_raw_row, number_diagnoses=0), lazy=True)

    def test_invalid_readmitted_value_fails(self):
        with pytest.raises(_SCHEMA_ERROR):
            RawDataSchema.validate(_df(_raw_row, readmitted="MAYBE"), lazy=True)

    def test_all_valid_readmitted_values_pass(self):
        for value in ["NO", ">30", "<30"]:
            RawDataSchema.validate(_df(_raw_row, readmitted=value))

    def test_invalid_gender_fails(self):
        with pytest.raises(_SCHEMA_ERROR):
            RawDataSchema.validate(_df(_raw_row, gender="Other"), lazy=True)

    def test_invalid_change_value_fails(self):
        with pytest.raises(_SCHEMA_ERROR):
            RawDataSchema.validate(_df(_raw_row, change="Yes"), lazy=True)

    def test_invalid_medication_value_fails(self):
        with pytest.raises(_SCHEMA_ERROR):
            RawDataSchema.validate(_df(_raw_row, insulin="Maybe"), lazy=True)

    def test_all_valid_medication_values_pass(self):
        for value in ["No", "Steady", "Up", "Down"]:
            RawDataSchema.validate(_df(_raw_row, insulin=value))

    def test_nullable_fields_accept_none(self):
        RawDataSchema.validate(_df(_raw_row, race=None, weight=None, medical_specialty=None))

    def test_multiple_violations_reported_together(self):
        with pytest.raises(_SCHEMA_ERROR) as exc_info:
            RawDataSchema.validate(
                _df(_raw_row, time_in_hospital=99, readmitted="WRONG"),
                lazy=True,
            )
        assert "time_in_hospital" in str(exc_info.value) or "readmitted" in str(exc_info.value)

    def test_float_numerical_values_coerced_to_int(self):
        df = _df(_raw_row)
        df["time_in_hospital"] = df["time_in_hospital"].astype(float)
        RawDataSchema.validate(df)


# ---------------------------------------------------------------------------
# CleanedDataSchema
# ---------------------------------------------------------------------------

class TestCleanedDataSchema:
    def test_valid_cleaned_data_passes(self):
        CleanedDataSchema.validate(_df(_cleaned_row))

    def test_binary_target_0_passes(self):
        CleanedDataSchema.validate(_df(_cleaned_row, readmitted=0))

    def test_binary_target_1_passes(self):
        CleanedDataSchema.validate(_df(_cleaned_row, readmitted=1))

    def test_raw_readmitted_string_fails(self):
        with pytest.raises(_SCHEMA_ERROR):
            CleanedDataSchema.validate(_df(_cleaned_row, readmitted="NO"), lazy=True)

    def test_readmitted_value_2_fails(self):
        with pytest.raises(_SCHEMA_ERROR):
            CleanedDataSchema.validate(_df(_cleaned_row, readmitted=2), lazy=True)

    def test_drop_columns_removed_check_fails_if_present(self):
        df = _df(_cleaned_row)
        df["encounter_id"] = 999  # should have been dropped
        with pytest.raises(_SCHEMA_ERROR):
            CleanedDataSchema.validate(df, lazy=True)

    def test_extra_non_forbidden_columns_allowed(self):
        df = _df(_cleaned_row)
        df["some_extra_col"] = "ok"
        CleanedDataSchema.validate(df)

    def test_time_in_hospital_out_of_range_fails(self):
        with pytest.raises(_SCHEMA_ERROR):
            CleanedDataSchema.validate(_df(_cleaned_row, time_in_hospital=0), lazy=True)

    def test_negative_numerical_feature_fails(self):
        with pytest.raises(_SCHEMA_ERROR):
            CleanedDataSchema.validate(_df(_cleaned_row, number_inpatient=-1), lazy=True)


# ---------------------------------------------------------------------------
# Integration: _validate method on BinaryReadmissionInputCleaningPipeline
# ---------------------------------------------------------------------------

class TestPipelineValidateMethod:
    """Tests the _validate wrapper that wraps SchemaErrors into ValueError."""

    def test_valid_data_does_not_raise(self):
        from unittest.mock import MagicMock
        from data_ingestion import BinaryReadmissionInputCleaningPipeline

        config = MagicMock()
        config.file_information = {}
        config.mlflow_information = {}
        config.data = {"back_end": "pandas", "validate": True}
        pipeline = BinaryReadmissionInputCleaningPipeline.__new__(
            BinaryReadmissionInputCleaningPipeline
        )
        pipeline.file_config = {}
        pipeline.mlflow_config = {}
        pipeline.data_config = {"back_end": "pandas"}
        pipeline.safe_to_run = True
        pipeline._back_end = "pandas"

        pipeline._validate(_df(_raw_row), RawDataSchema, "raw")

    def test_invalid_data_raises_value_error_with_context(self):
        from data_ingestion import BinaryReadmissionInputCleaningPipeline

        pipeline = BinaryReadmissionInputCleaningPipeline.__new__(
            BinaryReadmissionInputCleaningPipeline
        )
        pipeline.file_config = {}
        pipeline.mlflow_config = {}
        pipeline.data_config = {"back_end": "pandas"}
        pipeline.safe_to_run = True
        pipeline._back_end = "pandas"

        with pytest.raises(ValueError, match="Data validation failed"):
            pipeline._validate(
                _df(_raw_row, time_in_hospital=99),
                RawDataSchema,
                "raw",
            )
