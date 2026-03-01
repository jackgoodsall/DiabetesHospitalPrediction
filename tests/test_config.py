"""
Tests for components/config.py
"""
import pytest
from pathlib import Path
from pydantic import ValidationError

from components.config import (
    DataEngineeringConfig,
    ModelTrainingConfig,
    DataInputCleaningPipeLineConfig,
    load_yaml_config,
)


# ── load_yaml_config ───────────────────────────────────────────────────────────
class TestLoadYamlConfig:

    def test_loads_model_training_config(self, config_path):
        config = load_yaml_config(config_path, ModelTrainingConfig)
        assert isinstance(config, ModelTrainingConfig)

    def test_loads_data_input_config(self, config_path):
        config = load_yaml_config(config_path, DataInputCleaningPipeLineConfig)
        assert isinstance(config, DataInputCleaningPipeLineConfig)

    def test_missing_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_yaml_config(Path("configs/does_not_exist.yaml"), ModelTrainingConfig)

    def test_model_training_config_has_required_sections(self, config_path):
        config = load_yaml_config(config_path, ModelTrainingConfig)
        assert "model_names" in config.model
        assert "experiment_name" in config.mlflow_information

    def test_data_input_config_has_file_information(self, config_path):
        config = load_yaml_config(config_path, DataInputCleaningPipeLineConfig)
        assert "data_dir_path" in config.file_information
        assert "raw_data_file_name" in config.file_information


# ── ModelTrainingConfig ────────────────────────────────────────────────────────
class TestModelTrainingConfig:

    def test_valid_config_instantiates(self):
        config = ModelTrainingConfig(
            model={"model_names": ["xgboost"]},
            mlflow_information={"experiment_name": "test"},
            data={"target_column_name": "readmitted"},
        )
        assert config.model["model_names"] == ["xgboost"]

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            ModelTrainingConfig(
                model={"model_names": ["xgboost"]},
                # missing mlflow_information and data
            )

    def test_extra_fields_allowed(self):
        """extra='allow' means unknown keys don't raise."""
        config = ModelTrainingConfig(
            model={"model_names": ["xgboost"]},
            mlflow_information={"experiment_name": "test"},
            data={},
            xgboost={"n_estimators": 100},
        )
        assert config.model_extra["xgboost"]["n_estimators"] == 100


# ── DataEngineeringConfig ──────────────────────────────────────────────────────
class TestDataEngineeringConfig:

    def test_valid_config_instantiates(self):
        config = DataEngineeringConfig(
            numerical_features=["a", "b"],
            categorical_features=["c"],
            numerical_imputer_strat="mean",
            categorical_imputer_strat="most_frequent",
            scaler="standard",
            remainder="drop",
        )
        assert config.scaler == "standard"

    def test_invalid_scaler_raises(self):
        with pytest.raises((ValueError, TypeError)):
            DataEngineeringConfig(
                numerical_features=["a"],
                categorical_features=["b"],
                numerical_imputer_strat="mean",
                categorical_imputer_strat="most_frequent",
                scaler="invalid_scaler",  # not in Literal["standard", "minmax"]
                remainder="drop",
            )

    def test_validate_schema_passes_with_correct_df(self):
        import pandas as pd
        config = DataEngineeringConfig(
            numerical_features=["a"],
            categorical_features=["b"],
            numerical_imputer_strat="mean",
            categorical_imputer_strat="most_frequent",
            scaler="standard",
            remainder="drop",
        )
        df = pd.DataFrame({"a": [1.0], "b": ["x"]})
        config.validate_schema(df)  # should not raise

    def test_validate_schema_raises_on_missing_column(self):
        import pandas as pd
        config = DataEngineeringConfig(
            numerical_features=["a", "missing_col"],
            categorical_features=["b"],
            numerical_imputer_strat="mean",
            categorical_imputer_strat="most_frequent",
            scaler="standard",
            remainder="drop",
        )
        df = pd.DataFrame({"a": [1.0], "b": ["x"]})
        with pytest.raises(ValueError, match="Not all features are in the dataframe"):
            config.validate_schema(df)

    def test_config_is_immutable(self):
        """frozen=True means attributes cannot be reassigned."""
        config = DataEngineeringConfig(
            numerical_features=["a"],
            categorical_features=["b"],
            numerical_imputer_strat="mean",
            categorical_imputer_strat="most_frequent",
            scaler="standard",
            remainder="drop",
        )
        with pytest.raises((TypeError, AttributeError)):
            config.scaler = "minmax"
