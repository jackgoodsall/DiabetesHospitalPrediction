from pydantic import ConfigDict, BaseModel, ValidationError, field_validator
from typing import Literal, Any, Dict, List, Optional
from dataclasses import dataclass
import yaml
import pandas as pd
from pathlib import Path

# Define engineering config options
NumericalImputer = Literal["mean", "median", "most_frequent", "constant"]
CategoricalImputer = Literal["most_frequent", "constant"]
ScalerType = Literal["standard", "minmax"]


# Model training config
class ModelTrainingConfig(BaseModel):
    ## Basic overhead schema for the data_engineering_config
    model_config = ConfigDict(extra="allow")

    model: Dict[str, Any]
    mlflow_information: Dict[str, Any]
    data: Dict[str, Any]


## Config for the Data Input and Cleaning pipeline 
class DataInputCleaningPipeLineConfig(BaseModel):
    ## Basic overhead schema for the data_engineering_config
    model_config = ConfigDict(extra="allow")

    file_information: Dict[str, Any]
    mlflow_information: Dict[str, Any]
    data: Dict[str, Any]


@dataclass(frozen=True)
class DataEngineeringConfig:
    numerical_features: List[str]
    categorical_features: List[str]
    numerical_imputer_strat: NumericalImputer
    categorical_imputer_strat: CategoricalImputer
    scaler: ScalerType
    remainder: Literal["drop", "passthrough"]

    def __post_init__(self) -> None:
        valid_scalers = {"standard", "minmax"}
        if self.scaler not in valid_scalers:
            raise ValueError(f"Invalid scaler '{self.scaler}'. Must be one of {valid_scalers}")
        valid_remainders = {"drop", "passthrough"}
        if self.remainder not in valid_remainders:
            raise ValueError(f"Invalid remainder '{self.remainder}'. Must be one of {valid_remainders}")

    def validate_schema(self, df: pd.DataFrame) -> None:
        combined_features = set(self.numerical_features + self.categorical_features)
        dataframe_features = set(df.columns)
        if not combined_features <= dataframe_features:
            raise ValueError("Not all features are in the dataframe")



class TuningConfig(BaseModel):
    enabled: bool = False
    n_trials: int = 50
    metric: str = "pr_auc"
    direction: Literal["maximize", "minimize"] = "maximize"
    cv_folds: int = 3
    timeout: Optional[int] = None  # seconds; None = no timeout
    show_progress_bar: bool = False

    @field_validator("metric")
    @classmethod
    def _valid_metric(cls, v: str) -> str:
        allowed = {"pr_auc", "auc_roc", "f1", "brier_score"}
        if v not in allowed:
            raise ValueError(f"tuning.metric must be one of {allowed}, got '{v}'")
        return v


def load_yaml_config(config_path : Path,
                     config_type):
    ### Loads the different yaml configs
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    try:
        return config_type(**config)
    except ValidationError as e:
        raise ValueError(f"Configuration missing required sections:\n{e}")
    except:
        raise Exception




