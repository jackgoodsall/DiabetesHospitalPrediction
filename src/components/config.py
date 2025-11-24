from pydantic import ConfigDict, BaseModel, ValidationError
from typing import Literal, Any, Dict, List
from dataclasses import dataclass
import yaml
import pandas as pd


# Model training config
class ModelTrainingConfig(BaseModel):
    ## Basic overhead schema for the data_engineering_config
    model_config = ConfigDict(extra="allow")

    model: Dict[str, Any]
    mlflow_information: Dict[str, Any]
    data: Dict[str, Any]


def load_yaml_config(config_path) -> ModelTrainingConfig:
    ### Loads the different yaml configs
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    try:
        return ModelTrainingConfig(**config)
    except ValidationError as e:
        raise ValueError(f"Configuration missing required sections:\n{e}")


# Define engineering config options
NumericalImputer = Literal["mean", "median", "most_frequent", "constant"]
CategoricalImputer = Literal["most_frequent", "constant"]
ScalerType = Literal["standard", "minmax"]


@dataclass(frozen=True)
class DataEngineeringConfig:
    numerical_features: List[str]
    categorical_features: List[str]
    numerical_imputer_strat: NumericalImputer
    categorical_imputer_strat: CategoricalImputer
    scaler: ScalerType
    remainder: Literal["drop", "passthrough"]

    def validate_schema(self, df: pd.DataFrame) -> None:
        combined_features = set(self.numerical_features + self.categorical_features)
        dataframe_features = set(df.columns)
        if not combined_features <= dataframe_features:
            raise ValueError("Not all features are in the dataframe")
