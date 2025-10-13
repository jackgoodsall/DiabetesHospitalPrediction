import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ValidationError
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

NumericalImputer = Literal["mean", "median", "most_frequent", "constant"]
CategoricalImputer = Literal["most_frequent", "constant"]
ScalerType = Literal["standard", "minmax"]

@dataclass(frozen = True)
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

class DataEngineeringPipeLine:
    """
    Defines a pipeline for data cleaning/scaling
    """
    def __init__(self, data_eng_config: DataEngineeringConfig) -> None:
        self.config = data_eng_config
        self.numerical_features = data_eng_config.numerical_features
        self.categorical_features = data_eng_config.categorical_features
        self.is_fit = False

        logger.debug("Initailising dataframe pipeline with config", self.config)
        self._build_numerical_pipeline()
        self._build_categorical_pipeline()
        self._build_pipeline()

    ## Functions to build the pipeline
    ## Can be overridden in a daughter class if want to easily define a new one
    def _build_numerical_pipeline(self) -> None:

        if self.config.scaler == "standard":
            scaler = StandardScaler()
        else:  # "minmax"
            scaler = MinMaxScaler()

        numerical_pipeline = Pipeline([
                ("num_imputer" , SimpleImputer(strategy = self.config.numerical_imputer_strat)),
                ("scaler", scaler)                    
            ])
        self._numerical_pipeline = numerical_pipeline
        logger.debug("Built numerical pipeline", self._numerical_pipeline)

    def _build_categorical_pipeline(self) -> None:
        categorical_pipeline = Pipeline([
            ("cat_imputer", SimpleImputer(st= self.config.categorical_imputer_strat)),
            ("one_hot", OneHotEncoder()) 
            
        ])
        self._categorical_pipeline = categorical_pipeline
        logger.debug("Built categorical pipeline", self._categorical_pipeline)
        self._pipeline.set_output(transform="pandas")
        logger.debug("Built full column transformer: %s", self._pipeline)


    def _build_pipeline(self) -> None:
        self._pipeline = ColumnTransformer([
            (("num pipeline", self._numerical_pipeline, self.numerical_features),
            ("cat pipeline", self._categorical_pipeline, self.categorical_features)),
            
        ], remainder = self.config.remainder)
        logger.debug("Built full pipeline", self._pipeline)

    ## Functions to fit, transform and fit transform
    def fit(self, data: pd.DataFrame):
        
        self.config.validate_schema(data)
        logger.info("Data was valid with the config and fitting the pipeline")
        self._pipeline = self._pipeline.fit(data)
        self.is_fit = True
      
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fit:
            raise NotFittedError("Pipeline is not fit")
        return self._pipeline.transform(data)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)
        