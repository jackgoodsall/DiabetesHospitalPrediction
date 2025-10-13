import os 
import sys
import yaml
from pydantic import BaseModel, ValidationError
from typing import Dict, Any
import logging
from dataclasses import dataclass

import mlflow
from mlflow.models.signature import infer_signature

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from typing import Tuple, List


import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.exceptions import NotFittedError

@dataclass
class DataEngineeringConfig:
    numerical_features: List[str]
    categorical_featutres: List[str]



class DataEngineeringPipeLine:
    """
    Defines a pipeline for data cleaning/scaling
    """
    def __init__(self, data_eng_config: DataEngineeringConfig) -> None:
        self.config = data_eng_config
        self.numerical_features = data_eng_config.numerical_features
        self.categorical_features = data_eng_config.categorical_featutres
        self.is_fit = False

    def _build_numerical_pipeline(self) -> None:
        numerical_pipeline = Pipeline([
                ("num_imputer" , SimpleImputer(strategy = "mean")),
                ("standard_scaler", StandardScaler())                    
            ])
        self._numerical_pipeline = numerical_pipeline

    def _build_categorical_pipeline(self) -> None:
        categorical_pipeline = Pipeline([
            ("cat_imputer", SimpleImputer("most_"))
            ("one_hot", OneHotEncoder())
        ])
        self._categorical_pipeline = categorical_pipeline
    
    def _build_pipeline(self) -> None:
        self._pipeline = ColumnTransformer([
            ("num pipeline", self._numerical_pipeline, self.numerical_features),
            ("cat pipeline", self._categorical_pipeline, self.categorical_features)
        ])

    def fit(self, data: pd.DataFrame):
        self._pipeline = self._pipeline.fit(data)
        self.is_fit = True
      
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fit:
            raise NotFittedError("Pipeline is not fit")
        return self._pipeline.transform(data)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)
        