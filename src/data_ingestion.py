import logging
import os
import sys
from typing import Any, Dict, List, Tuple
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ValidationError, ConfigDict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import abc

from components.config import DataInputCleaningPipeLineConfig

logger = logging.getLogger(__name__)


class DataInputCleaningPipeline(abc.ABC):
    """
    Abstract META class for the input and cleaning pipeline portion.

    Provides required functionality that needs to be implemented for this pipeline.
    """

    @abc.abstractmethod
    def run_pipeline(self):
        raise NotImplementedError()


    @abc.abstractmethod
    def _load_data(self):
        raise NotImplementedError()


class BinaryReadmissionInputCleaningPipeline(DataInputCleaningPipeline):
    """
    Class for the data input and cleaning portion of the pipeline for the binary 
    classification task defined in the readme.

    
    """

    ## defines a mapping from backend to data loading process
    ## in this class
    __data_loading_mapping__ = {
        "pandas" : "load_data_to_pandas"
        }

    def __init__(self, config: DataInputCleaningPipeLineConfig):
        if type(config) != DataInputCleaningPipeLineConfig:
            raise TypeError()

        self.file_config = config.file_information or None
        self.mlflow_config = config.mlflow_information or None
        self.data_config = config.data or None

        self._back_end = self.data_config.get("back_end", "pandas").strip().lower()

        logger.info("Input and Cleaning pipeline module loaded")
        self.safe_to_run = True

    def load_data_to_pandas(self) -> pd.DataFrame:
        ### Loads data from file into a pandas dataframe
        self.file_path = os.path.join(
            self.file_config["data_dir_path"],
            self.file_config["raw_data_path"],
            self.file_config["raw_data_file_name"],
        )
        if os.path.exists(self.file_path):
            logging.info(msg="Raw file path exist and has been loaded")
            return pd.read_csv(self.file_path)
        else:
            logging.info(msg="Raw file path did not exist")
            raise FileNotFoundError("File does not exist")

    def transform_target_to_binary(self, data: pd.DataFrame) -> pd.DataFrame:
        target = self.data_config["target_column_name"]
        mapping = {
            "NO": 0,
            ">30": 1,
            "<30": 1,
        }
        data[target] = data[target].map(mapping)
        return data

    def remove_columns(self, data: pd.DataFrame, use_mlflow_logging = False) -> pd.DataFrame:
        columns_to_drop = self.data_config["drop_columns"]
        if use_mlflow_logging:
            mlflow.log_param("dropped columns", columns_to_drop)
        return data.drop(columns=columns_to_drop)

    def _load_data(self):
        method_name = self.__data_loading_mapping__.get(
            self._back_end
        )
        method = getattr(self, method_name)
        data = method()
        return data

    def _save_data_artefact(self, data, save_path):
        if self._back_end == "pandas":
            data.to_csv(save_path)
     
    def run_pipeline(self):
        ## Checks if mlflow logging to check if should use mlflow logging
        use_mlflow_logging = mlflow.active_run() is not None 
        if use_mlflow_logging:
            logger.info("Detected higher level run, will use mlflow logging aswell")
  
        if not self.safe_to_run:
            logger.info("Failed to start pipeline")

        try:
            data = self._load_data()
        except FileNotFoundError as e:
            logger.warning(f"File not found error {e}")
            raise FileNotFoundError(f"Could not find {self.file_path}")


        logger.info("Transforming target to binary class")
        data = self.transform_target_to_binary(data)

        logger.info("Removing unneeded columns defined in config")
        data = self.remove_columns(data, use_mlflow_logging)

        logger.info("Removed columns from the dataframe defined in the config")
        if self.file_config.get("save_cleaned_data", False):
            save_path = Path(
                self.file_config["data_dir_path"],
                self.file_config["processed_data_path"],
                self.file_config["processed_data_file_save_name"],
            )
            self._save_data_artefact(data, save_path)
            logger.info(f"Wrote data back out to {save_path}")
        
        return data
