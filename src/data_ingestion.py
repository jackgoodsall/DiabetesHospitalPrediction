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

logger = logging.getLogger(__name__)


class DataInputCleaningPipeLineConfig(BaseModel):
    ## Basic overhead schema for the data_engineering_config
    model_config = ConfigDict(extra="allow")

    file_information: Dict[str, Any]
    mlflow_information: Dict[str, Any]
    data: Dict[str, Any]


def load_yaml_config() -> DataInputCleaningPipeLineConfig:
    ### Loads the different yaml configs
    config_path = "configs/"
    data_engineering_config = "run_config.yaml"
    with open(os.path.join(config_path + data_engineering_config), "r") as f:
        config = yaml.safe_load(f)
    try:
        return DataInputCleaningPipeLineConfig(**config)
    except ValidationError as e:
        raise ValueError(f"Configuration missing required sections:\n{e}")


class DataInputCleaningPipeLine:
    """
    Class for a Data input and cleaning pipeline, integrated with logging and
    mlflow.
    """

    def __init__(self, config=None):
        if config is None:
            logger.info("No config provided attempting to load config")
            try:
                configs = load_yaml_config()
                self.file_config = configs.file_information
                self.mlflow_config = configs.mlflow_information
                self.data_config = configs.data
                self.safe_to_run = True
                logger.info("Configuration loading succesful")
            except ValidationError as e:
                logger.warning(f"Config loading failed error message {e}")
                self.safe_to_run = False
        else:
            logger.info("Config passed, using this as config")
            self.config = config

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

    def remove_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        columns_to_drop = self.data_config["drop_columns"]
        mlflow.log_param("dropped columns", columns_to_drop)
        return data.drop(columns=columns_to_drop)

    def run_pipeline(self):
        if mlflow.active_run() is not None:
            logger.info("Detected higher level run, starting nested run")
            run_ctx = mlflow.start_run(
                run_name="data engineering pipeline", nested=True
            )
        else:
            experiment_name = self.mlflow_config.get("experiment_name", "")
            logger.info(f"No higher level run detected, starting new run with name ")
            mlflow.set_experiment(experiment_name)
        with run_ctx as run:
            if not self.safe_to_run:
                logger.info("Failed to start pipeline")

            try:
                data = self.load_data_to_pandas()
            except FileNotFoundError as e:
                logger.warning(f"File not found error {e}")
                raise FileNotFoundError(f"Could not find {self.file_path}")

            if self.data_config["target_to_binary"]:
                logger.info("Transforming target to binary class")
                data = self.transform_target_to_binary(data)
            logger.info("Removing unneeded columns defined in config")
            data = self.remove_columns(data)
            logger.info("Removed columns from the dataframe defined in the config")
            save_path = Path(
                self.file_config["data_dir_path"],
                self.file_config["processed_data_path"],
                self.file_config["processed_data_file_save_name"],
            )
            data.to_csv(save_path)
            logger.info(f"Wrote data back out to {save_path}")
            ## Save as attribute to use externally
            self.data = data
