import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models.signature import infer_signature
from pydantic import BaseModel, ValidationError
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class DataInputCleaningPipeLineConfig(BaseModel):
    ## Basic overhead schema for the data_engineering_config
    file_information: Dict[str, Any]
    mlflow_information: Dict[str, Any]
    data: Dict[str, Any]

def load_yaml_config() -> DataInputCleaningPipeLineConfig:
    ### Loads the different yaml configs
    config_path = "../configs/"
    data_engineering_config = "data_engineering_config.yaml"
    with open(os.path.join(config_path + data_engineering_config), "r") as f:
        config = yaml.safe_load(f)
    try:
        return DataInputCleaningPipeLineConfig(**config)
    except ValidationError as e:
        raise ValueError(f"Configuration missing required sections:\n{e}")
    

class DataInputCleaningPipeLine:
    def __init__(self):
        logging.log("Attempting to load config")
        try:
            configs = load_yaml_config()
            self.file_config = configs.file_information
            self.mlflow_config = configs.mlflow_information
            self.data_config = configs.data
            self.safe_to_run = True
            logger.log("Configuration loading succesful")
        except ValidationError as e:
            logger.log(f"Config loading failed error message {e}")
            self.safe_to_run = False

    def load_data_to_pandas(self) -> pd.DataFrame:
        ### Loads data from file into a pandas dataframe
        file_path = os.path.join(
            self.file_config["data_dir_path"],
            self.file_config["raw_data_path"],
            self.file_config["raw_data_file_name"]
        )
        if os.path.exists(file_path):
            logging.info(msg = "Raw file path exist and has been loaded")
            return pd.read_csv(file_path)
        else:
            logging.info(msg = "Raw file path did not exist")
            raise FileNotFoundError("File does not exist")
        
    def transform_target_to_binary(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.data_config["target_to_binary"] == False: return data 
        return data
    
    def remove_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        columns_to_drop = self.data_config["drop_columns"]
        mlflow.log_param("dropped columns", columns_to_drop)
        return data.drop(columns = columns_to_drop)

    def run_pipeline(self):
        mlflow.set_experiment(self.mlflow_config["experiment_name"])
        with mlflow.start_run(run_name = "data engineering pipeline") as run:
            if  not self.safe_to_run:
                logger.info("Failed to start pipeline")
                mlflow.log_text("Failed to start pipeline")
            try:
                data = self.load_data_to_pandas()
            except FileNotFoundError as e:
                mlflow.log_text("File was not found experiment ending")
                return
            if self.data_config["target_to_binary"]:
                logger.info("Transforming target to binary class")
                data = self.transform_target_to_binary(data)
            
            
        
                



        




