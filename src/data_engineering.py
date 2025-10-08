import os 
import sys
import yaml
from pydantic import BaseModel, ValidationError
from typing import Dict, Any



import mlflow
from mlflow.models.signature import infer_signature

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.

from typing import Tuple, List

class DataEngineeringConfig(BaseModel):
    ## Basic overhead schema for the data_engineering_config
    file_information: Dict[str, Any]
    mlflow_information: Dict[str, Any]
    data: Dict[str, Any]

def load_yaml_config() -> DataEngineeringConfig:
    ### Loads the different yaml configs
    config_path = "../configs/"
    data_engineering_config = "data_engineering_config.yaml"
    with open(os.path.join(config_path + data_engineering_config), "rb") as f:
        config = yaml.safe_load(f)
    try:
        return DataEngineeringConfig(**config)
    except ValidationError as e:
        raise ValueError(f"Configuration missing required sections:\n{e}")
    

class DataEngineeringPipeLine:
    def __init__(self):
        configs = load_yaml_config()
        self.file_config = configs.file_information
        self.mlflow_config = configs.mlflow_information
        self.data_config = configs.data


    def load_data_to_pandas(self) -> pd.DataFrame:
        return pd.read_csv(self.file_config(os.path.join(
            self.file_config["data_dir_path"] + self.file_config["raw_data_path"]
            + self.file_config["raw_data_file_name"]
        )))

    def transform_target_to_binary(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.data_config["target_to_binary"] == False: return pd.DataFrame
        data[""]
        return data
    

    def run_pipeline(self):
        mlflow.set_experiment(self.mlflow_config["experiment_name"])
        with mlflow.start_run(run_name = "data engineering pipeline") as run:


# Used to make readdmision variable binary
def transform_target_to_binary(data_set: pd.DataFrame,
                               target_variable: str, 
                               mapping: dict= None) -> pd.DataFrame:
    ### Map a Cateogrial variable to a binary one
    if mapping is None: return data_set
    data_set["target_variable"].map(mapping)
    return data_set

def remove_columns(data_set: pd.DataFrame,
                   columns_to_remove) -> pd.DataFrame:
    return data_set.drop(columns = columns_to_remove)



def generate_new_features(data_set: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    return data_set, []

def run_data_engineering() -> None:

        mlflow.set_tag("data_set_tag",
                   EXPERIMENT_DATA_TAG)
        file_path = os.path.join(RAW_DATA_DIR, RAW_DATA_FILE_NAME)
        if not os.path.exists(file_path):
            mlflow.log_param("file_exists", False)
            mlflow.log_param("file_path", file_path)
            mlflow.set_tag("status", "error")
            mlflow.log_text("File was not found on disk.", artifact_file="logs/error.txt")
            ## If not file found just clean exit
            sys.exit(0)
        else:
            mlflow.log_param("file_exists", True)
            mlflow.set_tag("status", "ok")
            mlflow.log_param("file_path", file_path)

        data_set = read_csv_to_dataframe(file_path)
        
        #2 ) Preprocessing
        # 2.1 ) Remove columns
        config_file_path = os.path.join(CONFIG_FILES, "preprocessing_config.yaml")
        with open(config_file_path, "r") as f:
            preprocessing_config = yaml.safe_load(f)
        mlflow.log_artifact(config_file_path, artifact_path="config")
        
        
        columns_to_drop = preprocessing_config["data"]["drop_columns"]
        mlflow.log_param("dropped_columns", columns_to_drop)
        data_set = remove_columns(data_set, columns_to_drop)

        # 2.2 ) Engineer new features
        data_set, new_columns_engineered = generate_new_features(data_set)

        mlflow.log_param("new_features_added", new_columns_engineered)

        # 2.3 ) Split into test and train

        # 2.4 ) Impute values



        # 2.5 ) Scale Data - train on trrain fit on both


        # 3)




    mlflow.end_run()

if __name__ == "__main__":

    run_data_engineering()


