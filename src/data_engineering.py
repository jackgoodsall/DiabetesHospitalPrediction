import os 
import sys
import yaml

import mlflow
from mlflow.models.signature import infer_signature

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from typing import Tuple, List

RAW_DATA_DIR , RAW_DATA_FILE_NAME = "data/raw_data", "diabetic_data.csv"
PRORCESSD_DATA_DIR = "data/processed_data"

CONFIG_FILES = "configs"

EXPERIMENT_NAME = "Experiment-1"
EXPERIMENT_RAW_DATA_FILE_NAME = "experiment_1_processed.csv"
EXPERIMENT_DATA_TAG = ""

RANDOM_SEED =  124314124

def seed_everything(seed: int = RANDOM_SEED) -> None:
    ## Set seed for every library used from random seed
    np.random.seed(seed) 

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

def read_csv_to_dataframe(file_name: str) -> pd.DataFrame:
    return pd.read_csv(file_name)

def generate_new_features(data_set: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    return data_set, []

def run_data_engineering() -> None:
    seed_everything()

    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    #1) Ingestion
    with mlflow.start_run(run_name = "data ingestion") as run:
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


