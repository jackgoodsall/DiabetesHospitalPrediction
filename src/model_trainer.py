from components.data_splitter import split_df
from model_builder import build_estimator
from data_ingestion import DataInputCleaningPipeLine
from components.data_engineering import DataEngineeringPipeLine

from typing import Dict, Any
from pydantic import BaseModel, ValidationError

import os
import yaml
import logging

logger  = logging.getLogger(__name__)

class ModelTrainingConfig(BaseModel):
    ## Basic overhead schema for the data_engineering_config
    model: Dict[str, Any]



def load_yaml_config() -> ModelTrainingConfig:
    ### Loads the different yaml configs
    config_path = "../configs/"
    data_engineering_config = "model_training_config.yaml"
    with open(os.path.join(config_path + data_engineering_config), "r") as f:
        config = yaml.safe_load(f)
    try:
        return ModelTrainingConfig(**config)
    except ValidationError as e:
        raise ValueError(f"Configuration missing required sections:\n{e}")

class ModelTrainer:
    def __init__(self, config):
        logger.info("Initiating pipeline")
        logger.info("Running Data input pipeline")
        self._data_input_pipeline = DataInputCleaningPipeLine()
        logger.info("Data input pipeline initiation finished," \
            "data engineering pipeline initiation starting to run.")
        self._data_transformation_pipeline = DataEngineeringPipeLine()
        logger.info("Data engineering pipeline initation finished.")

    def run_pipeline(self):

        self._data_input_pipeline.run_pipeline()
        
        train, test = split_df(self._data_input_pipeline.data)
        self._data_transformation_pipeline.fit(self.train)

        self.train = self._data_transformation_pipeline.transform(train)
        self.test = self._data_transformation_pipeline.transform(test)

    