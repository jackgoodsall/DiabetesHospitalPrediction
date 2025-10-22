from src.components.training_splits import split_df
from model_builder import build_estimator
from data_ingestion import DataInputCleaningPipeLine
from components.data_engineering import DataEngineeringPipeLine

from typing import Dict, Any
from pydantic import BaseModel, ValidationError

import os
import yaml
import logging
import mlflow

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
        self.config = config

        logger.info("Initiating pipeline")
        logger.info("Running Data input pipeline")
        self._data_input_pipeline = DataInputCleaningPipeLine()
        logger.info("Data input pipeline initiation finished," \
            "data engineering pipeline initiation starting to run.")
        self._data_transformation_pipeline = DataEngineeringPipeLine()
        logger.info("Data engineering pipeline initation finished.")


    def run_pipeline(self):
        mlflow.set_experiment()
        logger.info("Running Pipeline")
        self._data_input_pipeline.run_pipeline()
        logger.info("Splitting data")
        train, test = split_df(self._data_input_pipeline.data)
        logger.info("Transforming data")
        self._data_transformation_pipeline.fit(self.train)

        self.train = self._data_transformation_pipeline.transform(train)
        self.test = self._data_transformation_pipeline.transform(test)

        logger.info("Creating model")


        self.models = []
        model_names = self.config.model["model_names"]

        with mlflow.start_run():
            if isinstance(model_names, str):
                model_names = [model_names]
            for model_name in model_names:
                estimator_configs = self.config.model.get(model_name, {})
                self.models.append(build_estimator(model_name,**estimator_configs))

        
        
    