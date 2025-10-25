from src.components.training_splits import split_df, cross_validation_splits
from model_builder import build_estimator
from data_ingestion import DataInputCleaningPipeLine
from components.data_engineering import DataEngineeringPipeLine

from typing import Dict, Any
from pydantic import BaseModel, ValidationError, ConfigDict

import os
import yaml
import logging
import mlflow

import numpy as np

logger  = logging.getLogger(__name__)

class ModelTrainingConfig(BaseModel):
    ## Basic overhead schema for the data_engineering_config
    model_config = ConfigDict(extra = "allow")

    model: Dict[str, Any]
    mlflow_information: Dict[str, Any]
    data_engineering: Dict[str, Any]

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
        self.config : ModelTrainingConfig = config

        logger.info("Initiating pipeline")
        logger.info("Running Data input pipeline")
        self._data_input_pipeline = DataInputCleaningPipeLine()
        logger.info("Data input pipeline initiation finished," \
            "data engineering pipeline initiation starting to run.")
        self._data_transformation_pipeline = DataEngineeringPipeLine(
            self.config.data_engineering
        )
        logger.info("Data engineering pipeline initation finished.")

    def _train_model(self,
                     model,
                     X,
                     y,
                     splitter):
        """
        Internal helper function to train a model on data X, y using the 
        splitter. Trains model on hold out splits with predictions done on the 
        OOF predictions, retrains and returns model train on whole dataset.

        Has implemented sklearn and XGBoost classfier API calls. Anything else
        needs manual implemention/adding
        """

        oof_predictions = np.zeros_like(y)

        for train_idx, val_indx in splitter:
            
            
            if hasattr(model, "fit"):
                model.fit(X, y)




    def run_pipeline(self):
        experiment_name = self.config.mlflow_information["experiment_name"]
        experiment_tag = self.config.mlflow_information["experimment_Tag"]
        mlflow.set_experiment()

        seed = np.random.randint(0, 2**32 - 1)
        logger.info(f"Generated random seed for this run as {seed}")
        np.random.seed(seed)
        mlflow.log()
        logger.info("Running Pipeline")
        self._data_input_pipeline.run_pipeline()
        logger.info("Splitting data")
        train, test = split_df(self._data_input_pipeline.data)
        logger.info("Transforming data")
        self._data_transformation_pipeline.fit(self.train)

        self.train = self._data_transformation_pipeline.transform(train)
        self.test = self._data_transformation_pipeline.transform(test)

        
        
        ### Get a list of models to train from the config, if only one convert to list
        logger.info("Getting models from config")
        self.models = []
        model_names = self.config.model["model_names"]
        if isinstance(model_names, str):
            model_names = [model_names]

        ### Split data into X and y for the model training step - makes code 
        ### cleaner/easier to read
        logger.info("Splitting into X and y arrays from dataframe")
        target = self.config.model_extra["data"]["target_column_name"]
        X = self.train.drop(columns = target)
        y = self.train[target]

        logger.info("Starting mlflow run and training models")
        with mlflow.start_run():
            for model_name in model_names:
                logger.log(f"Training model {model_name}")
                estimator_configs = self.config.model_extra.get(model_name, {})
                current_model = build_estimator(model_name, **estimator_configs)
                self._train_model(
                    current_model,
                    X,
                    y,
                    cross_validation_splits(self.train,
                                            random_state = seed)

                )
                
                



        
        
    