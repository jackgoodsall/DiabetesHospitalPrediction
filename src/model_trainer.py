from components.training_splits import split_df, cross_validation_splits
from components.model_evaluation import binary_classifcation_report
from model_builder import build_estimator
from data_ingestion import DataInputCleaningPipeLine
from components.data_engineering import DataEngineeringPipeLine, DataEngineeringConfig

from typing import Dict, Any
from pydantic import BaseModel, ValidationError, ConfigDict

import os
import yaml
import logging
import mlflow

import numpy as np

from datetime import datetime

import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = "configs/run_config.yaml"


class ModelTrainingConfig(BaseModel):
    ## Basic overhead schema for the data_engineering_config
    model_config = ConfigDict(extra="allow")

    model: Dict[str, Any]
    mlflow_information: Dict[str, Any]
    data: Dict[str, Any]


def load_yaml_config() -> ModelTrainingConfig:
    ### Loads the different yaml configs
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    try:
        return ModelTrainingConfig(**config)
    except ValidationError as e:
        raise ValueError(f"Configuration missing required sections:\n{e}")


class ModelTrainer:
    """
    Class for training a model, runs the full pipeline from data ingestion, data cleaning, 
    feature engineering, scaling and splitting. 

    Initalisation takes in config defined by the ModelTrainingConfig class defined above.

    Has one helper function for training the model using a CV split.

    Has one function to run the whole pipeline.
    """
    def __init__(self, config):
        self.config: ModelTrainingConfig = config

    def _train_model(self, model, X, y, splitter):
        """
        Internal helper function to train a model on data X, y using the
        splitter. Trains model on hold out splits with predictions done on the
        OOF predictions, retrains and returns model train on whole dataset.

        Has implemented sklearn and XGBoost classfier API calls. Anything else
        needs manual implemention/adding
        """

        oof_predictions = np.zeros(len(y))

        for train_idx, val_idx in splitter:
            (X_train, y_train, X_val, y_val) = (
                X[train_idx],
                y[train_idx],
                X[val_idx],
                y[val_idx],
            )
            if hasattr(model, "fit"):
                model.fit(X_train, y_train)
                oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]
        oof_metrics = binary_classifcation_report(y, oof_predictions)
        ## Refit model on whole training data
        model.fit(X, y)
        return model, oof_metrics

    def run_pipeline(self):
        '''
        Run the model training pipeline 
        '''
        # Set mlflow experiment
        experiment_name = self.config.mlflow_information["experiment_name"]
        experiment_tag = self.config.mlflow_information["experiment_tag"]
        mlflow.set_experiment(
            experiment_name=f"{experiment_name}{datetime.now():%Y%m%d_%H%M%S}"
        )

        ## Running data pipelines
        logger.info("Initiating pipeline")
        logger.info("Running Data input pipeline")
        self._data_input_pipeline = DataInputCleaningPipeLine()
        logger.info(
            "Data input pipeline initiation finished,"
            "data engineering pipeline initiation starting to run."
        )
        self._data_transformation_pipeline = DataEngineeringPipeLine(
            DataEngineeringConfig(**self.config.model_extra["data_engineering"])
        )
        logger.info("Data engineering pipeline initation finished.")


        ## Seeding and logging seed for reproducibility
        seed = np.random.randint(0, 2**32 - 1)
        logger.info(f"Generated random seed for this run as {seed}")
        np.random.seed(seed)

        ## Run data input pipeline
        logger.info("Running Pipeline")
        self._data_input_pipeline.run_pipeline()
        logger.info("Splitting data")
        target = self.config.data["target_column_name"]
        # Split the dataframe into a train and test split
        train, test = split_df(
            self._data_input_pipeline.data, test_size=0.2, target=target
        )
        logger.info("Transforming data")
        # Prepare data into numpy arrays
        X_train, y_train = train.drop(columns=target), train[target].to_numpy()
        X_test, y_test = test.drop(columns=target), test[target].to_numpy()
        ## Fit the data transformation pipelines on training data
        self._data_transformation_pipeline.fit(X_train)

        # Transform train and test data
        X_train = self._data_transformation_pipeline.transform(X_train).to_numpy()
        X_test = self._data_transformation_pipeline.transform(X_test).to_numpy()

        ### Get a list of models to train from the config, if only one convert to list
        logger.info("Getting models from config")
        self.models = []
        model_names = self.config.model["model_names"]
        if isinstance(model_names, str):
            model_names = [model_names]

        ## Start mlflow run for training models
        logger.info("Starting mlflow run and training models")
        nested = True if mlflow.active_run() else False
        with mlflow.start_run(run_name="Model Training", nested=nested):
            for model_name in model_names:
                # Get configs
                logger.info(f"Training model {model_name}")
                estimator_configs = self.config.model_extra.get(model_name, {})
                estimator_configs = (
                    estimator_configs if estimator_configs is not None else {}
                )
                # Build model
                current_model = build_estimator(model_name, **estimator_configs)
                mlflow.log_params(estimator_configs)
                # train model
                current_model, oof_predictions = self._train_model(
                    current_model,
                    X_train,
                    y_train,
                    cross_validation_splits(
                        X_train, return_indices=True, random_state=seed
                    ),
                )
                # Log the model using mlflow
                mlflow.xgboost.log_model(current_model , "xgboostmodel")
                logger.info(f"Training {model_name} was succesful!")
                # Make report on test data
                test_predictions = current_model.predict_proba(X_test)[:, 1]
                test_metrics = binary_classifcation_report(y_test, test_predictions)
                # Log and save model.
                mlflow.log_metrics(test_metrics)
                mlflow.log_metrics(oof_predictions)
                if self.config.mlflow_information["save_model"]:
                    save_dir = self.config.mlflow_information["save_dir"]
                    save_path = Path(save_dir)
                    save_path.mkdir(exist_ok = True)
                    save_name = save_path / f"{model_name}_{datetime.now():%Y%m%d_%H%M%S}.joblib"
                    joblib.dump(current_model, save_name)

if __name__ == "__main__":
    """ logging.basicConfig(
    level=logging.DEBUG,  # show everything DEBUG and up
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ) """

    config_object = load_yaml_config()
    with mlflow.start_run(run_name="a"):
        model_trainer = ModelTrainer(config_object)
        model_trainer.run_pipeline()
