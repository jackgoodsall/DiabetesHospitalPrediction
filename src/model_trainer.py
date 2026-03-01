from components.training_splits import split_df, cross_validation_splits
from components.model_evaluation import binary_classifcation_report
from model_builder import build_estimator
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


class ModelTrainer:
    """
    Specialized class for training a single model using Cross-Validation (CV)
    and calculating OOF (Out-Of-Fold) predictions and metrics.
    """

    def __init__(self, seed: int):
        """Initializes the trainer with a fixed seed for reproducibility."""
        self.seed = seed
        logger.debug(f"ModelTrainer initialized with seed: {self.seed}")

    def _train_model_cv(self, model, X: np.ndarray, y: np.ndarray):
        """
        Internal helper function to train a model on data X, y using CV splits.
        Calculates OOF metrics and returns the model refit on the whole dataset.
        """

        # 1. Prepare Splitter
        splitter = cross_validation_splits(
            X, return_indices=True, random_state=self.seed
        )
        oof_predictions = np.zeros(len(y))

        # 2. Run Cross-Validation Loop
        for fold, (train_idx, val_idx) in enumerate(splitter):
            logger.debug(f"Training Fold {fold + 1}")

            # Prepare data for this fold
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Check for standard fit method (for sklearn/XGBoost compatibility)
            if hasattr(model, "fit"):
                model.fit(X_train, y_train)
                # Predict probability (assuming binary classification)
                oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]

        # 3. Calculate OOF Metrics
        oof_metrics = binary_classifcation_report(y, oof_predictions)
        logger.info(f"OOF Metrics calculated: {oof_metrics}")

        # 4. Refit model on whole training data (for final production model)
        logger.info("Refitting final model on full training set.")
        model.fit(X, y)

        return model, oof_metrics, oof_predictions  

    def train_and_get_results(self, model, X_train: np.ndarray, y_train: np.ndarray):
        """
        Public method to run the training protocol.
        """
        return self._train_model_cv(model, X_train, y_train)
