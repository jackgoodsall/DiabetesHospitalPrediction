import logging
import mlflow
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Import all necessary components
from components.config import (
    load_yaml_config,
    ModelTrainingConfig,
    DataEngineeringConfig,
    DataInputCleaningPipeLineConfig
)
from data_ingestion import BinaryReadmissionInputCleaningPipeline
from components.data_engineering import DataEngineeringPipeLine
from components.training_splits import split_df
from components.model_evaluation import binary_classifcation_report
from model_builder import build_estimator
from model_trainer import ModelTrainer  

logger = logging.getLogger(__name__)

# Set up logging at the entry point
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

CONFIG_PATH = "./configs/run_config.yaml"


def run_ml_pipeline():
    """
    Main function to orchestrate the end-to-end ML pipeline.
    """
    try:
        # --- 0. Configuration and Global Setup ---
        config: ModelTrainingConfig = load_yaml_config(CONFIG_PATH, ModelTrainingConfig)

        data_cleaning_config: DataInputCleaningPipeLineConfig = load_yaml_config(CONFIG_PATH, DataInputCleaningPipeLineConfig)

        seed = np.random.randint(0, 2**16 - 1)
        np.random.seed(seed)
        logger.info(f"Generated random seed for this run as {seed}")

        experiment_name = config.mlflow_information["experiment_name"]
        mlflow.set_experiment(experiment_name)

        # Start a single root run for the entire pipeline
        with mlflow.start_run(
            run_name=f"Pipeline Run {datetime.now():%Y%m%d_%H%M%S}"
        ) as root_run:
            mlflow.log_param("global_seed", seed)
            # Log the entire config file
            mlflow.log_artifact("configs/run_config.yaml")

            # --- 1. Data Ingestion, Splitting, and Feature Engineering ---
            logger.info("Stage 1: Data Ingestion and Preparation")
            data_input_pipeline = BinaryReadmissionInputCleaningPipeline(data_cleaning_config)
            data = data_input_pipeline.run_pipeline()

            target = config.data["target_column_name"]

            # Split configuration. group_column (e.g. patient_nbr) makes the
            # split leak-free: all encounters for one patient stay on the same
            # side, so the model can't memorise patient-specific signal.
            split_config = config.model_extra.get("split", {})
            group_col = split_config.get("group_column")
            test_size = split_config.get("test_size", 0.2)
            stratify = split_config.get("stratify", True)
            cv_folds = split_config.get("cv_folds", 5)

            mlflow.log_params(
                {
                    "split_group_column": group_col,
                    "split_test_size": test_size,
                    "split_stratify": stratify,
                    "cv_folds": cv_folds,
                }
            )

            train_df, test_df = split_df(
                data,
                target=target,
                test_size=test_size,
                stratify=stratify,
                group_col=group_col,
                random_state=seed,
            )

            # Extract the grouping column for leak-free CV, then drop it so it
            # never leaks into the feature matrix.
            groups_train = None
            drop_for_features = [target]
            if group_col and group_col in train_df.columns:
                groups_train = train_df[group_col].to_numpy()
                drop_for_features.append(group_col)
                logger.info(
                    f"Group-aware split on '{group_col}': "
                    f"{train_df[group_col].nunique()} train / "
                    f"{test_df[group_col].nunique()} test groups, "
                    f"{len(train_df)} / {len(test_df)} rows"
                )

            X_train, y_train = (
                train_df.drop(columns=drop_for_features),
                train_df[target].to_numpy(),
            )
            X_test, y_test = (
                test_df.drop(columns=drop_for_features),
                test_df[target].to_numpy(),
            )

            # --- 2. Feature Transformation ---
            logger.info("Stage 2: Fitting and Transforming Data")
            # Ensure DataEngineeringConfig can be safely initialized
            de_config = DataEngineeringConfig(
                **config.model_extra.get("data_engineering", {})
            )
            data_transformer = DataEngineeringPipeLine(de_config)
            data_transformer.fit(X_train)

            # Transform data arrays
            X_train_transformed = data_transformer.transform(X_train).to_numpy()
            X_test_transformed = data_transformer.transform(X_test).to_numpy()

            # --- 3. Model Training Loop ---
            trainer = ModelTrainer(seed=seed, n_splits=cv_folds)
            model_names = (
                config.model["model_names"]
                if isinstance(config.model["model_names"], list)
                else [config.model["model_names"]]
            )

            for model_name in model_names:
                # Start a NESTED run for each model being trained
                with mlflow.start_run(run_name=model_name, nested=True) as model_run:
                    logger.info(f"Stage 3: Training {model_name}")

                    # A. Build Model and Log Params
                    estimator_configs = config.model_extra.get(model_name, {})
                    current_model = build_estimator(model_name, **estimator_configs)
                    mlflow.log_params(estimator_configs)

                    # B. Train Model (via the specialist ModelTrainer)
                    final_trained_model, oof_metrics, oof_predictions = (
                        trainer.train_and_get_results(
                            current_model, X_train_transformed, y_train, groups=groups_train
                        )
                    )

                    # C. Evaluation on Test Set and Logging Metrics
                    test_predictions = final_trained_model.predict_proba(
                        X_test_transformed
                    )[:, 1]
                    test_metrics = binary_classifcation_report(y_test, test_predictions)

                    mlflow.log_metrics(test_metrics)
                    mlflow.log_metrics({f"oof_{k}": v for k, v in oof_metrics.items()})

                    # D. Logging and Saving Artifacts (The Runner's responsibility!)
                    logger.info("Stage 4: Logging Artifacts")

                    # 1. Log the Transformer

                    transformer_path = Path("artefacts/pipeline")
                    transformer_path.mkdir(parents=True, exist_ok=True)
                    transformer_path = (
                        transformer_path / f"{model_name}_transformer.joblib"
                    )
                    joblib.dump(data_transformer, transformer_path)
                    mlflow.log_artifact(transformer_path, artifact_path="preprocessor")

                    # 2. Log the Model
                    # Prob a better way to log models depending on their
                    # inferance signiture but this will do for now.

                    if isinstance(final_trained_model, XGBClassifier):
                        mlflow.xgboost.log_model(
                            final_trained_model,
                            "model",
                            registered_model_name=model_name,
                        )
                    elif isinstance(final_trained_model, CatBoostClassifier):
                        mlflow.catboost.log_model(
                            final_trained_model,
                            "model",
                            registered_model_name=model_name,
                        )
                    elif isinstance(final_trained_model, LGBMClassifier):
                        mlflow.lightgbm.log_model(
                            final_trained_model,
                            "model",
                            registered_model_name=model_name,
                        )
                    else:
                        mlflow.sklearn.log_model(
                            final_trained_model,
                            "model",
                            registered_model_name=model_name,
                        )

                    # 3. Local Saving
                    if config.mlflow_information.get("save_model"):
                        save_dir = Path(config.mlflow_information["save_dir"])
                        save_dir.mkdir(exist_ok=True, parents = True)
                        save_name = (
                            save_dir
                            / f"{model_name}_{datetime.now():%Y%m%d_%H%M%S}.joblib"
                        )
                        joblib.dump(final_trained_model, save_name)
                        logger.info(f"Model saved locally to {save_name}")

    except Exception as e:
        logger.critical(
            f"The ML Pipeline encountered a critical failure.", exc_info=True
        )


if __name__ == "__main__":
    run_ml_pipeline()
