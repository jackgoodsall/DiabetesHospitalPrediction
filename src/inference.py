"""
inference.py
------------
Run predictions on new data using a trained model and its associated
preprocessor transformer.

Two loading strategies are supported:

  1. MLflow Model Registry (default)
     Loads the model registered under `model_name` at the requested stage
     (default: "Production") and fetches the matching transformer from the
     same run's artifacts.

  2. Local .joblib files
     Pass --model-path and --transformer-path to bypass MLflow entirely.

Usage examples
--------------
# From the registry (Production stage):
    python src/inference.py --input data/raw_data/diabetic_data.csv \
                            --model-name xgboost

# Specific stage:
    python src/inference.py --input data/raw_data/diabetic_data.csv \
                            --model-name xgboost --stage Staging

# From local files:
    python src/inference.py --input data/raw_data/diabetic_data.csv \
                            --model-path artefacts/models/xgboost_20260303.joblib \
                            --transformer-path artefacts/pipeline/xgboost_transformer.joblib

# Save predictions to a CSV:
    python src/inference.py --input new_patients.csv \
                            --model-name xgboost --output predictions.csv
"""

import logging
import argparse
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import numpy as np
from mlflow import MlflowClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

CONFIG_PATH = "./configs/run_config.yaml"
DEFAULT_THRESHOLD = 0.5


def _load_from_registry(model_name: str, stage: str):
    """
    Load a model and its preprocessor transformer from the MLflow Model Registry.

    The transformer is expected to be logged as an artifact under
    ``preprocessor/{model_name}_transformer.joblib`` in the same run that
    registered the model version.

    Returns
    -------
    model : mlflow.pyfunc.PyFuncModel
    transformer : DataEngineeringPipeLine
    run_id : str
    """
    client = MlflowClient()

    # Resolve the latest version at the requested stage
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise ValueError(
            f"No model version found for '{model_name}' at stage '{stage}'. "
            f"Train a model and promote it, or use --model-path / --transformer-path."
        )

    version = versions[0]
    run_id = version.run_id
    logger.info(
        f"Loading '{model_name}' v{version.version} (stage={stage}, run_id={run_id})"
    )

    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Download transformer artifact from the same run
    transformer_artifact = f"preprocessor/{model_name}_transformer.joblib"
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=transformer_artifact
    )
    transformer = joblib.load(local_path)
    logger.info(f"Loaded transformer from run artifact: {transformer_artifact}")

    return model, transformer, run_id


def _load_from_local(model_path: str, transformer_path: str):
    """
    Load a model and transformer from local .joblib paths.

    Returns
    -------
    model : fitted sklearn/XGBoost estimator
    transformer : DataEngineeringPipeLine
    run_id : None
    """
    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path)
    logger.info(f"Loaded model from: {model_path}")
    logger.info(f"Loaded transformer from: {transformer_path}")
    return model, transformer, None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    input_path: str,
    model_name: str | None = None,
    stage: str = "Production",
    model_path: str | None = None,
    transformer_path: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Load data, preprocess, and generate predictions.

    Parameters
    ----------
    input_path : str
        Path to a CSV file of raw patient records.
    model_name : str, optional
        Registered model name in MLflow (used when not using local paths).
    stage : str
        Model Registry stage to load from. Defaults to 'Production'.
    model_path : str, optional
        Path to a local .joblib model file.
    transformer_path : str, optional
        Path to a local .joblib transformer file.
    threshold : float
        Probability threshold for converting scores to binary labels.
    output_path : str, optional
        If provided, predictions CSV is written here.

    Returns
    -------
    pd.DataFrame with columns: [original features, readmission_probability, readmitted_predicted]
    """
    # --- Load raw data ---
    logger.info(f"Reading input data from: {input_path}")
    data = pd.read_csv(input_path)
    logger.info(f"Loaded {len(data)} rows, {data.shape[1]} columns")

    # --- Load model + transformer ---
    using_local = model_path is not None and transformer_path is not None
    if using_local:
        model, transformer, run_id = _load_from_local(model_path, transformer_path)
    elif model_name:
        model, transformer, run_id = _load_from_registry(model_name, stage)
    else:
        raise ValueError(
            "Provide either --model-name (registry) or both "
            "--model-path and --transformer-path (local)."
        )

    # --- Preprocess ---
    logger.info("Applying preprocessing transformer")
    X_transformed = transformer.transform(data).to_numpy()

    # --- Predict ---
    logger.info("Generating predictions")
    if using_local:
        # Raw sklearn/XGBoost estimator — use predict_proba directly
        probabilities = model.predict_proba(X_transformed)[:, 1]
    else:
        # mlflow.pyfunc model — returns a DataFrame
        prob_df = model.predict(pd.DataFrame(X_transformed))
        probabilities = np.array(prob_df).ravel()

    labels = (probabilities >= threshold).astype(int)

    # --- Build results DataFrame ---
    results = data.copy()
    results["readmission_probability"] = probabilities
    results["readmitted_predicted"] = labels

    pos_rate = labels.mean() * 100
    logger.info(
        f"Predictions complete — {labels.sum()} / {len(labels)} "
        f"predicted as readmitted ({pos_rate:.1f}%)"
    )

    # --- Optionally save ---
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out, index=False)
        logger.info(f"Predictions written to: {out}")

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained readmission model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the input CSV file of raw patient data."
    )

    source = parser.add_argument_group("Model source (choose one strategy)")
    source.add_argument(
        "--model-name",
        help="Registered model name in the MLflow Model Registry."
    )
    source.add_argument(
        "--stage", default="Production",
        choices=["Staging", "Production", "Archived"],
        help="Registry stage to load from (default: Production)."
    )
    source.add_argument(
        "--model-path",
        help="Path to a local .joblib model file (bypasses MLflow registry)."
    )
    source.add_argument(
        "--transformer-path",
        help="Path to a local .joblib transformer file (required with --model-path)."
    )

    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Probability threshold for binary label assignment (default: {DEFAULT_THRESHOLD})."
    )
    parser.add_argument(
        "--output",
        help="Optional path to write predictions CSV."
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    results = run_inference(
        input_path=args.input,
        model_name=args.model_name,
        stage=args.stage,
        model_path=args.model_path,
        transformer_path=args.transformer_path,
        threshold=args.threshold,
        output_path=args.output,
    )

    print(results[["readmission_probability", "readmitted_predicted"]].head(10).to_string())
