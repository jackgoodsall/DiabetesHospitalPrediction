# Diabetes Hospital Readmission Prediction

[![CI](https://github.com/jackgoodsall/DiabetesHospitalPrediction/actions/workflows/ci.yml/badge.svg)](https://github.com/jackgoodsall/DiabetesHospitalPrediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end MLOps project predicting 30-day hospital readmission for diabetic patients, using 100,000+ patient records from the [UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

The focus is production-grade ML engineering rather than modelling. The pipeline is decoupled, config-driven, and reproducible. Changing the dataset or model only requires config changes.

---

## Features

- **Decoupled pipeline** - data ingestion, feature engineering, training, evaluation, and explainability are separate components run by a single pipeline runner
- **Config-driven** - all pipeline behaviour (features, models, hyperparameters, tuning, evaluation strategy, MLflow experiment name) is set in `configs/run_config.yaml` with no hardcoded values
- **Pydantic config validation** - configs are parsed into typed models at startup; bad config raises an error before any data is loaded
- **Pandera data validation** - schema checks on the raw CSV and cleaned DataFrame catch loading errors, encoding issues, and column-drop failures before training starts
- **Optuna hyperparameter tuning** - optional Bayesian search over per-model search spaces using the same group-aware CV as training; each trial is logged as a nested MLflow run via `MLflowCallback`; toggled via `tuning:` in config
- **MLflow experiment tracking** - every run logs parameters, metrics, the config file, the fitted transformer, and the trained model as artifacts; nested runs per model within a parent pipeline run
- **Evaluation suite** - test-set metrics include AUC-ROC, PR-AUC, F1, Precision, Recall, Accuracy, and Brier score; ROC, Precision-Recall, and calibration curve plots are saved to MLflow; a dummy baseline is always logged for comparison
- **Threshold selection** - decision threshold is chosen on OOF predictions (not the test set) using one of three strategies: `f1`, `recall@X`, or `default` (0.5)
- **SHAP explainability** - beeswarm, bar, waterfall, and dependence plots computed on the held-out test set and logged to MLflow; explainer auto-selected per model family (`TreeExplainer` / `LinearExplainer`)
- **Group-aware splitting** - train/test and cross-validation splits use `patient_nbr` via `StratifiedGroupKFold` so the same patient never appears in both sets
- **MLflow Model Registry** - models are registered and promoted to stages (`Staging` / `Production`)
- **CLI inference** - score new records from the command line, loading from the registry or local `.joblib` files
- **REST API** - FastAPI service with `/health`, `/predict`, and `/predict/batch` endpoints; model loaded once at startup, Pydantic-validated schemas, Swagger docs at `/docs`
- **Docker** - single image handles both training and serving via a `MODE` build argument
- **CI** - tests run on every push and pull request via GitHub Actions and `uv`

---

## Tech Stack

| Area | Tools |
|------|-------|
| ML / Data | scikit-learn, XGBoost, LightGBM, CatBoost, pandas, NumPy |
| Hyperparameter tuning | Optuna, optuna-integration[mlflow] |
| Explainability | SHAP |
| Data validation | Pandera |
| Experiment tracking | MLflow (tracking + model registry) |
| Model serving | FastAPI, Uvicorn |
| Config & validation | Pydantic v2, PyYAML |
| Testing | pytest |
| Packaging | uv, pyproject.toml |
| Containerisation | Docker |
| CI | GitHub Actions |

---

## Project Structure

```
DiabetesHospitalPrediction/
├── .github/workflows/
│   └── ci.yml                    # GitHub Actions CI pipeline
├── configs/
│   └── run_config.yaml           # All pipeline config
├── data/
│   ├── raw_data/                 # Raw CSVs (gitignored)
│   └── processed_data/           # Cleaned data output
├── notebooks/
│   └── data_visualisation.ipynb  # Exploratory data analysis
├── src/
│   ├── components/
│   │   ├── config.py             # Pydantic config models + YAML loader
│   │   ├── data_engineering.py   # sklearn ColumnTransformer pipeline
│   │   ├── data_validation.py    # Pandera schemas (RawDataSchema, CleanedDataSchema)
│   │   ├── explainability.py     # SHAP plot generation and MLflow logging
│   │   ├── model_evaluation.py   # Metrics, threshold selection, dummy baseline, eval plots
│   │   └── training_splits.py    # Stratified / group-aware train/test and CV splits
│   ├── data_ingestion.py         # Raw data loading, cleaning, and Pandera validation
│   ├── model_builder.py          # Model factory (XGBoost, LightGBM, CatBoost, RF, LR, etc.)
│   ├── model_trainer.py          # Cross-validated training with OOF metrics
│   ├── hyperparameter_tuner.py   # Optuna search spaces and tuning
│   ├── pipeline_runner.py        # Runs the full pipeline end-to-end
│   ├── inference.py              # CLI inference against registry or local files
│   ├── api.py                    # FastAPI prediction service (health, predict, batch)
│   └── schemas.py                # Pydantic request/response models for the API
├── artefacts/
│   ├── models/                   # Saved .joblib model files
│   ├── pipeline/                 # Saved transformer artifacts
│   └── evaluation_plots/         # ROC, PR curve, calibration curve PNGs
├── tests/                        # pytest test suite
├── Dockerfile
└── pyproject.toml
```

---

## Pipeline

```
configs/run_config.yaml
        |
        v
+---------------------+
|   Data Ingestion    |  Load CSV -> Pandera validation -> drop columns -> binary target
+----------+----------+
           |
           v
+---------------------+
|  Group-Aware Split  |  patient_nbr keeps all encounters for one patient on the same side
+----------+----------+
           |
           v
+---------------------+
|  Data Engineering   |  Numerical: impute + scale | Categorical: impute + OHE
+----------+----------+  (fitted on train only)
           |
           v
+---------------------+          +------------------------+
|  [Optional] Optuna  +--------->+  Per-trial CV scoring  |
|  Hyperparameter     |          |  (TPE sampler, nested  |
|  Tuning             |          |   MLflow runs)         |
+----------+----------+          +------------------------+
           | best params
           v
+---------------------+
|   Model Training    |  StratifiedGroupKFold CV -> OOF metrics -> refit on full train
+----------+----------+  (nested MLflow run per model)
           |
           v
+---------------------+
|    Evaluation       |  Threshold on OOF -> test-set metrics + plots
+----------+----------+  (AUC-ROC, PR-AUC, F1, Brier; ROC/PR/calibration curves)
           |
           v
+---------------------+
|  SHAP               |  Beeswarm, bar, waterfall, dependence plots -> MLflow
+----------+----------+
           |
           v
+---------------------+
|  MLflow Artifacts   |  Config, transformer (.joblib), model -> MLflow registry
+---------------------+
```

---

## Supported Models

All models are configured in `configs/run_config.yaml` and support Optuna tuning. Add any combination to `model.model_names`:

| Key | Model |
|-----|-------|
| `xgboost` | XGBoost |
| `lightgbm` | LightGBM |
| `catboost` | CatBoost |
| `random_forest` | sklearn RandomForestClassifier |
| `extra_trees` | sklearn ExtraTreesClassifier |
| `gradient_boosting` | sklearn GradientBoostingClassifier |
| `logistic_regression` | sklearn LogisticRegression |

---

## Evaluation

All threshold-based metrics use a threshold chosen from OOF predictions, not the test set.

| Metric | Description |
|--------|-------------|
| `auc_roc` | Area under the ROC curve |
| `pr_auc` | Area under the Precision-Recall curve |
| `f1` | F1 score at the chosen threshold |
| `precision` | Precision at the chosen threshold |
| `recall` | Recall at the chosen threshold |
| `accuracy` | Accuracy at the chosen threshold |
| `brier_score` | Calibration quality (lower is better) |

OOF versions are prefixed `oof_`. Dummy baseline versions are prefixed `dummy_`.

Plots (ROC curve, PR curve, calibration curve) are saved to `artefacts/evaluation_plots/` and logged to MLflow.

### Threshold strategies

Set via `evaluation.threshold_strategy` in `run_config.yaml`:

| Strategy | Behaviour |
|----------|-----------|
| `f1` | Threshold that maximises F1 on OOF predictions (default) |
| `recall@0.X` | Smallest threshold that achieves recall >= X (e.g. `recall@0.8`) |
| `default` | Fixed 0.5 |

---

## Hyperparameter Tuning

Disabled by default. Enable in config:

```yaml
tuning:
  enabled: true
  n_trials: 50          # Optuna trials per model
  metric: "pr_auc"      # pr_auc | auc_roc | f1 | brier_score
  direction: "maximize" # maximize or minimize
  cv_folds: 3           # CV folds inside the tuning objective
  timeout: null         # optional time limit in seconds
```

When enabled, a TPE-sampler Optuna study runs before the final model refit. Each trial is logged as a nested MLflow run. The best hyperparameters are used for the final refit and evaluation.

Search spaces for all seven model types are in `src/hyperparameter_tuner.py`.

---

## Model Serving API

The model and preprocessing transformer are loaded from the MLflow registry once at startup and reused for all requests.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns model name, stage, and load status |
| `POST` | `/predict` | Single-record prediction; returns probability and binary label |
| `POST` | `/predict/batch` | Batch prediction; returns per-record results and aggregate stats |

Swagger UI is at **`http://localhost:8000/docs`**.

### Run the API locally

```bash
# Requires a trained model promoted to Production in the MLflow registry
uv run uvicorn api:app --app-dir src --reload --port 8000
```

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "time_in_hospital": 5,
    "num_lab_procedures": 42,
    "num_medications": 12,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_diagnoses": 7,
    "number_inpatient": 0,
    "race": "Caucasian",
    "gender": "Female",
    "age": "[50-60)",
    "weight": null,
    "change": "Ch",
    "diabetesMed": "Yes",
    "metformin": "Steady",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "glipizide": "No",
    "glyburide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "insulin": "Up",
    "glipizide-metformin": "No",
    "glyburide-metformin": "No",
    "medical_specialty": "InternalMedicine",
    "diag_1": "250.01",
    "diag_2": "401",
    "diag_3": "272",
    "A1Cresult": ">8"
  }'
```

```json
{
  "readmission_probability": 0.623,
  "readmitted": true,
  "threshold_used": 0.5,
  "model_name": "xgboost",
  "model_stage": "Production"
}
```

The `threshold` query parameter adjusts the decision boundary:

```bash
POST /predict?threshold=0.7   # higher precision
POST /predict?threshold=0.3   # higher recall
```

### Load from local files instead of MLflow

**Bash / macOS / Linux:**
```bash
MODEL_PATH=artefacts/models/xgboost_20260303_225723.joblib \
TRANSFORMER_PATH=artefacts/pipeline/xgboost_transformer.joblib \
uv run uvicorn api:app --app-dir src --port 8000
```

**PowerShell (Windows):**
```powershell
$env:MODEL_PATH="artefacts/models/xgboost_20260303_225723.joblib"; $env:TRANSFORMER_PATH="artefacts/pipeline/xgboost_transformer.joblib"; uv run uvicorn api:app --app-dir src --port 8000
```

---

## Results

Baseline XGBoost (no tuning, no class imbalance handling):

| Metric | OOF (CV) | Test set |
|--------|----------|----------|
| AUC-ROC | 0.627 | 0.629 |
| Accuracy | 0.634 | 0.635 |
| F1 | 0.571 | 0.576 |
| Precision | 0.617 | 0.626 |
| Recall | 0.531 | 0.533 |

OOF and test metrics are close, with no sign of overfitting. Dummy baseline metrics are logged alongside every run.

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) or pip
- Raw dataset CSV at `data/raw_data/diabetic_data.csv`

### Install

```bash
git clone https://github.com/jackgoodsall/DiabetesHospitalPrediction.git
cd DiabetesHospitalPrediction
uv sync
```

### Run the pipeline

```bash
uv run python src/pipeline_runner.py
```

Runs: data ingestion, Pandera validation, feature engineering, optional Optuna tuning, model training, evaluation, SHAP, and MLflow logging.

### Run inference

```bash
# From MLflow registry (Production stage)
uv run python src/inference.py \
    --input data/raw_data/diabetic_data.csv \
    --model-name xgboost

# From local files
uv run python src/inference.py \
    --input data/raw_data/diabetic_data.csv \
    --model-path artefacts/models/xgboost_<timestamp>.joblib \
    --transformer-path artefacts/pipeline/xgboost_transformer.joblib \
    --output predictions.csv
```

### View MLflow UI

```bash
uv run mlflow ui
```

Go to `http://localhost:5000` to browse runs, metrics, SHAP plots, and evaluation curves.

### Docker

```bash
# Train (default)
docker build -t diabetes-prediction .
docker run diabetes-prediction

# Serve via docker compose (mounts local artefacts, no MLflow registry needed)
docker compose up --build
# http://localhost:8000/docs

# Serve manually (requires a model in Production in MLflow)
docker build --build-arg MODE=serve -t diabetes-api .
docker run -p 8000:8000 diabetes-api
```

### Tests

```bash
uv run pytest tests/ -v
```

---

## Configuration

Everything is in `configs/run_config.yaml`:

```yaml
# Models to train
model:
  model_names: [xgboost, lightgbm]

# Train/test split
split:
  group_column: patient_nbr   # prevents patient-level leakage
  test_size: 0.2
  stratify: true
  cv_folds: 5

# Feature preprocessing
data_engineering:
  numerical_imputer_strat: mean
  scaler: standard            # or minmax

# Threshold and baseline
evaluation:
  threshold_strategy: "f1"   # f1 | recall@0.X | default
  dummy_strategy: "stratified"

# SHAP
explainability:
  enabled: true
  sample_size: 2000
  max_display: 20

# Optuna tuning (off by default)
tuning:
  enabled: false
  n_trials: 50
  metric: "pr_auc"
  direction: "maximize"
  cv_folds: 3

# Per-model hyperparameters (used when tuning is off)
xgboost:
  max_depth: 6
  eta: 0.1
  n_estimators: 1000
```

Configs are validated against Pydantic schemas at startup. Input data is validated against Pandera schemas before any processing.
