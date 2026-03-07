# Diabetes Hospital Readmission Prediction

[![CI](https://github.com/jackgoodsall/DiabetesHospitalPrediction/actions/workflows/ci.yml/badge.svg)](https://github.com/jackgoodsall/DiabetesHospitalPrediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end MLOps project for predicting 30-day hospital readmission of diabetic patients, built on 100,000+ real patient records from the [UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

The primary goal is demonstrating **production-grade ML engineering practices** — not just modelling. The pipeline is fully decoupled, config-driven, and reproducible. Swapping in a different dataset or model family requires only config changes.

---

## Highlights

- **Fully decoupled pipeline** — data ingestion, feature engineering, training, and evaluation are independent components wired together by a single runner
- **Config-driven** — all pipeline behaviour (features, models, hyperparameters, MLflow experiment name) is controlled via `configs/run_config.yaml`; no magic numbers in code
- **Pydantic config validation** — configs are parsed into typed models at startup; misconfiguration raises a clear error before any data is touched
- **MLflow experiment tracking** — every run logs parameters, metrics, the config file, the fitted transformer, and the trained model as artifacts; nested runs per model within a parent pipeline run
- **MLflow Model Registry** — models are registered and promoted to stages (`Staging` / `Production`) for lifecycle management
- **CLI inference script** — load from the registry or local `.joblib` files and score new patient records from the command line
- **Dockerised** — reproducible environment via a self-contained Docker image
- **CI with GitHub Actions** — tests run on every push and pull request via `uv`

---

## Tech Stack

| Area | Tools |
|------|-------|
| ML / Data | scikit-learn, XGBoost, pandas, NumPy |
| Experiment tracking | MLflow (tracking + model registry) |
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
│   └── ci.yml                   # GitHub Actions CI pipeline
├── configs/
│   └── run_config.yaml          # Single source of truth for all pipeline config
├── data/
│   ├── raw_data/                # Raw CSVs (gitignored)
│   └── processed_data/          # Cleaned data output
├── notebooks/
│   └── data_visualisation.ipynb # Exploratory data analysis
├── src/
│   ├── components/
│   │   ├── config.py            # Pydantic config models + YAML loader
│   │   ├── data_engineering.py  # sklearn ColumnTransformer pipeline
│   │   ├── model_evaluation.py  # Metrics and classification report
│   │   └── training_splits.py   # Stratified train/test split
│   ├── data_ingestion.py        # Raw data loading and cleaning
│   ├── model_builder.py         # Estimator factory (XGBoost, LightGBM, CatBoost, RF, LR, etc.)
│   ├── model_trainer.py         # Cross-validated training with OOF metrics
│   ├── pipeline_runner.py       # Orchestrates the full pipeline
│   └── inference.py             # CLI inference against registry or local files
├── artefacts/
│   ├── models/                  # Saved .joblib model files
│   └── pipeline/                # Saved transformer artifacts
├── tests/                       # pytest test suite
├── Dockerfile
└── pyproject.toml
```

---

## Pipeline Architecture

```
configs/run_config.yaml
        │
        ▼
┌─────────────────────┐
│   Data Ingestion    │  Load CSV → drop redundant columns → binary target
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Data Engineering   │  Numerical impute + scale │ Categorical impute + OHE
└──────────┬──────────┘  (fitted on train, applied to test — no leakage)
           │
           ▼
┌─────────────────────┐
│   Model Training    │  Stratified K-fold CV → OOF metrics → refit on full train
└──────────┬──────────┘  (one nested MLflow run per model)
           │
           ▼
┌─────────────────────┐
│    Evaluation       │  Test-set metrics: AUC-ROC, F1, Precision, Recall, Accuracy
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  MLflow Artifacts   │  Config, transformer (.joblib), model → MLflow registry
└─────────────────────┘
```

---

## Results

Baseline XGBoost run (no hyperparameter tuning, no class imbalance handling):

| Metric | OOF (CV) | Test set |
|--------|----------|----------|
| AUC-ROC | 0.627 | 0.629 |
| Accuracy | 0.634 | 0.635 |
| F1 | 0.571 | 0.576 |
| Precision | 0.617 | 0.626 |
| Recall | 0.531 | 0.533 |

OOF and test-set metrics track closely, indicating no significant overfitting.

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Raw dataset CSV placed at `data/raw_data/diabetic_data.csv`

### Install

```bash
git clone https://github.com/jackgoodsall/DiabetesHospitalPrediction.git
cd DiabetesHospitalPrediction
uv sync
```

### Run the full pipeline

```bash
uv run python src/pipeline_runner.py
```

This runs data ingestion → feature engineering → model training → evaluation → MLflow artifact logging in a single command.

### Run inference

```bash
# Load from MLflow registry (Production stage)
uv run python src/inference.py \
    --input data/raw_data/diabetic_data.csv \
    --model-name xgboost

# Load from local files
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

Navigate to `http://localhost:5000` to explore runs, compare metrics, and inspect artifacts.

### Run with Docker

```bash
docker build -t diabetes-prediction .
docker run diabetes-prediction
```

### Run tests

```bash
uv run pytest tests/ -v
```

---

## Configuration

All pipeline behaviour is controlled from `configs/run_config.yaml`. Key sections:

```yaml
model:
  model_names: [xgboost]   # Add logistic_regression, random_forest, etc.

data_engineering:
  numerical_imputer_strat: mean
  scaler: standard          # or minmax

xgboost:
  max_depth: 6
  eta: 0.1
  n_estimators: 1000
  # ... full XGBoost param set
```

Config is validated against Pydantic schemas at startup — invalid or missing fields raise a descriptive error immediately.
