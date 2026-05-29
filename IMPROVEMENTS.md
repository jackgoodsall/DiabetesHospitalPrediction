# ML / Data Science Improvement Roadmap

## Overview

The project has strong **MLOps scaffolding** — decoupled config-driven pipeline,
Pydantic validation, MLflow tracking + registry, Pandera data validation, a FastAPI serving
layer, Docker, and CI. That infrastructure is genuinely CV-worthy.

The **data-science core** has been progressively improved. The items below track what has
been completed and what remains, ordered by impact on the modelling portfolio piece.

---

## Completed

### Item 0 — Evaluation correctness
Probability-based metrics (PR-AUC, Brier score) are now separated from label-based metrics
(precision, recall, F1). `model_evaluation.py` computes both correctly, with `y_score` for
ranking metrics and a threshold-binarised `y_label` for label metrics.

### Item 1 — Group-aware split (patient-level leakage prevention)
`training_splits.py` uses `StratifiedGroupKFold` when a `group_col` is supplied.
`run_config.yaml` sets `split.group_column: patient_nbr`, so all encounters for one patient
stay on the same side of the train/test boundary. The same groups are passed into CV folds
for OOF predictions and Optuna tuning.

### Item 5 — PR-AUC, calibration curves, threshold selection
`model_evaluation.py` now exports:
- `binary_classifcation_report` — accuracy, precision, recall, F1, AUC-ROC, PR-AUC, Brier
- `select_threshold` — F1-optimal or recall@X threshold derived from OOF predictions
- `dummy_baseline_metrics` — `DummyClassifier` baseline so every metric has a floor
- `plot_roc_curve`, `plot_pr_curve`, `plot_calibration_curve` — logged to MLflow per model run

### Item 6 — SHAP explainability
`src/components/explainability.py` generates beeswarm, bar, waterfall, and dependence plots.
Explainer is chosen by model family (TreeExplainer, LinearExplainer, generic fallback).
Plots are logged as MLflow artifacts and the step is best-effort (never breaks a training run).

### Item 7 — Hyperparameter tuning (Optuna)
`src/hyperparameter_tuner.py` runs an Optuna TPE study per model, optimising OOF PR-AUC using
the same group-aware CV as the main pipeline (leak-free objective). Per-model search spaces
cover XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, Logistic Regression, and
Gradient Boosting. Results are logged as MLflow child runs via `MLflowCallback`.
`TuningConfig` in `config.py` + `tuning:` block in `run_config.yaml` control enabling,
n_trials, metric, cv_folds, and timeout.

---

## Remaining

### Item 2 — Reframe target to `<30`-day readmission

**Priority: High · Effort: Low**

`data_ingestion.py:transform_target_to_binary()` still maps **both** `>30` and `<30` to `1`.
The clinically meaningful task (and the one hospitals are penalised on) is readmission
**within 30 days**. Lumping late readmissions into the positive class dilutes the signal.

**What to build**
- Make the target mapping config-driven with two modes:
  - `binary_30d`: `<30 → 1`, `{>30, NO} → 0` (primary clinical task)
  - `binary_any`: current behaviour (kept for comparison)
- Add `data.target_definition` to `run_config.yaml`
- Parametrise `transform_target_to_binary` in `data_ingestion.py`

**Files:** `src/data_ingestion.py`, `configs/run_config.yaml`

---

### Item 3 — Domain feature engineering

**Priority: High · Effort: Medium**

The current `DataEngineeringPipeLine` only imputes, scales, and one-hot encodes. Three
targeted engineering steps substantially improve signal quality:

#### 3a — ICD-9 diagnosis grouping
`diag_1`, `diag_2`, `diag_3` have hundreds of unique raw codes after one-hot encoding,
producing a very wide, sparse feature matrix. Group them into ~18 clinical categories
(circulatory, respiratory, diabetes, injury, etc.) based on ICD-9 ranges. This cuts
cardinality dramatically and lets the model learn clinical patterns.

#### 3b — Keep `discharge_disposition_id`
This column is currently **dropped** in `run_config.yaml` (line ~35). Discharge to a skilled
nursing facility or against medical advice is one of the strongest predictors of readmission
in the literature. It should be kept and one-hot encoded.

#### 3c — Polypharmacy change count
Count how many of the ~24 medication columns have a value of `"Up"` or `"Down"` (indicating
a dose change during the encounter). A single integer feature captures the complexity of the
regimen change better than 24 sparse one-hot columns.

**Files:** new `src/components/feature_engineering.py`, `configs/run_config.yaml`

---

### Item 4 — Class-imbalance handling

**Priority: High · Effort: Low**

The dataset is imbalanced (the `<30` positive class is a minority after re-framing the
target). `model_builder.py` does not pass `class_weight` to any model.

**What to build**
- Add `class_weight: "balanced"` to the default configs for models that support it
  (LogisticRegression, RandomForest, ExtraTrees, GradientBoosting)
- Add `scale_pos_weight` to the XGBoost config (ratio of negatives to positives)
- Make the weight strategy config-driven (`imbalance.strategy: balanced | none`)

**Files:** `src/model_builder.py`, `configs/run_config.yaml`

---

### ~~Item 8 — EDA notebook~~ ✅ Done

Replaced `notebooks/data_visuliation.ipynb` (AutoViz only, typo in filename) with a
structured narrative EDA at `notebooks/data_visualisation.ipynb`. Each of the nine
sections produces a plot and ends with a **Decision** line that maps to a config or
feature-engineering change:

| Section | Finding | Decision |
|---------|---------|----------|
| 1. Target | `>30` dilutes `<30` signal | Use `binary_30d` target |
| 2. Patient leakage | Multi-encounter patients | Group split on `patient_nbr` ✅ |
| 3. Missingness | `weight` ~97% null | Drop `weight`; impute others ✅ |
| 4. ICD-9 cardinality | ~700 raw codes | Group into 18 clinical categories |
| 5. Discharge disposition | Strong rate variation | Remove from drop list |
| 6. Medication changes | Dose-change count correlates | Add `polypharmacy_changes` |
| 7. Numerical features | `number_inpatient` strongest | Keep all |
| 8. Age | Weak discriminator | Keep as ordinal categorical |
| 9. Class imbalance | ~8:1 ratio under `binary_30d` | `scale_pos_weight` / `class_weight` |

---

### Minor — Fix `stratergy` typo in `training_splits.py`

**Priority: Low · Effort: Trivial**

The parameter `stratergy` (line 73, 89 in `training_splits.py`) is a consistent misspelling
of `strategy`. Fix the parameter name and update all call sites.

**Files:** `src/components/training_splits.py`, `src/hyperparameter_tuner.py`

---

## Suggested order of execution

1. **Item 2** — reframe target to `<30` (changes the objective everything else is measured on)
2. **Item 4** — class-imbalance handling (cheap, immediately improves minority-class recall)
3. **Item 3** — feature engineering (biggest performance ceiling lift)
4. **Item 8** — EDA notebook documenting the evidence for items 2–3
5. **Typo fix** — `stratergy` → `strategy`

---

## Summary Table

| Item | Status | Priority | Effort | Primary files |
|------|--------|----------|--------|---------------|
| 0. Eval correctness | ✅ Done | Critical | Low | `model_evaluation.py` |
| 1. Group-aware split | ✅ Done | Critical | Low-Med | `training_splits.py`, YAML |
| 2. Reframe target (`<30`) | Remaining | High | Low | `data_ingestion.py`, YAML |
| 3. Domain feature engineering | Remaining | High | Medium | `feature_engineering.py` (new), YAML |
| 4. Class-imbalance handling | Remaining | High | Low | `model_builder.py`, YAML |
| 5. PR-AUC / calibration / threshold | ✅ Done | High | Low-Med | `model_evaluation.py` |
| 6. SHAP explainability | ✅ Done | High | Low-Med | `explainability.py` |
| 7. Hyperparameter tuning (Optuna) | ✅ Done | Medium | Medium | `hyperparameter_tuner.py`, `config.py`, YAML |
| 8. EDA notebook | ✅ Done | Medium | Low | `notebooks/data_visualisation.ipynb` |
| Typo: `stratergy` | Remaining | Low | Trivial | `training_splits.py` |
