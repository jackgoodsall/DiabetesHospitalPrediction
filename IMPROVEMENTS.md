# ML / Data Science Improvement Roadmap

## Overview

The project already has strong **MLOps scaffolding** — decoupled config-driven pipeline,
Pydantic validation, MLflow tracking + registry, Pandera data validation, a FastAPI serving
layer, Docker, and CI. That infrastructure is genuinely CV-worthy and is **not** repeated here.

What is currently thin is the **data-science core**: problem framing, leakage control,
feature engineering, evaluation correctness, and model performance. A baseline XGBoost scores
**AUC-ROC ≈ 0.63**, which is barely above chance and is the first thing a technical reviewer
will notice. The items below are ordered by how much they raise the ceiling of the project as
a *modelling* portfolio piece — and several of them are the kind of insight that distinguishes
a data scientist from someone who only wired up a pipeline.

> **Already implemented — do not re-add:** FastAPI serving, Pandera schema validation,
> MLflow model registry/promotion, multi-model estimator factory, CLI inference, Docker, CI.

---

## 0. Correctness fixes (do these first — they are bugs, and they're quick) — ✅ DONE

**Priority: Critical · Effort: Low · Status: Implemented**

> Implemented: ROC-AUC now computed on probabilities (`model_evaluation.py`); `split_df`
> stratifies and takes a `random_state`; `cross_validation_splits` gained `stratified` and
> `stratified_group` strategies and the trainer now uses stratified CV by default.

These are small changes with outsized credibility impact: a reviewer who spots them in your
code (and they will) reads them as "doesn't understand evaluation." Fixing them proactively
flips that signal.

### 0a. AUC-ROC is computed on hard labels, not probabilities
- **Where:** `src/components/model_evaluation.py`
- **Bug:** `binary_classifcation_report()` binarizes `y_predicted` at the 0.5 threshold
  (`y_predicted = (y_predicted > threshold).astype(int)`) and then passes those **0/1 labels**
  into `roc_auc_score`. ROC-AUC must be computed on the continuous score/probability.
- **Fix:** compute threshold-dependent metrics (accuracy/precision/recall/F1) on the
  thresholded labels, but compute `roc_auc_score` (and PR-AUC, Brier score) on the raw
  probabilities. Keep both the probability vector and the label vector in the function.
- **Why it matters on the CV:** the headline metric in the README is currently understated and
  computed incorrectly. Fixing it both improves the number and shows you know the difference
  between ranking metrics and threshold metrics.

### 0b. The train/test split is neither stratified nor reproducible
- **Where:** `src/components/training_splits.py` → `split_df()`
- **Bug:** the function accepts `startify` and `group_cols` arguments but ignores both — it
  calls `train_test_split(df, test_size=...)` with no `stratify`, no `random_state`. The README
  claims a "stratified train/test split"; the code does not do this.
- **Fix:** honour `stratify=df[target]` and pass a fixed `random_state`. (Also fix the
  `startify` → `stratify` typo.) See item 1 for the group-aware version, which supersedes this.

### 0c. CV is plain `KFold`, not `StratifiedKFold`
- **Where:** `cross_validation_splits()` — `stratergy="stratified"` is accepted but never
  implemented; only `KFold` exists.
- **Fix:** implement the stratified branch with `StratifiedKFold` (needs `y` passed through).
  On an imbalanced target, unstratified folds give noisy OOF estimates.

---

## 1. Patient-level leakage — group-aware splitting — ✅ DONE

**Priority: Critical · Effort: Low-Medium · Status: Implemented**

> Implemented: `patient_nbr` is retained through ingestion (removed from `drop_columns`,
> un-forbidden in `CleanedDataSchema`), the train/test split and CV are grouped on it via
> `StratifiedGroupKFold`, and it is dropped from the feature matrix immediately after the
> split. Controlled by the new `split:` section in `run_config.yaml`. Verified: 0 patient
> overlap between train and test.

This is the single most impressive fix available on this dataset, and almost everyone who does
this project misses it.

### The problem
The UCI dataset contains **multiple hospital encounters per patient** (`patient_nbr` repeats).
The pipeline currently drops `patient_nbr` in `configs/run_config.yaml → data.drop_columns`
*before* splitting, then does a random encounter-level split. Result: the same patient's
encounters appear in both train and test, so the model can memorise patient-specific signal and
the reported metrics are optimistically biased. This is textbook data leakage.

### What to build
- Retain `patient_nbr` through ingestion and the split, then drop it immediately after.
- Use `sklearn.model_selection.GroupShuffleSplit` for the train/test split and
  `StratifiedGroupKFold` for CV, grouping on `patient_nbr`.
- Wire `group_cols` (already a stub argument in `split_df`) through to these splitters.

### Where
- `src/components/training_splits.py` — implement group-aware split + CV
- `src/pipeline_runner.py` — keep `patient_nbr` until after the split, then drop
- `configs/run_config.yaml` — add a `split:` section (`group_column`, `test_size`, `seed`)

### What it adds to the CV
- Demonstrates you understand leakage beyond the trivial "fit scaler on train only" case
- "I found and fixed a patient-level leak that inflated metrics" is a great interview story
- Honest, leakage-free metrics are more credible than a suspiciously high score

---

## 2. Problem framing — predict `<30`-day readmission

**Priority: High · Effort: Low**

### The problem
`data_ingestion.py → transform_target_to_binary()` maps **both** `>30` and `<30` to `1`.
The clinically and academically meaningful task (and the one the dataset is famous for) is
**readmission within 30 days** — the metric hospitals are actually penalised on. Lumping
`>30` (a late, weakly-related event) into the positive class dilutes the signal and is part of
why AUC is stuck near 0.63.

### What to build
- Make the target mapping **config-driven** with at least two modes:
  - `binary_30d`: `<30 → 1`, `{>30, NO} → 0` (recommended primary task)
  - `binary_any`: current behaviour (kept for comparison)
- Document the choice and its clinical rationale in the README.

### Where
- `src/data_ingestion.py` — parametrise the mapping
- `configs/run_config.yaml` — add `data.target_definition`

### What it adds to the CV
- Shows you frame the problem around the real-world decision, not whatever is easiest to code
- Lets you tell a comparison story ("`<30` is harder but the right target; here's the trade-off")

---

## 3. Domain feature engineering (the biggest performance lever)

**Priority: High · Effort: Medium**

Raw one-hot encoding of high-cardinality fields is the main reason the model underperforms.

### 3a. Group ICD-9 diagnosis codes into clinical categories
`diag_1`, `diag_2`, `diag_3` are ICD-9 codes with **hundreds of distinct values each**.
One-hot encoding them (current behaviour) produces a huge, sparse, noisy matrix. The
well-established approach (Strack et al., 2014, the paper that published this dataset) is to
bucket codes into ~9 clinical groups: Circulatory, Respiratory, Digestive, Diabetes, Injury,
Musculoskeletal, Genitourinary, Neoplasms, Other. This collapses thousands of columns into a
handful of meaningful ones and typically gives the largest single AUC gain.

### 3b. Reconsider the dropped administrative columns
`admission_type_id`, `discharge_disposition_id`, and `admission_source_id` are currently
dropped. `discharge_disposition_id` in particular is **strongly predictive** (e.g. discharged
to hospice / expired vs. home) and is used in the canonical analyses. Map their ID codes to
human-readable categories (the dataset ships an `IDS_mapping.csv`) and keep them. Note: rows
where the patient died or entered hospice cannot be readmitted and are usually filtered out —
another defensible, documented modelling decision.

### 3c. Derived utilisation features
- `total_prior_visits = number_outpatient + number_emergency + number_inpatient`
- `n_medication_changes` / `n_meds_on` aggregated across the drug columns
- A "service utilisation" interaction (`num_medications × time_in_hospital`)

### Where
- New file: `src/components/feature_engineering.py` (ICD-9 grouping, ID mapping, derived feats)
  as a sklearn-compatible transformer that slots into the existing `ColumnTransformer`
- `configs/run_config.yaml` — toggle each feature group on/off
- `notebooks/` — show before/after cardinality and AUC

### What it adds to the CV
- Domain knowledge applied to features is the clearest "real data scientist" signal
- Citing the source paper's methodology shows literature awareness
- This is where the headline metric actually moves

---

## 4. Class-imbalance handling

**Priority: High · Effort: Low**

Once the target is `<30`-day readmission (~11% positive), imbalance is severe and must be
handled explicitly.

### What to build
- Log the class ratio to MLflow at ingestion.
- Add config-driven `scale_pos_weight` (XGBoost/LightGBM) and `class_weight="balanced"`
  (RF / LogisticRegression / ExtraTrees) via the estimator factory.
- Optionally compare against resampling (SMOTE / random undersampling) using
  `imbalanced-learn`, applied **inside** the CV folds only (never before the split).

### Where
- `src/model_builder.py` — inject weighting params from config
- `src/data_ingestion.py` / `pipeline_runner.py` — log class distribution
- `configs/run_config.yaml` — `imbalance:` section

### What it adds to the CV
- Imbalance is the norm in medical/fraud data — a standard interview topic
- Pairs naturally with the PR-AUC / threshold work in item 5

---

## 5. Honest evaluation: PR-AUC, calibration, and threshold selection — ✅ DONE

**Priority: High · Effort: Low-Medium · Status: Implemented**

> Implemented: `binary_classifcation_report` now returns `pr_auc` and `brier_score` alongside
> the existing metrics. `select_threshold` chooses a decision threshold from OOF scores using
> configurable strategies (`"f1"`, `"recall@X"`, `"default"`). `dummy_baseline_metrics` runs a
> `DummyClassifier` and returns the same metric dict for comparison. Three plot helpers
> (`plot_roc_curve`, `plot_pr_curve`, `plot_calibration_curve`) produce Matplotlib figures that
> `pipeline_runner.py` saves to `artefacts/evaluation_plots/` and logs to MLflow under
> `evaluation_plots/`. Dummy baseline metrics are logged as `dummy_*` keys. Threshold strategy
> is controlled by `evaluation.threshold_strategy` in `run_config.yaml`. Covered by 10 new
> tests in `tests/test_model_evaluation.py`.

On an imbalanced clinical problem, ROC-AUC and a hard 0.5 threshold are the wrong defaults.

### What was built
- **PR-AUC (average precision)** and **Brier score** added to the metrics report.
- **Calibration curve** (reliability diagram) logged to MLflow as an artifact.
- **Decision-threshold selection** (`"f1"` maximise, `"recall@X"` target operating point)
  computed on OOF predictions (validation signal), applied to the test set report.
- **`DummyClassifier` baseline** — all metrics logged with `dummy_` prefix so gains are
  immediately legible relative to a trivial model.

### Where
- `src/components/model_evaluation.py` — extended report + `select_threshold` + plot helpers
- `src/pipeline_runner.py` — threshold selection, baseline logging, curve artifacts
- `configs/run_config.yaml` — `evaluation:` section (`threshold_strategy`, `dummy_strategy`)

### What it adds to the CV
- Shows evaluation maturity beyond a single accuracy number
- Calibration + threshold selection is exactly the stakeholder-aware thinking healthcare needs

---

## 6. Model explainability (SHAP) — ✅ DONE

**Priority: High · Effort: Low-Medium · Status: Implemented**

> Implemented in `src/components/explainability.py`: explainer auto-selected by model family
> (`TreeExplainer` for GBMs/forests, `LinearExplainer` for logistic regression), computed on a
> sample of the **test set**, producing beeswarm / bar / waterfall / dependence plots that are
> logged to MLflow under `shap/`. Wired into `pipeline_runner.py` (best-effort, never breaks a
> run) and controlled by the `explainability:` section in `run_config.yaml`. Covered by
> `tests/test_explainability.py`.

### What to build
SHAP values for the trained model, logged as MLflow artifacts.

| Plot | Purpose |
|------|---------|
| Beeswarm summary | Global feature importance by mean abs SHAP |
| Bar | Clean global importance for reports |
| Waterfall | Single-patient explanation from the test set |
| Dependence | Effect of a key feature (e.g. `number_inpatient`) |

### Where
- New: `src/components/explainability.py` (`shap.TreeExplainer` for GBMs/RF,
  `LinearExplainer` for LogReg), computed on the test set only
- `src/pipeline_runner.py` — log SHAP plots after final training

### What it adds to the CV
- Explainability is effectively mandatory for healthcare ML and expected at senior level
- Lets you sanity-check that the model relies on clinically sensible features (and validates
  the feature-engineering work in item 3)

---

## 7. Hyperparameter tuning (Optuna) — ✅ DONE

**Priority: Medium · Effort: Medium · Status: Implemented**

> Implemented: `src/hyperparameter_tuner.py` defines per-model TPE search spaces for all 7
> supported estimators (XGBoost, LightGBM, CatBoost, RandomForest, ExtraTrees, LogReg,
> GradientBoosting). The `tune()` function runs an Optuna study with `MLflowCallback` logging
> each trial as a nested MLflow run, optimising OOF PR-AUC (or any `TuningConfig` metric) using
> the same group-aware CV as the main pipeline. Best params are merged over base config and used
> for the final refit. Controlled by the `tuning:` section in `run_config.yaml`
> (`enabled: false` by default). `TuningConfig` Pydantic model validates config. 20 new tests
> in `tests/test_hyperparameter_tuner.py`.

### What was built
- `src/hyperparameter_tuner.py` — `tune()` public API + search spaces for all models
- `src/pipeline_runner.py` — optional tuning phase (stage 3a) before final refit
- `src/components/config.py` — `TuningConfig` Pydantic model
- `configs/run_config.yaml` — `tuning:` section (`enabled`, `n_trials`, `metric`, `direction`, `cv_folds`, `timeout`)

### What it adds to the CV
- Optuna + MLflow integration is industry-standard and a strong signal
- TPE sampler with group-aware OOF objective: tuning on a leak-free, correctly framed target

---

## 8. EDA notebook — narrative, not just AutoViz

**Priority: Medium · Effort: Low**

### What to build
Replace the AutoViz-only notebook with a structured EDA that *justifies* the decisions above
(fix the filename typo `data_visuliation.ipynb` → `data_visualisation.ipynb`).

| Section | Content |
|---------|---------|
| Target distribution | Class counts + imbalance ratio for `<30` vs `>30` vs `NO` |
| Leakage check | Encounters-per-patient distribution → motivates group split (item 1) |
| Missingness | % null per column (`weight`, `medical_specialty`, payer fields) |
| Diagnosis cardinality | Unique ICD-9 counts → motivates grouping (item 3a) |
| Discharge disposition | Readmission rate by disposition → motivates keeping it (item 3b) |
| Correlations | Numeric correlations and association with the target |

### What it adds to the CV
- Each EDA section becomes the *evidence* for a modelling choice — analysis before modelling
- EDA notebooks are often the artifact shared with non-technical stakeholders

---

## Suggested order of execution

1. **Item 0** — correctness fixes (cheap, restores credibility of all metrics)
2. **Item 1** — group-aware split (everything downstream depends on a leak-free split)
3. **Item 2** — reframe target to `<30` days
4. **Item 3** — feature engineering (biggest performance gain)
5. **Item 4 + 5** — imbalance handling + honest evaluation (interpret the gains correctly)
6. **Item 6** — SHAP (explain the improved model)
7. **Item 7** — Optuna (squeeze the last few points on a now-correct objective)
8. **Item 8** — EDA notebook documenting the evidence for all of the above

## Summary Table

| Item | Priority | Effort | Primary files |
|------|----------|--------|---------------|
| 0. Eval correctness fixes ✅ | Critical | Low | `model_evaluation.py`, `training_splits.py` |
| 1. Group-aware split (leakage) ✅ | Critical | Low-Med | `training_splits.py`, `pipeline_runner.py`, YAML |
| 2. Reframe target (`<30`) | High | Low | `data_ingestion.py`, YAML |
| 3. Domain feature engineering | High | Medium | `feature_engineering.py` (new), YAML |
| 4. Class-imbalance handling | High | Low | `model_builder.py`, `pipeline_runner.py`, YAML |
| 5. PR-AUC / calibration / threshold | High | Low-Med | `model_evaluation.py`, `pipeline_runner.py` |
| 6. SHAP explainability ✅ | High | Low-Med | `explainability.py` (new), `pipeline_runner.py` |
| 7. Hyperparameter tuning (Optuna) | Medium | Medium | `hyperparameter_tuner.py` (new), `config.py`, YAML |
| 8. EDA notebook | Medium | Low | `notebooks/data_visualisation.ipynb` |
