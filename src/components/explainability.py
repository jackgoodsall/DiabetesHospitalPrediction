"""
explainability.py
-----------------
SHAP-based model explainability for the trained classifier.

Generates and persists the standard SHAP plots for a fitted model and logs them
as MLflow artifacts:

  beeswarm   - global feature importance + direction of effect (per-sample)
  bar        - global mean(|SHAP|) ranking (clean view for reports)
  waterfall  - explanation of a single prediction (one test patient)
  dependence - effect of the single most important feature across the sample

Design notes:
  * The explainer is chosen from the model family — TreeExplainer for the
    gradient-boosted / forest models (exact and fast, no background data needed),
    LinearExplainer for logistic regression, and a generic fallback otherwise.
  * SHAP is computed on a sample of the *test* set only, so explanations reflect
    held-out behaviour and stay tractable on the wide one-hot feature space.
  * For binary classifiers that return per-class SHAP values, the positive
    (readmitted) class is selected so plots are interpretable.
  * The whole step is best-effort: callers should treat a failure as non-fatal,
    since explainability must never break a training run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: no display needed when saving figures
import matplotlib.pyplot as plt

import mlflow
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

# Model class-name fragments that TreeExplainer handles exactly.
_TREE_MODEL_HINTS = (
    "XGB",
    "LGBM",
    "CatBoost",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "HistGradientBoosting",
    "DecisionTree",
)


def _build_explainer(model, background: pd.DataFrame):
    """Select an appropriate SHAP explainer for the given fitted model."""
    name = type(model).__name__
    if any(hint in name for hint in _TREE_MODEL_HINTS):
        logger.info(f"Using shap.TreeExplainer for {name}")
        return shap.TreeExplainer(model)
    if "Linear" in name or "LogisticRegression" in name:
        logger.info(f"Using shap.LinearExplainer for {name}")
        return shap.LinearExplainer(model, background)
    # Generic (model-agnostic) fallback with a small background set.
    logger.info(f"Using generic shap.Explainer for {name}")
    bg = shap.utils.sample(background, min(100, len(background)))
    return shap.Explainer(model, bg)


def _explain(explainer, X_df: pd.DataFrame):
    """Run the explainer, tolerating explainers that lack check_additivity."""
    try:
        return explainer(X_df, check_additivity=False)
    except TypeError:
        return explainer(X_df)


def _to_positive_class(explanation):
    """Collapse a (n, features, n_classes) explanation to the positive class."""
    values = explanation.values
    if getattr(values, "ndim", 2) == 3:
        return explanation[:, :, -1]
    return explanation


def _save_current_figure(path: Path) -> None:
    fig = plt.gcf()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _safe_name(feature: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(feature))[:60]


def compute_and_log_shap(
    model,
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    model_name: str,
    output_dir: str = "artefacts/shap",
    sample_size: int = 2000,
    max_display: int = 20,
    random_state: int = 0,
    log_to_mlflow: bool = True,
) -> List[Path]:
    """
    Compute SHAP values for ``model`` on ``X`` and persist the standard plots.

    Args:
        model: a fitted classifier.
        X: transformed feature matrix (the test set — no leakage).
        feature_names: column names matching ``X``'s second axis.
        model_name: used to prefix the output filenames.
        sample_size: cap on rows used for SHAP (subsampled if larger).
        max_display: number of features to show in summary plots.

    Returns:
        The list of saved plot paths (also logged to MLflow when a run is active).
    """
    X = np.asarray(X)
    feature_names = list(feature_names)
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"X width ({X.shape[1]})"
        )

    # Subsample for tractable computation and readable plots.
    if sample_size and len(X) > sample_size:
        rng = np.random.default_rng(random_state)
        sel = rng.choice(len(X), size=sample_size, replace=False)
        X = X[sel]

    X_df = pd.DataFrame(X, columns=feature_names)

    explainer = _build_explainer(model, X_df)
    explanation = _to_positive_class(_explain(explainer, X_df))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: List[Path] = []

    # 1. Beeswarm — global importance with direction.
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    beeswarm_path = out / f"{model_name}_shap_beeswarm.png"
    _save_current_figure(beeswarm_path)
    artifacts.append(beeswarm_path)

    # 2. Bar — clean global mean(|SHAP|) ranking.
    shap.plots.bar(explanation, max_display=max_display, show=False)
    bar_path = out / f"{model_name}_shap_bar.png"
    _save_current_figure(bar_path)
    artifacts.append(bar_path)

    # 3. Waterfall — single-patient explanation (first sampled row).
    shap.plots.waterfall(explanation[0], max_display=max_display, show=False)
    waterfall_path = out / f"{model_name}_shap_waterfall.png"
    _save_current_figure(waterfall_path)
    artifacts.append(waterfall_path)

    # 4. Dependence — most important feature.
    mean_abs = np.abs(explanation.values).mean(axis=0)
    top_idx = int(np.argmax(mean_abs))
    shap.plots.scatter(explanation[:, top_idx], show=False)
    dependence_path = out / f"{model_name}_shap_dependence_{_safe_name(feature_names[top_idx])}.png"
    _save_current_figure(dependence_path)
    artifacts.append(dependence_path)

    if log_to_mlflow and mlflow.active_run() is not None:
        for path in artifacts:
            mlflow.log_artifact(str(path), artifact_path="shap")
        logger.info(f"Logged {len(artifacts)} SHAP plots to MLflow")

    return artifacts
