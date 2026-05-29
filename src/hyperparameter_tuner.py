"""
Optuna-based hyperparameter tuning with MLflow integration.

Each model has a dedicated search-space function that maps an optuna.Trial to a
dict of constructor kwargs.  The objective minimises / maximises OOF PR-AUC
(or another metric from TuningConfig) using the same group-aware CV that the
main pipeline uses, so the optimisation target is leak-free and consistent.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import optuna
from optuna_integration.mlflow import MLflowCallback
import mlflow

from components.training_splits import cross_validation_splits
from model_builder import build_estimator

logger = logging.getLogger(__name__)

# Suppress optuna's per-trial INFO noise; WARNING keeps study summaries visible.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Per-model search spaces ────────────────────────────────────────────────────

def _xgboost_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
    }


def _lightgbm_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "verbose": -1,
    }


def _catboost_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": 0,
    }


def _random_forest_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }


def _extra_trees_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }


def _logistic_regression_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "max_iter": 1000,
        "solver": "lbfgs",
        "n_jobs": -1,
    }


def _gradient_boosting_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }


_SEARCH_SPACES = {
    "xgboost": _xgboost_space,
    "lightgbm": _lightgbm_space,
    "catboost": _catboost_space,
    "random_forest": _random_forest_space,
    "extra_trees": _extra_trees_space,
    "logistic_regression": _logistic_regression_space,
    "gradient_boosting": _gradient_boosting_space,
}


# ── Objective ─────────────────────────────────────────────────────────────────

def _make_objective(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    n_splits: int,
    seed: int,
    metric: str,
):
    """Return a closure that Optuna calls for each trial."""
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, brier_score_loss

    _metric_fn = {
        "pr_auc": average_precision_score,
        "auc_roc": roc_auc_score,
        "f1": lambda yt, ys: f1_score(yt, (ys > 0.5).astype(int), zero_division=0),
        "brier_score": brier_score_loss,
    }[metric]
    _higher_is_better = metric != "brier_score"

    space_fn = _SEARCH_SPACES.get(model_name)
    if space_fn is None:
        raise ValueError(f"No search space defined for model '{model_name}'")

    def objective(trial: optuna.Trial) -> float:
        params = space_fn(trial)
        model = build_estimator(model_name, **params)

        strategy = "stratified_group" if groups is not None else "stratified"
        splitter = cross_validation_splits(
            X, y=y, groups=groups, n_splits=n_splits,
            stratergy=strategy, return_indices=True, random_state=seed,
        )

        fold_scores = []
        for train_idx, val_idx in splitter:
            model.fit(X[train_idx], y[train_idx])
            y_prob = model.predict_proba(X[val_idx])[:, 1]
            fold_scores.append(_metric_fn(y[val_idx], y_prob))

        return float(np.mean(fold_scores))

    return objective


# ── Public API ─────────────────────────────────────────────────────────────────

def tune(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_trials: int = 50,
    metric: str = "pr_auc",
    direction: str = "maximize",
    cv_folds: int = 3,
    seed: int = 0,
    timeout: Optional[int] = None,
    show_progress_bar: bool = False,
    mlflow_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run an Optuna study for `model_name` and return the best hyperparameters.

    Results are logged to MLflow as child runs via MLflowCallback.  The returned
    dict can be passed directly to `build_estimator(model_name, **best_params)`.
    """
    objective = _make_objective(
        model_name, X_train, y_train, groups, cv_folds, seed, metric
    )

    mlflow_cb = MLflowCallback(
        metric_name=f"oof_{metric}",
        mlflow_kwargs={"nested": True},
    )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction=direction,
        study_name=f"{model_name}_tuning",
        sampler=sampler,
    )

    logger.info(
        f"Optuna tuning: {model_name} | {n_trials} trials | "
        f"optimising {direction} {metric} | {cv_folds}-fold CV"
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress_bar,
        callbacks=[mlflow_cb],
    )

    best = study.best_trial
    logger.info(
        f"Best trial #{best.number}: {metric}={best.value:.4f} | params={best.params}"
    )

    # Only log summary to the parent run when one is active (avoids auto-creating
    # a throwaway run in test contexts or when called standalone).
    if mlflow.active_run() is not None:
        try:
            mlflow.log_metric(f"tuning_best_{metric}", best.value)
            mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        except Exception:
            logger.warning("Could not log tuning summary to MLflow.", exc_info=True)

    return best.params
