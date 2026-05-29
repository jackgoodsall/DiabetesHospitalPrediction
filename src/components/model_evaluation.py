from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier

from typing import Dict, Any, Tuple, Optional
import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def binary_classifcation_report(
    y_true, y_predicted, predicted_labels=False, threshold=0.5
) -> Dict[str, float]:
    """
    Generate evaluation metrics for a binary classification task.

    Returns accuracy, precision, recall, F1, AUC-ROC, PR-AUC, and Brier score.
    Ranking metrics (ROC-AUC, PR-AUC, Brier) use raw probabilities; threshold
    metrics (accuracy/precision/recall/F1) use labels binarised at `threshold`.
    """
    assert len(y_predicted) == len(y_true), "Predicted and true vector must have same size"

    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)

    if predicted_labels:
        y_score = y_predicted
        y_label = y_predicted.astype(int)
    else:
        y_score = y_predicted
        y_label = (y_predicted > threshold).astype(int)

    report: Dict[str, float] = {}
    report["accuracy"] = accuracy_score(y_true, y_label)
    report["precision"] = precision_score(y_true, y_label, zero_division=0)
    report["recall"] = recall_score(y_true, y_label, zero_division=0)
    report["f1"] = f1_score(y_true, y_label, zero_division=0)
    report["auc_roc"] = roc_auc_score(y_true, y_score)
    report["pr_auc"] = average_precision_score(y_true, y_score)
    report["brier_score"] = brier_score_loss(y_true, y_score)
    return report


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    strategy: str = "f1",
    target_recall: float = 0.7,
) -> float:
    """
    Choose a decision threshold from validation-set scores.

    Strategies
    ----------
    ``"f1"``           — maximise F1 score
    ``"recall@X"``     — smallest threshold that achieves recall >= target_recall
                         (pass target_recall or embed it as ``"recall@0.8"``)
    ``"default"``      — return 0.5 unchanged
    """
    if strategy == "default":
        return 0.5

    # Support inline target: "recall@0.75"
    if strategy.startswith("recall@"):
        target_recall = float(strategy.split("@")[1])
        strategy = "recall"

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # thresholds has length len(precisions) - 1
    if strategy == "f1":
        with np.errstate(invalid="ignore", divide="ignore"):
            f1s = np.where(
                (precisions[:-1] + recalls[:-1]) > 0,
                2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
                0.0,
            )
        best_idx = int(np.argmax(f1s))
        return float(thresholds[best_idx])

    if strategy == "recall":
        # Find smallest threshold (highest recall) that satisfies >= target
        valid = recalls[:-1] >= target_recall
        if not valid.any():
            return float(thresholds[0])
        return float(thresholds[valid][-1])

    raise ValueError(f"Unknown threshold strategy: '{strategy}'")


def dummy_baseline_metrics(y_true: np.ndarray, strategy: str = "stratified", random_state: int = 0) -> Dict[str, float]:
    """Return metrics for a DummyClassifier — every real metric should beat these."""
    clf = DummyClassifier(strategy=strategy, random_state=random_state)
    # DummyClassifier only needs y to fit
    clf.fit(np.zeros((len(y_true), 1)), y_true)
    y_prob = clf.predict_proba(np.zeros((len(y_true), 1)))[:, 1]
    return binary_classifcation_report(y_true, y_prob)


# ── Plotting helpers (return file-like objects so they can be logged) ──────────

def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, model_name: str = "") -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC-ROC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, model_name: str = "") -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"PR-AUC = {ap:.3f}")
    ax.axhline(baseline, color="grey", linestyle="--", lw=1, label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_calibration_curve(
    y_true: np.ndarray, y_score: np.ndarray, model_name: str = "", n_bins: int = 10
) -> plt.Figure:
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy="uniform"
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_predicted, fraction_of_positives, "s-", lw=2, label=model_name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig
