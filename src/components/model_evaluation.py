from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


from typing import Dict, Any

import os

import numpy as np


def binary_classifcation_report(
    y_true, y_predicted, predicted_labels=False, threshold=0.5
) -> Dict[str, float]:
    """
    Helper function to generate a report for a binary classifcation task.

    Calculates:
        Accuracy
        Precision
        Recall
        F1 Score
        AUC ROC

    Returns:
        report_dictionary: Dict[str, float]
    """
    assert len(y_predicted) == len(y_true), "Predicted and true vector must have same size"

    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)

    # Separate the continuous score (used for ranking metrics like ROC-AUC) from
    # the thresholded label (used for accuracy/precision/recall/F1). Computing
    # ROC-AUC on thresholded 0/1 labels is incorrect — it discards the ranking
    # information the metric is meant to measure.
    if predicted_labels:
        # Caller already passed hard labels; no underlying score is available.
        y_score = y_predicted
        y_label = y_predicted.astype(int)
    else:
        y_score = y_predicted
        y_label = (y_predicted > threshold).astype(int)

    report_dictionary = {}

    report_dictionary["accuracy"] = accuracy_score(y_true, y_label)
    report_dictionary["precision"] = precision_score(y_true, y_label, zero_division=0)
    report_dictionary["recall"] = recall_score(y_true, y_label, zero_division=0)
    report_dictionary["f1"] = f1_score(y_true, y_label, zero_division=0)
    report_dictionary["auc_roc"] = roc_auc_score(y_true, y_score)

    return report_dictionary
