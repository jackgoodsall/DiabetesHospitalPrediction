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
    assert len(y_predicted) == len(y_true)
    "Predicted and true vector must have same size"

    if not predicted_labels:
        y_predicted = (y_predicted > threshold).astype(int)

    report_dictionary = {}

    accuracy = accuracy_score(y_true, y_predicted)
    precision = precision_score(y_true, y_predicted)
    recall = recall_score(y_true, y_predicted)
    f1 = f1_score(y_true, y_predicted)
    auc_roc = roc_auc_score(y_true, y_predicted)

    report_dictionary["accuracy"] = accuracy
    report_dictionary["precision"] = precision
    report_dictionary["recall"] = recall
    report_dictionary["f1"] = f1
    report_dictionary["auc_roc"] = auc_roc

    return report_dictionary
