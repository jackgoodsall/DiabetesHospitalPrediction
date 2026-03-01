"""
Tests for components/model_evaluation.py
"""
import pytest
import numpy as np

from components.model_evaluation import binary_classifcation_report


class TestBinaryClassificationReport:

    # ── Happy path ─────────────────────────────────────────────────────────────
    def test_returns_all_required_keys(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        result = binary_classifcation_report(y_true, y_pred)
        assert set(result.keys()) == {"accuracy", "precision", "recall", "f1", "auc_roc"}

    def test_perfect_predictions_score_one(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        # Pass as already-binarised labels
        result = binary_classifcation_report(y, y, predicted_labels=True)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)
        assert result["auc_roc"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y = np.array([0, 1, 0, 1])
        y_flipped = np.array([1, 0, 1, 0])
        result = binary_classifcation_report(y, y_flipped, predicted_labels=True)
        assert result["accuracy"] == pytest.approx(0.0)

    def test_scores_are_between_zero_and_one(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=100)
        y_pred = rng.random(100)
        result = binary_classifcation_report(y_true, y_pred)
        for val in result.values():
            assert 0.0 <= val <= 1.0

    # ── Threshold behaviour ────────────────────────────────────────────────────
    def test_default_threshold_is_half(self):
        """Probabilities exactly at 0.5 should be treated as positive."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.6, 0.4, 0.1])
        result = binary_classifcation_report(y_true, y_prob)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_custom_threshold_changes_predictions(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.4, 0.4, 0.1, 0.1])
        # Default threshold 0.5 → all predicted 0 → recall = 0
        result_default = binary_classifcation_report(y_true, y_prob, threshold=0.5)
        # Low threshold → all predicted 1 → recall = 1
        result_low = binary_classifcation_report(y_true, y_prob, threshold=0.3)
        assert result_low["recall"] == pytest.approx(1.0)
        assert result_default["recall"] == pytest.approx(0.0)

    # ── Edge cases / errors ────────────────────────────────────────────────────
    def test_mismatched_lengths_raises_assertion(self):
        with pytest.raises(AssertionError):
            binary_classifcation_report(np.array([0, 1]), np.array([0, 1, 0]))

    def test_returns_floats(self):
        y = np.array([0, 1, 0, 1])
        result = binary_classifcation_report(y, y, predicted_labels=True)
        for val in result.values():
            assert isinstance(val, float)
