"""
Tests for components/model_evaluation.py
"""
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from components.model_evaluation import (
    binary_classifcation_report,
    select_threshold,
    dummy_baseline_metrics,
    plot_roc_curve,
    plot_pr_curve,
    plot_calibration_curve,
)


class TestBinaryClassificationReport:

    # ── Happy path ─────────────────────────────────────────────────────────────
    def test_returns_all_required_keys(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        result = binary_classifcation_report(y_true, y_pred)
        assert set(result.keys()) == {"accuracy", "precision", "recall", "f1", "auc_roc", "pr_auc", "brier_score"}

    def test_perfect_predictions_score_one(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        result = binary_classifcation_report(y, y, predicted_labels=True)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)
        assert result["auc_roc"] == pytest.approx(1.0)
        assert result["pr_auc"] == pytest.approx(1.0)
        assert result["brier_score"] == pytest.approx(0.0)

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
        # brier_score is also in [0, 1]
        for k, val in result.items():
            assert 0.0 <= val <= 1.0, f"{k} = {val} out of [0, 1]"

    # ── Threshold behaviour ────────────────────────────────────────────────────
    def test_default_threshold_is_half(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.6, 0.4, 0.1])
        result = binary_classifcation_report(y_true, y_prob)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_custom_threshold_changes_predictions(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.4, 0.4, 0.1, 0.1])
        result_default = binary_classifcation_report(y_true, y_prob, threshold=0.5)
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


class TestSelectThreshold:

    def _scores(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_score = np.where(y_true == 1, rng.uniform(0.5, 1.0, 200), rng.uniform(0.0, 0.6, 200))
        return y_true, y_score

    def test_default_returns_half(self):
        y_true, y_score = self._scores()
        assert select_threshold(y_true, y_score, strategy="default") == pytest.approx(0.5)

    def test_f1_threshold_in_unit_interval(self):
        y_true, y_score = self._scores()
        t = select_threshold(y_true, y_score, strategy="f1")
        assert 0.0 <= t <= 1.0

    def test_recall_strategy_achieves_target(self):
        y_true, y_score = self._scores()
        target = 0.7
        t = select_threshold(y_true, y_score, strategy="recall", target_recall=target)
        labels = (y_score >= t).astype(int)
        from sklearn.metrics import recall_score
        achieved = recall_score(y_true, labels, zero_division=0)
        assert achieved >= target - 0.05  # small tolerance for edge quantisation

    def test_recall_inline_string(self):
        y_true, y_score = self._scores()
        t1 = select_threshold(y_true, y_score, strategy="recall@0.7")
        t2 = select_threshold(y_true, y_score, strategy="recall", target_recall=0.7)
        assert t1 == pytest.approx(t2)

    def test_unknown_strategy_raises(self):
        y_true, y_score = self._scores()
        with pytest.raises(ValueError, match="Unknown threshold strategy"):
            select_threshold(y_true, y_score, strategy="nonsense")


class TestDummyBaseline:

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=100)
        result = dummy_baseline_metrics(y)
        assert "auc_roc" in result
        assert "pr_auc" in result
        assert "brier_score" in result

    def test_stratified_dummy_near_base_rate(self):
        y = np.array([0] * 90 + [1] * 10)
        result = dummy_baseline_metrics(y, strategy="stratified")
        # PR-AUC of a stratified dummy should be close to the class prevalence
        assert result["pr_auc"] == pytest.approx(0.1, abs=0.05)


class TestPlottingHelpers:

    def _data(self):
        rng = np.random.default_rng(7)
        y_true = rng.integers(0, 2, size=100)
        y_score = rng.random(100)
        return y_true, y_score

    def test_roc_curve_returns_figure(self):
        y_true, y_score = self._data()
        fig = plot_roc_curve(y_true, y_score, model_name="test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pr_curve_returns_figure(self):
        y_true, y_score = self._data()
        fig = plot_pr_curve(y_true, y_score, model_name="test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_calibration_curve_returns_figure(self):
        y_true, y_score = self._data()
        fig = plot_calibration_curve(y_true, y_score, model_name="test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
