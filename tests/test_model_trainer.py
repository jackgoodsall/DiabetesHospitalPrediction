"""
Tests for model_trainer.py  (ModelTrainer class)
"""
import pytest
import numpy as np

from model_builder import build_estimator
from model_trainer import ModelTrainer


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def small_X_y():
    """100 samples, 5 features, balanced binary target."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    y = rng.integers(0, 2, size=100)
    return X, y


@pytest.fixture
def trainer():
    return ModelTrainer(seed=42)


# ── Initialisation ─────────────────────────────────────────────────────────────
class TestModelTrainerInit:

    def test_seed_stored(self):
        t = ModelTrainer(seed=123)
        assert t.seed == 123


# ── train_and_get_results ──────────────────────────────────────────────────────
class TestTrainAndGetResults:

    def test_returns_three_items(self, trainer, small_X_y):
        X, y = small_X_y
        model = build_estimator("random_forest", n_estimators=10)
        result = trainer.train_and_get_results(model, X, y)
        assert len(result) == 3

    def test_returned_model_can_predict(self, trainer, small_X_y):
        X, y = small_X_y
        model = build_estimator("random_forest", n_estimators=10)
        trained_model, _, _ = trainer.train_and_get_results(model, X, y)
        preds = trained_model.predict_proba(X)
        assert preds.shape == (len(X), 2)

    def test_oof_predictions_correct_length(self, trainer, small_X_y):
        X, y = small_X_y
        model = build_estimator("random_forest", n_estimators=10)
        _, _, oof_preds = trainer.train_and_get_results(model, X, y)
        assert len(oof_preds) == len(y)

    def test_oof_metrics_contains_all_keys(self, trainer, small_X_y):
        X, y = small_X_y
        model = build_estimator("random_forest", n_estimators=10)
        _, oof_metrics, _ = trainer.train_and_get_results(model, X, y)
        assert set(oof_metrics.keys()) == {"accuracy", "precision", "recall", "f1", "auc_roc"}

    def test_oof_metrics_values_between_zero_and_one(self, trainer, small_X_y):
        X, y = small_X_y
        model = build_estimator("random_forest", n_estimators=10)
        _, oof_metrics, _ = trainer.train_and_get_results(model, X, y)
        for val in oof_metrics.values():
            assert 0.0 <= val <= 1.0

    def test_final_model_is_refit_on_all_data(self, trainer, small_X_y):
        """After training, model should predict on entire training set without error."""
        X, y = small_X_y
        model = build_estimator("random_forest", n_estimators=10)
        trained_model, _, _ = trainer.train_and_get_results(model, X, y)
        # If refit failed, predict_proba would raise NotFittedError
        trained_model.predict_proba(X)

    def test_reproducible_with_same_seed(self, small_X_y):
        X, y = small_X_y
        t1 = ModelTrainer(seed=42)
        t2 = ModelTrainer(seed=42)
        m1 = build_estimator("random_forest", n_estimators=10, random_state=42)
        m2 = build_estimator("random_forest", n_estimators=10, random_state=42)

        _, metrics1, oof1 = t1.train_and_get_results(m1, X, y)
        _, metrics2, oof2 = t2.train_and_get_results(m2, X, y)

        np.testing.assert_array_almost_equal(oof1, oof2)
        assert metrics1 == metrics2

    def test_works_with_xgboost(self, trainer, small_X_y):
        X, y = small_X_y
        model = build_estimator(
            "xgboost",
            n_estimators=5,
            eval_metric="logloss",
            random_state=42,
        )
        trained_model, oof_metrics, oof_preds = trainer.train_and_get_results(model, X, y)
        assert len(oof_preds) == len(y)
        assert "f1" in oof_metrics
