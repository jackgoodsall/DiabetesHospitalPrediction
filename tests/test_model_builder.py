"""
Tests for model_builder.py
"""
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from model_builder import build_estimator, __MODELS__


class TestBuildEstimator:

    def test_builds_logistic_regression(self):
        model = build_estimator("logistic_regression")
        assert isinstance(model, LogisticRegression)

    def test_builds_random_forest(self):
        model = build_estimator("random_forest")
        assert isinstance(model, RandomForestClassifier)

    def test_builds_xgboost(self):
        model = build_estimator("xgboost")
        assert isinstance(model, XGBClassifier)

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_estimator("neural_network")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            build_estimator("")

    def test_kwargs_passed_to_model(self):
        model = build_estimator("random_forest", n_estimators=7)
        assert model.n_estimators == 7

    def test_multiple_kwargs_passed(self):
        model = build_estimator("logistic_regression", C=0.1, max_iter=200)
        assert model.C == 0.1
        assert model.max_iter == 200

    def test_name_is_case_insensitive(self):
        """build_estimator strips and lowercases name."""
        model = build_estimator("  XGBOOST  ")
        assert isinstance(model, XGBClassifier)

    def test_model_registry_contains_expected_keys(self):
        assert "logistic_regression" in __MODELS__
        assert "random_forest" in __MODELS__
        assert "xgboost" in __MODELS__

    def test_returned_model_has_fit_method(self):
        """Any returned estimator must be sklearn-compatible."""
        for name in __MODELS__:
            model = build_estimator(name)
            assert hasattr(model, "fit")
            assert hasattr(model, "predict_proba")
