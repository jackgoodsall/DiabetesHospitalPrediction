"""
Tests for hyperparameter_tuner.py
"""
import pytest
import numpy as np

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

from hyperparameter_tuner import tune, _SEARCH_SPACES, _make_objective


@pytest.fixture
def tiny_dataset():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 5))
    y = rng.integers(0, 2, size=120)
    return X, y


class TestSearchSpaces:

    def test_all_models_have_search_space(self):
        from model_builder import __MODELS__
        for name in __MODELS__:
            assert name in _SEARCH_SPACES, f"No search space for '{name}'"

    @pytest.mark.parametrize("model_name", list(_SEARCH_SPACES.keys()))
    def test_space_returns_dict(self, model_name):
        study = optuna.create_study()
        trial = study.ask()
        params = _SEARCH_SPACES[model_name](trial)
        assert isinstance(params, dict)
        assert len(params) > 0


class TestMakeObjective:

    def test_objective_returns_float(self, tiny_dataset):
        X, y = tiny_dataset
        objective = _make_objective("random_forest", X, y, None, 2, 0, "pr_auc")
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        score = objective(trial)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_objective_with_groups(self, tiny_dataset):
        X, y = tiny_dataset
        groups = np.repeat(np.arange(40), 3)  # 40 patients, 3 encounters each
        objective = _make_objective("random_forest", X, y, groups, 2, 0, "pr_auc")
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        score = objective(trial)
        assert isinstance(score, float)

    def test_brier_score_objective(self, tiny_dataset):
        X, y = tiny_dataset
        objective = _make_objective("logistic_regression", X, y, None, 2, 0, "brier_score")
        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        score = objective(trial)
        assert 0.0 <= score <= 1.0

    def test_unknown_metric_raises(self, tiny_dataset):
        X, y = tiny_dataset
        with pytest.raises(KeyError):
            _make_objective("random_forest", X, y, None, 2, 0, "not_a_metric")


class TestTune:

    def test_returns_dict_of_params(self, tiny_dataset):
        X, y = tiny_dataset
        params = tune(
            "random_forest", X, y,
            n_trials=3, metric="pr_auc", direction="maximize", cv_folds=2, seed=0,
        )
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_returned_params_build_valid_model(self, tiny_dataset):
        from model_builder import build_estimator
        X, y = tiny_dataset
        params = tune(
            "logistic_regression", X, y,
            n_trials=3, metric="pr_auc", direction="maximize", cv_folds=2, seed=1,
        )
        model = build_estimator("logistic_regression", **params)
        model.fit(X, y)
        preds = model.predict_proba(X)
        assert preds.shape == (len(y), 2)

    def test_unknown_model_raises(self, tiny_dataset):
        X, y = tiny_dataset
        with pytest.raises(ValueError, match="No search space"):
            tune("not_a_model", X, y, n_trials=1, cv_folds=2, seed=0)

    def test_timeout_respected(self, tiny_dataset):
        """Timeout should cut the study short without raising."""
        X, y = tiny_dataset
        params = tune(
            "random_forest", X, y,
            n_trials=9999, metric="pr_auc", direction="maximize",
            cv_folds=2, seed=0, timeout=2,
        )
        assert isinstance(params, dict)


class TestTuningConfig:

    def test_valid_config(self):
        from components.config import TuningConfig
        cfg = TuningConfig(enabled=True, n_trials=20, metric="pr_auc", direction="maximize")
        assert cfg.enabled is True
        assert cfg.n_trials == 20

    def test_defaults(self):
        from components.config import TuningConfig
        cfg = TuningConfig()
        assert cfg.enabled is False
        assert cfg.metric == "pr_auc"
        assert cfg.direction == "maximize"

    def test_invalid_metric_raises(self):
        from components.config import TuningConfig
        with pytest.raises(Exception):
            TuningConfig(metric="not_valid")

    def test_invalid_direction_raises(self):
        from components.config import TuningConfig
        with pytest.raises(Exception):
            TuningConfig(direction="sideways")
