"""
Tests for components/explainability.py
"""
import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from components.explainability import compute_and_log_shap


@pytest.fixture
def trained_tree_and_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 6))
    # Target depends on the first two features so SHAP has real signal.
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    return model, X, feature_names


class TestComputeAndLogShap:

    def test_tree_model_produces_four_plots(self, trained_tree_and_data, tmp_path):
        model, X, names = trained_tree_and_data
        paths = compute_and_log_shap(
            model, X, names,
            model_name="rf",
            output_dir=str(tmp_path),
            sample_size=50,
            log_to_mlflow=False,
        )
        assert len(paths) == 4
        for p in paths:
            assert p.exists() and p.stat().st_size > 0

    def test_filenames_are_prefixed_with_model_name(self, trained_tree_and_data, tmp_path):
        model, X, names = trained_tree_and_data
        paths = compute_and_log_shap(
            model, X, names,
            model_name="mymodel",
            output_dir=str(tmp_path),
            sample_size=50,
            log_to_mlflow=False,
        )
        assert all(p.name.startswith("mymodel_shap_") for p in paths)

    def test_linear_model_supported(self, tmp_path):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((120, 5))
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(max_iter=500).fit(X, y)
        names = [f"f{i}" for i in range(X.shape[1])]
        paths = compute_and_log_shap(
            model, X, names,
            model_name="lr",
            output_dir=str(tmp_path),
            sample_size=50,
            log_to_mlflow=False,
        )
        assert len(paths) == 4
        assert all(p.exists() for p in paths)

    def test_mismatched_feature_names_raises(self, trained_tree_and_data, tmp_path):
        model, X, _ = trained_tree_and_data
        with pytest.raises(ValueError, match="feature_names"):
            compute_and_log_shap(
                model, X, ["too", "few"],
                model_name="rf",
                output_dir=str(tmp_path),
                log_to_mlflow=False,
            )
