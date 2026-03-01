"""
Tests for components/data_engineering.py
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from components.data_engineering import DataEngineeringPipeLine
from components.config import DataEngineeringConfig


# ── Helpers ────────────────────────────────────────────────────────────────────
def make_pipeline(scaler="standard") -> DataEngineeringPipeLine:
    config = DataEngineeringConfig(
        numerical_features=["num_a", "num_b"],
        categorical_features=["cat_a"],
        numerical_imputer_strat="mean",
        categorical_imputer_strat="most_frequent",
        scaler=scaler,
        remainder="drop",
    )
    return DataEngineeringPipeLine(config)


# ── Instantiation ──────────────────────────────────────────────────────────────
class TestDataEngineeringPipelineInit:

    def test_is_not_fit_after_init(self, de_config):
        pipeline = DataEngineeringPipeLine(de_config)
        assert pipeline.is_fit is False

    def test_stores_config(self, de_config):
        pipeline = DataEngineeringPipeLine(de_config)
        assert pipeline.config is de_config


# ── Fit ────────────────────────────────────────────────────────────────────────
class TestDataEngineeringPipelineFit:

    def test_fit_sets_is_fit_true(self, small_df, de_config):
        pipeline = DataEngineeringPipeLine(de_config)
        pipeline.fit(small_df)
        assert pipeline.is_fit is True

    def test_fit_returns_self(self, small_df, de_config):
        pipeline = DataEngineeringPipeLine(de_config)
        result = pipeline.fit(small_df)
        assert result is pipeline

    def test_fit_raises_if_columns_missing(self, de_config):
        bad_df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        pipeline = DataEngineeringPipeLine(de_config)
        with pytest.raises(ValueError, match="Not all features are in the dataframe"):
            pipeline.fit(bad_df)


# ── Transform ──────────────────────────────────────────────────────────────────
class TestDataEngineeringPipelineTransform:

    def test_transform_raises_if_not_fit(self, small_df, de_config):
        pipeline = DataEngineeringPipeLine(de_config)
        with pytest.raises(NotFittedError):
            pipeline.transform(small_df)

    def test_transform_returns_dataframe(self, small_df, de_config):
        pipeline = DataEngineeringPipeLine(de_config)
        pipeline.fit(small_df)
        result = pipeline.transform(small_df)
        assert isinstance(result, pd.DataFrame)

    def test_transform_drops_remainder_columns(self, small_df, de_config):
        """'remainder=drop' means the target column is not in the output."""
        pipeline = DataEngineeringPipeLine(de_config)
        pipeline.fit(small_df)
        result = pipeline.transform(small_df)
        assert "target" not in result.columns

    def test_output_has_no_nulls_after_imputation(self, small_df_with_nans, de_config):
        pipeline = DataEngineeringPipeLine(de_config)
        pipeline.fit(small_df_with_nans)
        result = pipeline.transform(small_df_with_nans)
        assert result.isnull().sum().sum() == 0

    def test_no_data_leakage_from_test_set(self, small_df, de_config):
        """Imputer stats must come only from train split."""
        train = small_df.iloc[:40]
        test = small_df.iloc[40:]
        pipeline = DataEngineeringPipeLine(de_config)
        pipeline.fit(train)

        train_mean = train["num_a"].mean()
        fitted_mean = pipeline._pipeline.named_transformers_["num pipeline"][
            "num_imputer"
        ].statistics_[0]  # first numerical feature mean
        assert fitted_mean == pytest.approx(train_mean)

    def test_standard_scaler_produces_approx_zero_mean(self, small_df, de_config):
        pipeline = DataEngineeringPipeLine(de_config)
        result = pipeline.fit_transform(small_df)
        num_col = [c for c in result.columns if "num_a" in c][0]
        assert result[num_col].mean() == pytest.approx(0.0, abs=1e-9)

    def test_minmax_scaler_output_in_zero_one_range(self, small_df):
        pipeline = make_pipeline(scaler="minmax")
        result = pipeline.fit_transform(small_df)
        num_cols = [c for c in result.columns if "num_a" in c or "num_b" in c]
        for col in num_cols:
            assert result[col].min() >= -1e-9
            assert result[col].max() <= 1.0 + 1e-9


# ── fit_transform convenience method ──────────────────────────────────────────
class TestFitTransform:

    def test_fit_transform_equivalent_to_fit_then_transform(self, small_df, de_config):
        p1 = DataEngineeringPipeLine(de_config)
        out1 = p1.fit_transform(small_df)

        p2 = DataEngineeringPipeLine(de_config)
        p2.fit(small_df)
        out2 = p2.transform(small_df)

        pd.testing.assert_frame_equal(out1.reset_index(drop=True),
                                      out2.reset_index(drop=True))
