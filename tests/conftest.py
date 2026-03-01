"""
Shared fixtures for all tests.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from components.config import DataEngineeringConfig


# ── Paths ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def config_path() -> Path:
    return Path("configs/run_config.yaml")


# ── Tiny synthetic dataframe that mirrors the real feature schema ──────────────
NUM_FEATURES = ["num_a", "num_b"]
CAT_FEATURES = ["cat_a"]


@pytest.fixture
def small_df() -> pd.DataFrame:
    """
    Minimal dataframe with two numerical and one categorical column
    plus a binary target.  Small enough to run fast in every test.
    """
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "num_a": rng.standard_normal(n),
            "num_b": rng.standard_normal(n),
            "cat_a": rng.choice(["A", "B", "C"], size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )


@pytest.fixture
def small_df_with_nans(small_df) -> pd.DataFrame:
    df = small_df.copy()
    df.loc[0, "num_a"] = np.nan
    df.loc[5, "cat_a"] = np.nan
    return df


@pytest.fixture
def X_y(small_df) -> tuple:
    X = small_df.drop(columns=["target"]).select_dtypes(include="number").to_numpy()
    y = small_df["target"].to_numpy()
    return X, y


# ── DataEngineeringConfig that matches small_df schema ────────────────────────
@pytest.fixture
def de_config() -> DataEngineeringConfig:
    return DataEngineeringConfig(
        numerical_features=NUM_FEATURES,
        categorical_features=CAT_FEATURES,
        numerical_imputer_strat="mean",
        categorical_imputer_strat="most_frequent",
        scaler="standard",
        remainder="drop",
    )
