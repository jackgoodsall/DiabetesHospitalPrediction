from __future__ import annotations
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def split_df(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    startify: bool = False,
    group_cols=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to train test split a pandas dataframe.

    Can extend to stratified based on target if wanted.
    """
    train_df, test_df = train_test_split(df, test_size=test_size)
    return train_df, test_df


def cross_validation_splits(
    X: np.ndarray,
    target: str = None,
    n_splits: int = 5,
    stratergy: str = "kfold",
    shuffle=True,
    return_indices: bool = False,
    random_state=None,
):
    """
    Yield CV splits as (train_idx, val_idx) or (train_df, val_df) depending
    on whether return_indices is true or not.

    1  available - kfold.
    """
    if stratergy not in {"kfold", "stratified"}:
        raise ValueError(f"Unknown stratergy {stratergy}")
    # Split on index
    idx = np.arange(len(X))

    # K fold
    if stratergy == "kfold":
        splitter = KFold(n_splits, shuffle=shuffle, random_state=random_state)
        for tr_idx, val_idx in splitter.split(idx):
            if return_indices:
                yield tr_idx, val_idx
            else:
                yield X[tr_idx], X[val_idx]

