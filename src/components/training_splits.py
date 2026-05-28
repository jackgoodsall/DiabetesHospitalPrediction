from __future__ import annotations
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedGroupKFold,
    GroupShuffleSplit,
    train_test_split,
)


def split_df(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    stratify: bool = True,
    group_col: Optional[str] = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train/test split a pandas dataframe.

    Args:
        target: name of the target column (used for stratification).
        test_size: fraction of rows (or groups) held out for the test set.
        stratify: when True, preserve the target class ratio across the split.
        group_col: when given, the split is group-aware — every row sharing a
            group value (e.g. all encounters for one ``patient_nbr``) is kept on
            the same side of the split. This prevents patient-level leakage,
            where the same patient appears in both train and test.
        random_state: seed for a reproducible split.

    Group + stratified splitting uses ``StratifiedGroupKFold`` and takes the
    first of ``round(1 / test_size)`` folds as the test set, giving an
    approximately ``test_size`` split that is both grouped and stratified.
    """
    if group_col is not None:
        if group_col not in df.columns:
            raise ValueError(f"group_col '{group_col}' not found in dataframe columns")

        y = df[target].to_numpy()
        groups = df[group_col].to_numpy()

        if stratify:
            n_splits = max(2, round(1 / test_size))
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            train_idx, test_idx = next(splitter.split(df, y, groups))
        else:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            train_idx, test_idx = next(splitter.split(df, y, groups))

        return df.iloc[train_idx], df.iloc[test_idx]

    stratify_arg = df[target].to_numpy() if stratify else None
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )
    return train_df, test_df


def cross_validation_splits(
    X: np.ndarray,
    y: np.ndarray = None,
    groups: np.ndarray = None,
    n_splits: int = 5,
    stratergy: str = "kfold",
    shuffle=True,
    return_indices: bool = False,
    random_state=None,
):
    """
    Yield CV splits as (train_idx, val_idx) or (train_arr, val_arr) depending
    on whether return_indices is True.

    Strategies:
        kfold            - plain KFold (no labels needed).
        stratified       - StratifiedKFold; preserves class ratio per fold (needs y).
        stratified_group - StratifiedGroupKFold; preserves class ratio while
                           keeping each group on one side of the fold (needs y
                           and groups). Use this to avoid patient-level leakage.
    """
    if stratergy not in {"kfold", "stratified", "stratified_group"}:
        raise ValueError(f"Unknown stratergy {stratergy}")

    # Split on index so the same indices work for both return modes.
    idx = np.arange(len(X))
    # StratifiedKFold/StratifiedGroupKFold only accept random_state when shuffling.
    rs = random_state if shuffle else None

    if stratergy == "kfold":
        splitter = KFold(n_splits, shuffle=shuffle, random_state=rs)
        split_iter = splitter.split(idx)
    elif stratergy == "stratified":
        if y is None:
            raise ValueError("stratergy='stratified' requires y")
        splitter = StratifiedKFold(n_splits, shuffle=shuffle, random_state=rs)
        split_iter = splitter.split(idx, y)
    else:  # stratified_group
        if y is None or groups is None:
            raise ValueError("stratergy='stratified_group' requires y and groups")
        splitter = StratifiedGroupKFold(n_splits, shuffle=shuffle, random_state=rs)
        split_iter = splitter.split(idx, y, groups)

    for tr_idx, val_idx in split_iter:
        if return_indices:
            yield tr_idx, val_idx
        else:
            yield X[tr_idx], X[val_idx]

