from __future__ import annotations
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split


def split_df(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    startify: bool = False,
    group_cols=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ### Helper function to split dataframe into 2 dataframes from
    ### main dataframe.
    train_df, val_df = train_test_split(df, test_size=test_size)
    return train_df, val_df
