"""
Tests for components/training_splits.py
"""
import pytest
import numpy as np
import pandas as pd

from components.training_splits import split_df, cross_validation_splits


# ── split_df ───────────────────────────────────────────────────────────────────
class TestSplitDf:

    def test_returns_two_dataframes(self, small_df):
        train, test = split_df(small_df, target="target", test_size=0.2)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_correct_split_size(self, small_df):
        n = len(small_df)
        _, test = split_df(small_df, target="target", test_size=0.2)
        assert len(test) == pytest.approx(n * 0.2, abs=1)

    def test_no_overlap_between_splits(self, small_df):
        train, test = split_df(small_df, target="target", test_size=0.2)
        train_idx = set(train.index)
        test_idx = set(test.index)
        assert train_idx.isdisjoint(test_idx)

    def test_all_rows_accounted_for(self, small_df):
        train, test = split_df(small_df, target="target", test_size=0.2)
        assert len(train) + len(test) == len(small_df)

    def test_columns_preserved(self, small_df):
        train, test = split_df(small_df, target="target", test_size=0.2)
        assert list(train.columns) == list(small_df.columns)
        assert list(test.columns) == list(small_df.columns)


# ── cross_validation_splits ────────────────────────────────────────────────────
class TestCrossValidationSplits:

    def test_yields_correct_number_of_folds(self):
        X = np.zeros((100, 5))
        splits = list(cross_validation_splits(X, n_splits=5, return_indices=True))
        assert len(splits) == 5

    def test_custom_n_splits(self):
        X = np.zeros((60, 3))
        splits = list(cross_validation_splits(X, n_splits=3, return_indices=True))
        assert len(splits) == 3

    def test_no_overlap_between_train_and_val(self):
        X = np.zeros((100, 5))
        for tr_idx, val_idx in cross_validation_splits(X, return_indices=True):
            assert len(set(tr_idx) & set(val_idx)) == 0

    def test_all_indices_covered_across_folds(self):
        """Every sample should appear in exactly one validation fold."""
        X = np.zeros((100, 5))
        all_val_indices = []
        for _, val_idx in cross_validation_splits(X, return_indices=True):
            all_val_indices.extend(val_idx.tolist())
        assert sorted(all_val_indices) == list(range(100))

    def test_return_arrays_when_indices_false(self):
        X = np.ones((50, 3))
        for X_train, X_val in cross_validation_splits(X, return_indices=False):
            assert isinstance(X_train, np.ndarray)
            assert isinstance(X_val, np.ndarray)

    def test_reproducible_with_same_random_state(self):
        X = np.zeros((50, 5))
        splits_a = list(cross_validation_splits(X, return_indices=True, random_state=42))
        splits_b = list(cross_validation_splits(X, return_indices=True, random_state=42))
        for (tr_a, va_a), (tr_b, va_b) in zip(splits_a, splits_b):
            np.testing.assert_array_equal(tr_a, tr_b)
            np.testing.assert_array_equal(va_a, va_b)

    def test_different_random_states_give_different_splits(self):
        X = np.zeros((100, 5))
        splits_a = list(cross_validation_splits(X, return_indices=True, random_state=0))
        splits_b = list(cross_validation_splits(X, return_indices=True, random_state=99))
        # At least one fold should differ
        any_different = any(
            not np.array_equal(va_a, va_b)
            for (_, va_a), (_, va_b) in zip(splits_a, splits_b)
        )
        assert any_different

    def test_unknown_strategy_raises_value_error(self):
        X = np.zeros((50, 3))
        with pytest.raises(ValueError, match="Unknown stratergy"):
            list(cross_validation_splits(X, stratergy="random"))
