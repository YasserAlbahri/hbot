"""
Time-series cross-validation with purging and embargo.
"""

import pandas as pd
import numpy as np
from typing import Iterator, Tuple, List
from loguru import logger


class TimeSeriesSplit:
    """
    Time-series cross-validation splitter.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = None,
        gap: int = 0
    ):
        """
        Initialize TimeSeriesSplit.
        
        Args:
            n_splits: Number of splits
            test_size: Size of test set (if None, calculated automatically)
            gap: Gap between train and test (embargo period)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.
        
        Args:
            X: DataFrame with time index
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        for i in range(self.n_splits):
            # Calculate split points
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                test_end = n_samples
            
            train_end = test_start - self.gap
            
            if train_end <= 0:
                continue
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices


class PurgedKFold:
    """
    Purged K-Fold for time-series with purging and embargo.
    
    Based on López de Prado's "Advances in Financial Machine Learning"
    """
    
    def __init__(
        self,
        n_splits: int = 3,
        t1: pd.Series = None,
        pct_embargo: float = 0.01
    ):
        """
        Initialize PurgedKFold.
        
        Args:
            n_splits: Number of splits
            t1: Series with end times for each sample (for purging)
            pct_embargo: Percentage of samples for embargo period
        """
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
    
    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged train/test splits.
        
        Args:
            X: DataFrame with time index
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if self.t1 is None:
            # Use index as t1 if not provided
            self.t1 = pd.Series(X.index, index=X.index)
        
        indices = np.arange(len(X))
        mbrg = int(len(X) * self.pct_embargo)
        test_starts = [
            (i[0], i[-1] + 1) for i in np.array_split(np.arange(len(X)), self.n_splits)
        ]
        
        for test_start, test_end in test_starts:
            test_indices = indices[test_start:test_end]
            
            # Purge: remove train samples that overlap with test labels
            max_t1_idx = self.t1.iloc[test_indices].max()
            train_indices = indices[self.t1 <= max_t1_idx]
            
            # Embargo: remove samples after test period
            min_t0_idx = X.index[test_indices[0]]
            train_indices = train_indices[X.index[train_indices] < min_t0_idx]
            
            # Remove embargo period
            if mbrg > 0:
                train_indices = train_indices[:-mbrg]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


class WalkForwardSplit:
    """
    Walk-forward analysis splitter.
    """
    
    def __init__(
        self,
        train_window: int,
        test_window: int,
        step: int = None
    ):
        """
        Initialize WalkForwardSplit.
        
        Args:
            train_window: Size of training window
            test_window: Size of test window
            step: Step size (defaults to test_window)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step = step or test_window
    
    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward splits.
        
        Args:
            X: DataFrame with time index
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        start = 0
        
        while start + self.train_window + self.test_window <= n_samples:
            train_end = start + self.train_window
            test_end = train_end + self.test_window
            
            train_indices = np.arange(start, train_end)
            test_indices = np.arange(train_end, test_end)
            
            yield train_indices, test_indices
            
            start += self.step


