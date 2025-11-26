"""
Tests for labeling module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlab.labeling import Labeler
from quantlab.data_loader import DataLoader
from quantlab.feature_engineering import FeatureEngineer


class TestLabeling:
    """Test Labeler class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.loader = DataLoader()
        self.df = self.loader._create_sample_data('EURUSD', '15m', 1000)
        
        # Add ATR for triple barrier
        fe = FeatureEngineer({'indicators': {'atr': {'enabled': True, 'periods': [14]}}})
        self.df = fe.build_features(self.df)
        
        self.labeler = Labeler(pt_sl=(2.0, 1.0), max_holding_bars=20)
    
    def test_triple_barrier_labels_format(self):
        """Test that triple_barrier_labels returns correct format."""
        labels = self.labeler.triple_barrier_labels(self.df)
        
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(self.df)
        assert labels.name == 'label'
        
        # Should only contain -1, 0, or 1
        unique_values = labels.unique()
        assert all(val in [-1, 0, 1] for val in unique_values if pd.notna(val))
    
    def test_triple_barrier_labels_index(self):
        """Test that labels have same index as input."""
        labels = self.labeler.triple_barrier_labels(self.df)
        
        assert labels.index.equals(self.df.index)
    
    def test_fixed_horizon_labels(self):
        """Test fixed-horizon labeling."""
        labels = self.labeler.fixed_horizon_labels(self.df, horizon=5, threshold=0.01)
        
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(self.df)
        assert all(val in [-1, 0, 1] for val in labels.unique() if pd.notna(val))
    
    def test_trend_labels(self):
        """Test trend-based labeling."""
        labels = self.labeler.trend_labels(self.df, ma_period=50, lookback=5)
        
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(self.df)
        assert all(val in [-1, 0, 1] for val in labels.unique() if pd.notna(val))
    
    def test_triple_barrier_statistics(self):
        """Test that triple barrier produces reasonable statistics."""
        labels = self.labeler.triple_barrier_labels(self.df)
        
        tp_count = (labels == 1).sum()
        sl_count = (labels == -1).sum()
        time_count = (labels == 0).sum()
        
        # Should have some labels (not all NaN)
        total_labeled = tp_count + sl_count + time_count
        assert total_labeled > 0
        
        # Labels should be distributed (not all one type)
        if total_labeled > 10:
            assert tp_count > 0 or sl_count > 0


