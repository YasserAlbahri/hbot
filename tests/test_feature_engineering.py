"""
Tests for feature_engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlab.feature_engineering import FeatureEngineer
from quantlab.data_loader import DataLoader


class TestFeatureEngineering:
    """Test FeatureEngineer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.loader = DataLoader()
        self.df = self.loader._create_sample_data('EURUSD', '15m', 1000)
        
        # Simple config
        self.config = {
            'indicators': {
                'rsi': {'enabled': True, 'periods': [14]},
                'macd': {'enabled': True, 'fast': 12, 'slow': 26, 'signal': 9},
                'atr': {'enabled': True, 'periods': [14]},
                'ema': {'enabled': True, 'periods': [21]},
                'sma': {'enabled': True, 'periods': [20]}
            },
            'candlestick': {'enabled': True},
            'time_features': {'enabled': True},
            'sessions': {'enabled': True},
            'multi_timeframe': {'enabled': False}
        }
        
        self.fe = FeatureEngineer(self.config)
    
    def test_build_features_returns_dataframe(self):
        """Test that build_features returns DataFrame."""
        features_df = self.fe.build_features(self.df)
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == len(self.df)
    
    def test_features_added(self):
        """Test that features are actually added."""
        features_df = self.fe.build_features(self.df)
        
        # Should have more columns than original
        assert len(features_df.columns) > len(self.df.columns)
        
        # Check specific features
        assert 'rsi_14' in features_df.columns
        assert 'atr_14' in features_df.columns
        assert 'ema_21' in features_df.columns
        assert 'sma_20' in features_df.columns
    
    def test_candlestick_features(self):
        """Test candlestick features."""
        features_df = self.fe.build_features(self.df)
        
        assert 'body' in features_df.columns
        assert 'upper_shadow' in features_df.columns
        assert 'lower_shadow' in features_df.columns
        assert 'is_doji' in features_df.columns
    
    def test_time_features(self):
        """Test time-based features."""
        features_df = self.fe.build_features(self.df)
        
        assert 'hour' in features_df.columns
        assert 'day_of_week' in features_df.columns
        assert 'hour_sin' in features_df.columns
        assert 'hour_cos' in features_df.columns
    
    def test_session_features(self):
        """Test session features."""
        features_df = self.fe.build_features(self.df)
        
        assert 'is_london_session' in features_df.columns
        assert 'is_newyork_session' in features_df.columns
        assert 'is_asia_session' in features_df.columns
    
    def test_no_nan_in_original_columns(self):
        """Test that original OHLCV columns don't have NaN after feature engineering."""
        features_df = self.fe.build_features(self.df)
        
        # Original columns should not have NaN (except maybe at start due to indicators)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            if col in features_df.columns:
                # Allow some NaN at the beginning due to rolling windows
                nan_count = features_df[col].isna().sum()
                assert nan_count < len(features_df) * 0.1  # Less than 10% NaN
    
    def test_price_features(self):
        """Test price-based features."""
        features_df = self.fe.build_features(self.df)
        
        assert 'returns' in features_df.columns
        assert 'volatility_5' in features_df.columns
        assert 'price_position' in features_df.columns
    
    def test_volume_features(self):
        """Test volume-based features."""
        features_df = self.fe.build_features(self.df)
        
        assert 'volume_sma_20' in features_df.columns
        assert 'volume_ratio' in features_df.columns

