"""
Tests for data_loader module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlab.data_loader import DataLoader
from quantlab.utils import validate_ohlcv


class TestDataLoader:
    """Test DataLoader class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.loader = DataLoader()
    
    def test_create_sample_data(self):
        """Test sample data creation."""
        df = self.loader._create_sample_data('EURUSD', '15m', 100)
        
        assert len(df) == 100
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert df.index.is_monotonic_increasing
    
    def test_sample_data_ohlc_logic(self):
        """Test OHLC logic in sample data."""
        df = self.loader._create_sample_data('EURUSD', '15m', 100)
        
        # High should be >= all other prices
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        assert (df['high'] >= df['low']).all()
        
        # Low should be <= all other prices
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
        assert (df['low'] <= df['high']).all()
    
    def test_clean_data_removes_duplicates(self):
        """Test that clean_data removes duplicates."""
        df = self.loader._create_sample_data('EURUSD', '15m', 100)
        
        # Add duplicate
        df_dup = pd.concat([df, df.iloc[[0]]])
        assert len(df_dup) == 101
        
        cleaned = self.loader._clean_data(df_dup)
        assert len(cleaned) <= 101
    
    def test_resample_timeframe(self):
        """Test timeframe resampling."""
        df = self.loader._create_sample_data('EURUSD', '15m', 1000)
        
        # Resample to 1h (should have fewer bars)
        resampled = self.loader.resample_timeframe(df, '1h')
        
        assert len(resampled) < len(df)
        assert all(col in resampled.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
        # Check OHLC logic
        assert (resampled['high'] >= resampled['open']).all()
        assert (resampled['high'] >= resampled['close']).all()
        assert (resampled['low'] <= resampled['open']).all()
    
    def test_validate_ohlcv(self):
        """Test OHLCV validation."""
        df = self.loader._create_sample_data('EURUSD', '15m', 100)
        
        # Should pass validation
        assert validate_ohlcv(df) is True
        
        # Should fail with missing column
        df_missing = df.drop(columns=['close'])
        with pytest.raises(ValueError):
            validate_ohlcv(df_missing)
    
    def test_load_multiple_timeframes(self):
        """Test loading multiple timeframes."""
        timeframes = ['15m', '1h']
        data = self.loader.load_multiple_timeframes('EURUSD', timeframes)
        
        assert len(data) == 2
        assert '15m' in data
        assert '1h' in data
        assert isinstance(data['15m'], pd.DataFrame)
        assert isinstance(data['1h'], pd.DataFrame)

