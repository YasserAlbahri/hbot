"""
Tests for backtest module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlab.backtest import Backtester
from quantlab.data_loader import DataLoader


class TestBacktester:
    """Test Backtester class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.loader = DataLoader()
        self.df = self.loader._create_sample_data('EURUSD', '15m', 1000)
        self.backtester = Backtester(initial_capital=10000.0)
    
    def test_backtest_with_no_signals(self):
        """Test backtest with no signals (should not crash)."""
        signals = pd.Series(0, index=self.df.index)
        
        results = self.backtester.backtest_signals(
            self.df,
            signals,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )
        
        assert isinstance(results, pd.DataFrame)
        assert 'equity' in results.columns
        assert len(results) == len(self.df)
        assert results['equity'].iloc[0] == self.backtester.initial_capital
    
    def test_backtest_with_signals(self):
        """Test backtest with some signals."""
        signals = pd.Series(0, index=self.df.index)
        signals.iloc[10:20] = 1  # Buy signals
        signals.iloc[50:60] = -1  # Sell signals
        
        results = self.backtester.backtest_signals(
            self.df,
            signals,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )
        
        assert isinstance(results, pd.DataFrame)
        assert 'equity' in results.columns
        assert 'returns' in results.columns
        assert 'cumulative_returns' in results.columns
        assert 'drawdown' in results.columns
    
    def test_backtest_equity_curve(self):
        """Test that equity curve is calculated correctly."""
        signals = pd.Series(0, index=self.df.index)
        signals.iloc[100] = 1  # Single buy signal
        
        results = self.backtester.backtest_signals(
            self.df,
            signals,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )
        
        # Equity should start at initial capital
        assert results['equity'].iloc[0] == self.backtester.initial_capital
        
        # Equity should be non-negative
        assert (results['equity'] >= 0).all()
    
    def test_backtest_drawdown(self):
        """Test drawdown calculation."""
        signals = pd.Series(0, index=self.df.index)
        signals.iloc[100] = 1
        
        results = self.backtester.backtest_signals(
            self.df,
            signals,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )
        
        # Drawdown should be <= 0 (negative or zero)
        assert (results['drawdown'] <= 0).all()
        
        # Peak should be >= equity
        assert (results['peak'] >= results['equity']).all()
    
    def test_backtest_probabilities(self):
        """Test backtesting with probabilities."""
        # Create dummy probabilities (3 classes: -1, 0, 1)
        n_samples = len(self.df)
        probabilities = np.random.rand(n_samples, 3)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        results = self.backtester.backtest_probabilities(
            self.df,
            probabilities,
            threshold=0.6
        )
        
        assert isinstance(results, pd.DataFrame)
        assert 'equity' in results.columns


