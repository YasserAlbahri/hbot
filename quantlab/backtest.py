"""
Backtesting module for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger


class Backtester:
    """
    Backtest trading strategies.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0001
    ):
        """
        Initialize Backtester.
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade (0.001 = 0.1%)
            slippage: Slippage per trade (0.0001 = 0.01%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
    
    def backtest_signals(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        position_size: float = 0.1
    ) -> pd.DataFrame:
        """
        Backtest trading signals.
        
        Args:
            df: DataFrame with OHLCV data
            signals: Trading signals (+1 buy, -1 sell, 0 hold)
            stop_loss_pct: Stop loss percentage (0.02 = 2%)
            take_profit_pct: Take profit percentage (0.04 = 4%)
            position_size: Position size as fraction of capital (0.1 = 10%)
            
        Returns:
            DataFrame with backtest results
        """
        logger.info("Running backtest...")
        
        results = pd.DataFrame(index=df.index)
        results['signal'] = signals
        results['price'] = df['close']
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        entry_index = None
        equity = [capital]
        
        for i in range(len(results)):
            current_price = results['price'].iloc[i]
            signal = results['signal'].iloc[i]
            
            # Check for exit conditions
            if position != 0:
                if position == 1:  # Long position
                    pnl_pct = (current_price - entry_price) / entry_price
                    if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                        # Exit long
                        exit_value = capital * (1 + pnl_pct) * (1 - self.commission - self.slippage)
                        capital = exit_value
                        position = 0
                        entry_price = 0
                        entry_index = None
                elif position == -1:  # Short position
                    pnl_pct = (entry_price - current_price) / entry_price
                    if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                        # Exit short
                        exit_value = capital * (1 + pnl_pct) * (1 - self.commission - self.slippage)
                        capital = exit_value
                        position = 0
                        entry_price = 0
                        entry_index = None
            
            # Check for entry conditions
            if position == 0 and signal != 0:
                if signal == 1:  # Buy signal
                    position = 1
                    entry_price = current_price * (1 + self.slippage)
                    entry_index = i
                elif signal == -1:  # Sell signal
                    position = -1
                    entry_price = current_price * (1 - self.slippage)
                    entry_index = i
            
            # Calculate current equity
            if position != 0:
                if position == 1:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                current_equity = capital * (1 + pnl_pct)
            else:
                current_equity = capital
            
            equity.append(current_equity)
        
        # Store results
        results['equity'] = equity[1:]  # Remove initial capital
        results['returns'] = results['equity'].pct_change()
        results['cumulative_returns'] = (1 + results['returns']).cumprod() - 1
        
        # Calculate drawdown
        results['peak'] = results['equity'].expanding().max()
        results['drawdown'] = (results['equity'] - results['peak']) / results['peak']
        
        logger.info(f"Backtest completed - Final Equity: ${results['equity'].iloc[-1]:.2f}")
        
        return results
    
    def backtest_probabilities(
        self,
        df: pd.DataFrame,
        probabilities: np.ndarray,
        threshold: float = 0.6,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04
    ) -> pd.DataFrame:
        """
        Backtest using probability predictions.
        
        Args:
            df: DataFrame with OHLCV data
            probabilities: Probability array (n_samples, n_classes)
            threshold: Minimum probability threshold to enter trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            
        Returns:
            DataFrame with backtest results
        """
        # Convert probabilities to signals
        # Assuming 3 classes: [-1, 0, 1]
        pred_classes = np.argmax(probabilities, axis=1) - 1  # Convert to [-1, 0, 1]
        max_probs = np.max(probabilities, axis=1)
        
        # Only trade when confidence is high
        signals = pd.Series(0, index=df.index)
        signals[max_probs >= threshold] = pred_classes[max_probs >= threshold]
        
        return self.backtest_signals(df, signals, stop_loss_pct, take_profit_pct)

