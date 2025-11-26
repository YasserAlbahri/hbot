"""
Labeling module - Triple-Barrier and other labeling methods.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from loguru import logger
from numba import jit


@jit(nopython=True)
def _triple_barrier_core(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    pt_multiplier: float,
    sl_multiplier: float,
    max_holding: int
) -> np.ndarray:
    """
    Core triple-barrier labeling logic (Numba-accelerated).
    
    Args:
        close_prices: Array of close prices
        high_prices: Array of high prices
        low_prices: Array of low prices
        pt_multiplier: Take profit multiplier (e.g., 2.0 for 2x ATR)
        sl_multiplier: Stop loss multiplier (e.g., 1.0 for 1x ATR)
        max_holding: Maximum holding period in bars
        
    Returns:
        Array of labels: +1 (TP), -1 (SL), 0 (Time barrier)
    """
    n = len(close_prices)
    labels = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        if i >= n - 1:
            labels[i] = 0
            continue
        
        entry_price = close_prices[i]
        pt_level = entry_price * (1 + pt_multiplier)
        sl_level = entry_price * (1 - sl_multiplier)
        
        # Check barriers
        touched_tp = False
        touched_sl = False
        barrier_idx = -1
        
        for j in range(i + 1, min(i + max_holding + 1, n)):
            if high_prices[j] >= pt_level:
                touched_tp = True
                barrier_idx = j
                break
            if low_prices[j] <= sl_level:
                touched_sl = True
                barrier_idx = j
                break
        
        if touched_tp:
            labels[i] = 1
        elif touched_sl:
            labels[i] = -1
        else:
            # Time barrier
            labels[i] = 0
    
    return labels


class Labeler:
    """
    Generate labels for machine learning using Triple-Barrier method.
    """
    
    def __init__(
        self,
        pt_sl: Tuple[float, float] = (2.0, 1.0),
        max_holding_bars: int = 20,
        volatility_measure: str = 'atr'
    ):
        """
        Initialize Labeler.
        
        Args:
            pt_sl: Tuple of (take_profit_multiplier, stop_loss_multiplier)
            max_holding_bars: Maximum holding period in bars
            volatility_measure: Volatility measure to use ('atr' or 'std')
        """
        self.pt_multiplier = pt_sl[0]
        self.sl_multiplier = pt_sl[1]
        self.max_holding_bars = max_holding_bars
        self.volatility_measure = volatility_measure
    
    def triple_barrier_labels(
        self,
        df: pd.DataFrame,
        pt_sl: Optional[Tuple[float, float]] = None,
        max_holding_bars: Optional[int] = None,
        use_atr: bool = True
    ) -> pd.Series:
        """
        Generate Triple-Barrier labels.
        
        Args:
            df: DataFrame with OHLCV and volatility measure (ATR or std)
            pt_sl: Override default PT/SL multipliers
            max_holding_bars: Override default max holding period
            use_atr: If True, use ATR-based barriers; else use fixed percentage
            
        Returns:
            Series of labels: +1 (TP hit), -1 (SL hit), 0 (Time barrier)
        """
        logger.info("Generating Triple-Barrier labels...")
        
        pt_mult = pt_sl[0] if pt_sl else self.pt_multiplier
        sl_mult = pt_sl[1] if pt_sl else self.sl_multiplier
        max_hold = max_holding_bars if max_holding_bars else self.max_holding_bars
        
        if use_atr:
            # Use ATR-based barriers
            if 'atr_14' not in df.columns:
                logger.warning("ATR not found, calculating...")
                from ta.volatility import AverageTrueRange
                atr = AverageTrueRange(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    window=14
                )
                df['atr_14'] = atr.average_true_range()
            
            # Calculate barriers based on ATR
            pt_levels = df['close'] + (df['atr_14'] * pt_mult)
            sl_levels = df['close'] - (df['atr_14'] * sl_mult)
        else:
            # Use fixed percentage barriers
            pt_levels = df['close'] * (1 + pt_mult / 100)
            sl_levels = df['close'] * (1 - sl_mult / 100)
        
        # Convert to numpy for numba
        close_arr = df['close'].values
        high_arr = df['high'].values
        low_arr = df['low'].values
        
        # Calculate relative barriers
        if use_atr:
            pt_rel = (pt_levels - df['close']) / df['close']
            sl_rel = (df['close'] - sl_levels) / df['close']
            pt_mult_adj = pt_rel.mean()
            sl_mult_adj = sl_rel.mean()
        else:
            pt_mult_adj = pt_mult / 100
            sl_mult_adj = sl_mult / 100
        
        # Generate labels
        labels = _triple_barrier_core(
            close_arr,
            high_arr,
            low_arr,
            pt_mult_adj,
            sl_mult_adj,
            max_hold
        )
        
        labels_series = pd.Series(labels, index=df.index, name='label')
        
        # Statistics
        tp_count = (labels_series == 1).sum()
        sl_count = (labels_series == -1).sum()
        time_count = (labels_series == 0).sum()
        
        logger.info(
            f"Labels generated: TP={tp_count} ({tp_count/len(labels_series)*100:.1f}%), "
            f"SL={sl_count} ({sl_count/len(labels_series)*100:.1f}%), "
            f"Time={time_count} ({time_count/len(labels_series)*100:.1f}%)"
        )
        
        return labels_series
    
    def fixed_horizon_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        threshold: float = 0.01
    ) -> pd.Series:
        """
        Generate fixed-horizon labels based on future return.
        
        Args:
            df: DataFrame with close prices
            horizon: Number of bars ahead to look
            threshold: Minimum return threshold to classify as +1 or -1
            
        Returns:
            Series of labels: +1 (up), -1 (down), 0 (neutral)
        """
        future_returns = df['close'].pct_change(horizon).shift(-horizon)
        
        labels = pd.Series(0, index=df.index)
        labels[future_returns > threshold] = 1
        labels[future_returns < -threshold] = -1
        
        return labels
    
    def trend_labels(
        self,
        df: pd.DataFrame,
        ma_period: int = 50,
        lookback: int = 5
    ) -> pd.Series:
        """
        Generate trend-based labels.
        
        Args:
            df: DataFrame with close prices
            ma_period: Moving average period
            lookback: Bars to look ahead
            
        Returns:
            Series of labels: +1 (uptrend), -1 (downtrend), 0 (neutral)
        """
        ma = df['close'].rolling(ma_period).mean()
        future_ma = ma.shift(-lookback)
        
        labels = pd.Series(0, index=df.index)
        labels[future_ma > ma] = 1
        labels[future_ma < ma] = -1
        
        return labels

