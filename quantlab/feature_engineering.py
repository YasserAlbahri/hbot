"""
Feature engineering module - builds technical indicators and features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
import ta
from .data_validation import validate_features_schema, check_data_leakage
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.utils import dropna


class FeatureEngineer:
    """
    Build comprehensive features from OHLCV data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            config: Feature configuration dictionary
        """
        self.config = config or {}
        self.indicators_config = self.config.get('indicators', {})
        self.multi_tf_config = self.config.get('multi_timeframe', {})
        self.time_features_config = self.config.get('time_features', {})
        self.sessions_config = self.config.get('sessions', {})
    
    def build_features(
        self,
        df: pd.DataFrame,
        multi_timeframe_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Build comprehensive features from OHLCV data.
        
        Args:
            df: Base timeframe OHLCV DataFrame
            multi_timeframe_data: Dictionary of other timeframes for multi-TF features
            
        Returns:
            DataFrame with features added
        """
        logger.info("Building features...")
        features_df = df.copy()
        
        # Technical Indicators
        if self.indicators_config.get('rsi', {}).get('enabled', True):
            features_df = self._add_rsi(features_df)
        
        if self.indicators_config.get('macd', {}).get('enabled', True):
            features_df = self._add_macd(features_df)
        
        if self.indicators_config.get('atr', {}).get('enabled', True):
            features_df = self._add_atr(features_df)
        
        if self.indicators_config.get('bollinger_bands', {}).get('enabled', True):
            features_df = self._add_bollinger_bands(features_df)
        
        if self.indicators_config.get('stochastic', {}).get('enabled', True):
            features_df = self._add_stochastic(features_df)
        
        if self.indicators_config.get('ema', {}).get('enabled', True):
            features_df = self._add_ema(features_df)
        
        if self.indicators_config.get('sma', {}).get('enabled', True):
            features_df = self._add_sma(features_df)
        
        # Candlestick Patterns
        if self.config.get('candlestick', {}).get('enabled', True):
            features_df = self._add_candlestick_features(features_df)
        
        # Support & Resistance
        if self.config.get('support_resistance', {}).get('enabled', True):
            features_df = self._add_support_resistance(features_df)
        
        # Multi-timeframe Features
        if multi_timeframe_data and self.multi_tf_config.get('enabled', True):
            features_df = self._add_multi_timeframe_features(features_df, multi_timeframe_data)
        
        # Time-based Features
        if self.time_features_config.get('enabled', True):
            features_df = self._add_time_features(features_df)
        
        # Session Features
        if self.sessions_config.get('enabled', True):
            features_df = self._add_session_features(features_df)
        
        # Price-based Features
        features_df = self._add_price_features(features_df)
        
        # Volume Features
        features_df = self._add_volume_features(features_df)
        
        # Validate features
        validate_features_schema(features_df)
        check_data_leakage(features_df)
        
        logger.info(f"Built {len(features_df.columns)} features")
        return features_df
    
    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI indicators."""
        periods = self.indicators_config.get('rsi', {}).get('periods', [14])
        for period in periods:
            rsi = RSIIndicator(close=df['close'], window=period)
            df[f'rsi_{period}'] = rsi.rsi()
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator."""
        config = self.indicators_config.get('macd', {})
        macd = MACD(
            close=df['close'],
            window_fast=config.get('fast', 12),
            window_slow=config.get('slow', 26),
            window_sign=config.get('signal', 9)
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        return df
    
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ATR indicators."""
        periods = self.indicators_config.get('atr', {}).get('periods', [14])
        for period in periods:
            atr = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=period
            )
            df[f'atr_{period}'] = atr.average_true_range()
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands."""
        config = self.indicators_config.get('bollinger_bands', {})
        bb = BollingerBands(
            close=df['close'],
            window=config.get('period', 20),
            window_dev=config.get('std_dev', 2)
        )
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        return df
    
    def _add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        config = self.indicators_config.get('stochastic', {})
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=config.get('k_period', 14),
            smooth_window=config.get('d_period', 3)
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        return df
    
    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA indicators."""
        periods = self.indicators_config.get('ema', {}).get('periods', [9, 21, 50])
        for period in periods:
            ema = EMAIndicator(close=df['close'], window=period)
            df[f'ema_{period}'] = ema.ema_indicator()
        return df
    
    def _add_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA indicators."""
        periods = self.indicators_config.get('sma', {}).get('periods', [20, 50, 100])
        for period in periods:
            sma = SMAIndicator(close=df['close'], window=period)
            df[f'sma_{period}'] = sma.sma_indicator()
        return df
    
    def _add_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features."""
        # Body, shadows
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Body ratio
        df['body_ratio'] = df['body'] / (df['total_range'] + 1e-10)
        
        # Doji (body < 10% of range)
        df['is_doji'] = (df['body_ratio'] < 0.1).astype(int)
        
        # Hammer (long lower shadow, small body)
        df['is_hammer'] = (
            (df['lower_shadow'] > 2 * df['body']) &
            (df['upper_shadow'] < df['body'])
        ).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = (
            (df['close'] > df['open']) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        ).astype(int)
        
        df['bearish_engulfing'] = (
            (df['close'] < df['open']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1))
        ).astype(int)
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support and resistance levels."""
        config = self.config.get('support_resistance', {})
        lookback = config.get('lookback', 20)
        
        # Pivot highs and lows
        df['pivot_high'] = (
            (df['high'] == df['high'].rolling(window=lookback, center=True).max()) &
            (df['high'] >= df['high'].shift(1)) &
            (df['high'] >= df['high'].shift(-1))
        ).astype(int)
        
        df['pivot_low'] = (
            (df['low'] == df['low'].rolling(window=lookback, center=True).min()) &
            (df['low'] <= df['low'].shift(1)) &
            (df['low'] <= df['low'].shift(-1))
        ).astype(int)
        
        # Distance to nearest support/resistance
        pivot_highs = df[df['pivot_high'] == 1]['high']
        pivot_lows = df[df['pivot_low'] == 1]['low']
        
        if len(pivot_highs) > 0:
            df['dist_to_resistance'] = df.apply(
                lambda row: (pivot_highs[pivot_highs.index <= row.name] - row['close']).min()
                if len(pivot_highs[pivot_highs.index <= row.name]) > 0 else np.nan,
                axis=1
            )
        else:
            df['dist_to_resistance'] = np.nan
        
        if len(pivot_lows) > 0:
            df['dist_to_support'] = df.apply(
                lambda row: (row['close'] - pivot_lows[pivot_lows.index <= row.name]).min()
                if len(pivot_lows[pivot_lows.index <= row.name]) > 0 else np.nan,
                axis=1
            )
        else:
            df['dist_to_support'] = np.nan
        
        return df
    
    def _add_multi_timeframe_features(
        self,
        df: pd.DataFrame,
        multi_tf_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Add features from multiple timeframes."""
        indicators = self.multi_tf_config.get('indicators', ['rsi', 'atr', 'ema_21'])
        timeframes = self.multi_tf_config.get('timeframes', [])
        
        for tf in timeframes:
            if tf not in multi_tf_data:
                continue
            
            tf_df = multi_tf_data[tf]
            
            # Resample to base timeframe
            tf_df_resampled = tf_df.resample(df.index.freq or '15min').last()
            
            for indicator in indicators:
                if indicator == 'rsi' and 'rsi_14' in tf_df_resampled.columns:
                    df[f'rsi_14_{tf}'] = tf_df_resampled['rsi_14']
                elif indicator == 'atr' and 'atr_14' in tf_df_resampled.columns:
                    df[f'atr_14_{tf}'] = tf_df_resampled['atr_14']
                elif indicator == 'ema_21' and 'ema_21' in tf_df_resampled.columns:
                    df[f'ema_21_{tf}'] = tf_df_resampled['ema_21']
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading session features."""
        # Convert to UTC if needed
        if df.index.tz is None:
            df_tz = df.index.tz_localize('UTC')
        else:
            df_tz = df.index.tz_convert('UTC')
        
        hour = df_tz.hour
        
        # London session (08:00-16:00 UTC)
        df['is_london_session'] = ((hour >= 8) & (hour < 16)).astype(int)
        
        # New York session (13:00-21:00 UTC)
        df['is_newyork_session'] = ((hour >= 13) & (hour < 21)).astype(int)
        
        # Asia session (00:00-08:00 UTC)
        df['is_asia_session'] = ((hour >= 0) & (hour < 8)).astype(int)
        
        # Overlap (London + NY: 13:00-16:00 UTC)
        df['is_overlap_session'] = ((hour >= 13) & (hour < 16)).astype(int)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['returns_1'] = df['returns'].shift(1)
        df['returns_2'] = df['returns'].shift(2)
        df['returns_3'] = df['returns'].shift(3)
        
        # Volatility
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # Price position in range
        df['price_position'] = (df['close'] - df['low'].rolling(20).min()) / (
            df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-10
        )
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving averages
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
        
        # Volume price trend
        df['vpt'] = (df['volume'] * df['returns']).cumsum()
        
        return df

