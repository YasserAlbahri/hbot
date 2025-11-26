"""
Data loading and preprocessing module.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from loguru import logger
from .utils import validate_ohlcv, ensure_directory, get_project_root
from .data_validation import validate_ohlcv_schema


class DataLoader:
    """
    Load and preprocess OHLCV data.
    """
    
    def __init__(self, data_dir: Optional[str] = None, timezone: str = "UTC"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing raw data
            timezone: Timezone for timestamps
        """
        self.project_root = get_project_root()
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "data" / "raw"
        self.timezone = timezone
        ensure_directory(str(self.data_dir))
    
    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., '1h', '15m')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Try CSV file first
        csv_file = self.data_dir / f"{symbol}_{timeframe}.csv"
        
        if csv_file.exists():
            logger.info(f"Loading data from CSV: {csv_file}")
            df = pd.read_csv(
                csv_file,
                parse_dates=['timestamp'],
                index_col='timestamp'
            )
        else:
            logger.warning(f"CSV file not found: {csv_file}. Creating sample data.")
            df = self._create_sample_data(symbol, timeframe)
        
        # Ensure timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize(self.timezone)
        else:
            df.index = df.index.tz_convert(self.timezone)
        
        # Filter by date range
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date, tz=self.timezone)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date, tz=self.timezone)]
        
        # Standardize column names (lowercase)
        df.columns = df.columns.str.lower()
        
        # Validate with schema
        try:
            validate_ohlcv_schema(df)
        except Exception:
            # Fallback to basic validation
            validate_ohlcv(df)
        
        # Clean data
        df = self._clean_data(df)
        
        logger.info(f"Loaded {len(df)} rows for {symbol} {timeframe}")
        return df
    
    def _create_sample_data(self, symbol: str, timeframe: str, n_bars: int = 10000) -> pd.DataFrame:
        """
        Create sample OHLCV data for testing.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            n_bars: Number of bars to generate
            
        Returns:
            Sample DataFrame
        """
        logger.info(f"Creating sample data for {symbol} {timeframe}")
        
        # Generate timestamps
        freq_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        freq = freq_map.get(timeframe, '15min')
        timestamps = pd.date_range(
            end=pd.Timestamp.now(tz=self.timezone),
            periods=n_bars,
            freq=freq
        )
        
        # Generate random walk prices
        np.random.seed(42)
        base_price = 1.1000 if 'USD' in symbol else 1800.0
        returns = np.random.normal(0, 0.001, n_bars)
        prices = base_price * (1 + returns).cumprod()
        
        # Generate OHLCV
        data = {
            'open': prices * (1 + np.random.normal(0, 0.0001, n_bars)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0005, n_bars))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0005, n_bars))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_bars)
        }
        
        df = pd.DataFrame(data, index=timestamps)
        
        # Ensure OHLC logic
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean OHLCV data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        original_len = len(df)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by index
        df = df.sort_index()
        
        # Fill missing values (forward fill for prices, 0 for volume)
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].fillna(method='ffill')
        df['volume'] = df['volume'].fillna(0)
        
        # Remove outliers (using IQR method)
        for col in price_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Remove rows with zero volume (optional)
        df = df[df['volume'] > 0]
        
        removed = original_len - len(df)
        if removed > 0:
            logger.warning(f"Removed {removed} rows during cleaning")
        
        return df
    
    def resample_timeframe(
        self,
        df: pd.DataFrame,
        new_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to a new timeframe.
        
        Args:
            df: Source DataFrame with OHLCV data
            new_timeframe: Target timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            
        Returns:
            Resampled DataFrame with OHLCV data
            
        Raises:
            ValueError: If timeframe is not supported
        """
        """
        Resample OHLCV data to a new timeframe.
        
        Args:
            df: Source DataFrame
            new_timeframe: Target timeframe
            
        Returns:
            Resampled DataFrame
        """
        timeframe_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        freq = timeframe_map.get(new_timeframe)
        if not freq:
            raise ValueError(f"Unsupported timeframe: {new_timeframe}")
        
        # Resample OHLCV
        resampled = pd.DataFrame()
        resampled['open'] = df['open'].resample(freq).first()
        resampled['high'] = df['high'].resample(freq).max()
        resampled['low'] = df['low'].resample(freq).min()
        resampled['close'] = df['close'].resample(freq).last()
        resampled['volume'] = df['volume'].resample(freq).sum()
        
        # Remove NaN rows
        resampled = resampled.dropna()
        
        logger.info(f"Resampled to {new_timeframe}: {len(resampled)} bars")
        return resampled
    
    def load_multiple_timeframes(
        self,
        symbol: str,
        timeframes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> dict:
        """
        Load data for multiple timeframes.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        data = {}
        for tf in timeframes:
            data[tf] = self.load_ohlcv(symbol, tf, start_date, end_date)
        
        return data
    
    def save_processed(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        output_dir: Optional[str] = None
    ):
        """
        Save processed data.
        
        Args:
            df: DataFrame to save
            symbol: Trading symbol
            timeframe: Timeframe
            output_dir: Output directory (default: data/processed)
        """
        if output_dir is None:
            output_dir = self.project_root / "data" / "processed"
        else:
            output_dir = Path(output_dir)
        
        ensure_directory(str(output_dir))
        
        output_file = output_dir / f"{symbol}_{timeframe}.csv"
        df.to_csv(output_file)
        logger.info(f"Saved processed data to {output_file}")

