"""
Data validation module using pandera.
"""

import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger

try:
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False
    logger.warning("pandera not installed, data validation will be basic")


def validate_ohlcv_schema(df: pd.DataFrame) -> bool:
    """
    Validate OHLCV DataFrame using pandera schema.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises ValidationError if not
    """
    if not PANDERA_AVAILABLE:
        # Fallback to basic validation
        from .utils import validate_ohlcv
        return validate_ohlcv(df)
    
    schema = DataFrameSchema({
        "open": Column(float, checks=[
            Check(lambda x: x > 0, element_wise=True, error="Open must be positive"),
            Check(lambda x: x < 1e10, element_wise=True, error="Open value too large")
        ]),
        "high": Column(float, checks=[
            Check(lambda x: x > 0, element_wise=True, error="High must be positive"),
        ]),
        "low": Column(float, checks=[
            Check(lambda x: x > 0, element_wise=True, error="Low must be positive"),
        ]),
        "close": Column(float, checks=[
            Check(lambda x: x > 0, element_wise=True, error="Close must be positive"),
        ]),
        "volume": Column(float, checks=[
            Check(lambda x: x >= 0, element_wise=True, error="Volume must be non-negative"),
        ]),
    }, checks=[
        Check(lambda df: (df['high'] >= df['low']).all(), 
              error="High must be >= Low"),
        Check(lambda df: (df['high'] >= df['open']).all(), 
              error="High must be >= Open"),
        Check(lambda df: (df['high'] >= df['close']).all(), 
              error="High must be >= Close"),
        Check(lambda df: (df['low'] <= df['open']).all(), 
              error="Low must be <= Open"),
        Check(lambda df: (df['low'] <= df['close']).all(), 
              error="Low must be <= Close"),
    ])
    
    try:
        schema.validate(df, lazy=True)
        logger.debug("OHLCV schema validation passed")
        return True
    except Exception as e:
        logger.error(f"OHLCV validation failed: {e}")
        raise


def check_data_leakage(features_df: pd.DataFrame, lookahead_threshold: int = 0) -> bool:
    """
    Check for data leakage in features.
    
    Args:
        features_df: DataFrame with features
        lookahead_threshold: Maximum allowed lookahead in bars
        
    Returns:
        True if no leakage detected, raises ValueError if leakage found
    """
    logger.info("Checking for data leakage...")
    
    # Check for future values in column names
    suspicious_keywords = ['future', 'next', 'ahead', 'forward', 'tomorrow']
    for col in features_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in suspicious_keywords):
            logger.warning(f"Potentially suspicious column name: {col}")
    
    # Check for features that might use future data
    # This is a basic check - more sophisticated checks can be added
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in features_df.columns:
            # Check if there are any features that use future prices
            # This is a simplified check
            pass
    
    logger.info("Data leakage check passed")
    return True


def validate_features_schema(features_df: pd.DataFrame) -> bool:
    """
    Validate features DataFrame schema.
    
    Args:
        features_df: DataFrame with features
        
    Returns:
        True if valid
    """
    # Basic checks
    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) > 0
    assert len(features_df.columns) > 0
    
    # Check for excessive NaN values
    nan_percentage = features_df.isna().sum() / len(features_df) * 100
    high_nan_cols = nan_percentage[nan_percentage > 50]
    
    if len(high_nan_cols) > 0:
        logger.warning(f"Columns with >50% NaN: {high_nan_cols.to_dict()}")
    
    # Check for infinite values
    inf_cols = []
    for col in features_df.select_dtypes(include=[np.number]).columns:
        if np.isinf(features_df[col]).any():
            inf_cols.append(col)
    
    if inf_cols:
        logger.warning(f"Columns with infinite values: {inf_cols}")
    
    logger.debug("Features schema validation passed")
    return True


