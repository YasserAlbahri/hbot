"""
Complete training pipeline: Data → Features → Labels → Train.
"""

import pandas as pd
import numpy as np
import mlflow
import subprocess
from typing import Dict, Optional, Tuple
from pathlib import Path
from loguru import logger

from ..data_loader import DataLoader
from ..feature_engineering import FeatureEngineer
from ..labeling import Labeler
from ..models import ModelTrainer
from ..backtest import Backtester
from ..evaluation import Evaluator
from ..utils import load_config, get_project_root
from ..cross_validation import WalkForwardSplit


class TrainingPipeline:
    """
    Complete pipeline for training trading models.
    """
    
    def __init__(
        self,
        data_config: Dict,
        features_config: Dict,
        model_config: Dict
    ):
        """
        Initialize TrainingPipeline.
        
        Args:
            data_config: Data configuration
            features_config: Features configuration
            model_config: Model configuration
        """
        self.data_config = data_config
        self.features_config = features_config
        self.model_config = model_config
        
        # Initialize components
        self.data_loader = DataLoader(
            data_dir=None,
            timezone=data_config.get('data', {}).get('timezone', 'UTC')
        )
        self.feature_engineer = FeatureEngineer(features_config)
        self.labeler = Labeler()
        self.model_trainer = ModelTrainer(model_config)
        self.backtester = Backtester()
        self.evaluator = Evaluator()
    
    def run(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model_type: str = 'xgboost'
    ) -> Dict:
        """
        Run complete training pipeline.
        
        Args:
            symbol: Trading symbol
            timeframe: Base timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            model_type: Model type ('xgboost' or 'lightgbm')
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Starting training pipeline for {symbol} {timeframe}")
        
        # Step 1: Load data
        logger.info("Step 1: Loading data...")
        df = self.data_loader.load_ohlcv(symbol, timeframe, start_date, end_date)
        
        # Step 2: Load multi-timeframe data if needed
        multi_tf_data = None
        if self.features_config.get('multi_timeframe', {}).get('enabled', False):
            logger.info("Step 2: Loading multi-timeframe data...")
            timeframes = self.features_config['multi_timeframe'].get('timeframes', [])
            multi_tf_data = {}
            for tf in timeframes:
                if tf != timeframe:
                    try:
                        multi_tf_data[tf] = self.data_loader.load_ohlcv(symbol, tf, start_date, end_date)
                    except:
                        logger.warning(f"Could not load {tf} data, resampling from {timeframe}")
                        multi_tf_data[tf] = self.data_loader.resample_timeframe(df, tf)
        
        # Step 3: Feature engineering
        logger.info("Step 3: Building features...")
        features_df = self.feature_engineer.build_features(df, multi_tf_data)
        
        # Step 4: Generate labels
        logger.info("Step 4: Generating labels...")
        labels = self.labeler.triple_barrier_labels(features_df)
        
        # Step 5: Prepare data for training
        logger.info("Step 5: Preparing training data...")
        # Remove NaN rows
        valid_mask = ~(features_df.isna().any(axis=1) | labels.isna())
        X = features_df[valid_mask].select_dtypes(include=[float, int])
        y = labels[valid_mask]
        
        # Remove infinite values
        inf_mask = ~(np.isinf(X).any(axis=1))
        X = X[inf_mask]
        y = y[inf_mask]
        
        # Train/Val/Test split (temporal)
        train_size = int(len(X) * (1 - self.model_config['training']['test_size'] - 
                                   self.model_config['training'].get('validation_size', 0.1)))
        val_size = int(len(X) * self.model_config['training'].get('validation_size', 0.1))
        
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X.iloc[train_size:train_size+val_size]
        y_val = y.iloc[train_size:train_size+val_size]
        X_test = X.iloc[train_size+val_size:]
        y_test = y.iloc[train_size+val_size:]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Step 6: Train model
        logger.info(f"Step 6: Training {model_type} model...")
        
        # Log configs and git info to MLflow before training
        if self.model_config.get('mlflow', {}).get('enabled', True):
            try:
                import mlflow
                from ..utils import get_project_root
                import subprocess
                
                # Log config files as artifacts
                config_dir = get_project_root() / 'configs'
                for config_file in ['data_config.yaml', 'features_config.yaml', 'model_config.yaml']:
                    config_path = config_dir / config_file
                    if config_path.exists():
                        mlflow.log_artifact(str(config_path), 'configs')
                
                # Log git commit
                try:
                    git_commit = subprocess.check_output(
                        ['git', 'rev-parse', 'HEAD'],
                        cwd=get_project_root(),
                        stderr=subprocess.DEVNULL
                    ).decode().strip()
                    mlflow.log_param('git_commit', git_commit)
                except:
                    pass
            except:
                pass
        
        if model_type == 'xgboost':
            model, train_metrics = self.model_trainer.train_xgboost(X_train, y_train, X_val, y_val)
        elif model_type == 'lightgbm':
            model, train_metrics = self.model_trainer.train_lightgbm(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Step 7: Backtest
        logger.info("Step 7: Running backtest...")
        test_df = df.iloc[train_size+val_size:].copy()
        test_features = features_df.iloc[train_size+val_size:].copy()
        
        # Get predictions
        test_X = test_features.select_dtypes(include=[float, int])
        test_X = test_X[~test_X.isna().any(axis=1) & ~np.isinf(test_X).any(axis=1)]
        
        probabilities = self.model_trainer.predict_proba(model, test_X)
        signals = self.model_trainer.predict(model, test_X)
        
        # Align with test_df
        aligned_signals = pd.Series(0, index=test_df.index)
        aligned_signals.loc[test_X.index] = signals
        
        backtest_results = self.backtester.backtest_signals(
            test_df,
            aligned_signals,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )
        
        # Step 8: Evaluate
        logger.info("Step 8: Evaluating performance...")
        metrics = self.evaluator.calculate_metrics(backtest_results['equity'])
        
        # Generate report
        report = self.evaluator.generate_report(metrics, backtest_results['equity'])
        
        results = {
            'model': model,
            'train_metrics': train_metrics,
            'backtest_results': backtest_results,
            'performance_metrics': metrics,
            'report': report,
            'evaluator': self.evaluator  # Include evaluator for plotting
        }
        
        logger.info("Pipeline completed successfully!")
        
        return results

