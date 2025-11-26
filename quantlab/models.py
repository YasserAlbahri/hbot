"""
Machine Learning models for trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ModelTrainer:
    """
    Train and evaluate ML models for trading.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize ModelTrainer.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.mlflow_config = config.get('mlflow', {})
        
        # Setup MLflow
        if self.mlflow_config.get('enabled', True):
            mlflow.set_tracking_uri(self.mlflow_config.get('tracking_uri', 'file:./mlruns'))
            experiment_name = self.mlflow_config.get('experiment_name', 'hbot_quant_lab')
            try:
                mlflow.create_experiment(experiment_name)
            except:
                pass
            mlflow.set_experiment(experiment_name)
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict] = None
    ) -> Tuple[XGBClassifier, Dict]:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            params: Model parameters (uses config if None)
            
        Returns:
            Tuple of (trained model, metrics dictionary)
        """
        logger.info("Training XGBoost model...")
        
        xgb_config = self.config.get('xgboost', {})
        if not xgb_config.get('enabled', True):
            raise ValueError("XGBoost is disabled in config")
        
        model_params = params or xgb_config.get('params', {})
        
        # Start MLflow run
        with mlflow.start_run(run_name="xgboost_training") if self.mlflow_config.get('enabled') else nullcontext():
            # Log parameters
            if self.mlflow_config.get('enabled'):
                mlflow.log_params(model_params)
            
            # Train model
            model = XGBClassifier(**model_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            
            metrics = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'n_features': X_train.shape[1],
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val)
            }
            
            # Log metrics
            if self.mlflow_config.get('enabled'):
                mlflow.log_metrics(metrics)
                mlflow.xgboost.log_model(model, "model")
            
            logger.info(f"XGBoost trained - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            return model, metrics
    
    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict] = None
    ) -> Tuple[LGBMClassifier, Dict]:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            params: Model parameters (uses config if None)
            
        Returns:
            Tuple of (trained model, metrics dictionary)
        """
        logger.info("Training LightGBM model...")
        
        lgbm_config = self.config.get('lightgbm', {})
        if not lgbm_config.get('enabled', True):
            raise ValueError("LightGBM is disabled in config")
        
        model_params = params or lgbm_config.get('params', {})
        
        # Start MLflow run
        with mlflow.start_run(run_name="lightgbm_training") if self.mlflow_config.get('enabled') else nullcontext():
            # Log parameters
            if self.mlflow_config.get('enabled'):
                mlflow.log_params(model_params)
            
            # Train model
            model = LGBMClassifier(**model_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_names=['validation'],
                verbose=-1
            )
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            
            metrics = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'n_features': X_train.shape[1],
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val)
            }
            
            # Log metrics
            if self.mlflow_config.get('enabled'):
                mlflow.log_metrics(metrics)
                mlflow.lightgbm.log_model(model, "model")
            
            logger.info(f"LightGBM trained - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            return model, metrics
    
    def predict_proba(self, model, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            model: Trained model
            X: Features
            
        Returns:
            Probability array
        """
        return model.predict_proba(X)
    
    def predict(self, model, X: pd.DataFrame) -> np.ndarray:
        """
        Get predictions.
        
        Args:
            model: Trained model
            X: Features
            
        Returns:
            Predictions array
        """
        return model.predict(X)


from contextlib import nullcontext

