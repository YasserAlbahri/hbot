#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all module imports."""
    print("Testing imports...")
    
    try:
        from quantlab import __version__
        print(f"✅ quantlab.__init__ - Version: {__version__}")
    except Exception as e:
        print(f"❌ quantlab.__init__ - Error: {e}")
        return False
    
    try:
        from quantlab.utils import load_config, setup_logging, get_project_root
        print("✅ quantlab.utils - All functions imported")
    except Exception as e:
        print(f"❌ quantlab.utils - Error: {e}")
        return False
    
    try:
        from quantlab.data_loader import DataLoader
        print("✅ quantlab.data_loader - DataLoader imported")
    except Exception as e:
        print(f"❌ quantlab.data_loader - Error: {e}")
        return False
    
    try:
        from quantlab.feature_engineering import FeatureEngineer
        print("✅ quantlab.feature_engineering - FeatureEngineer imported")
    except Exception as e:
        print(f"❌ quantlab.feature_engineering - Error: {e}")
        return False
    
    try:
        from quantlab.labeling import Labeler
        print("✅ quantlab.labeling - Labeler imported")
    except Exception as e:
        print(f"❌ quantlab.labeling - Error: {e}")
        return False
    
    try:
        from quantlab.models import ModelTrainer
        print("✅ quantlab.models - ModelTrainer imported")
    except Exception as e:
        print(f"❌ quantlab.models - Error: {e}")
        return False
    
    try:
        from quantlab.backtest import Backtester
        print("✅ quantlab.backtest - Backtester imported")
    except Exception as e:
        print(f"❌ quantlab.backtest - Error: {e}")
        return False
    
    try:
        from quantlab.evaluation import Evaluator
        print("✅ quantlab.evaluation - Evaluator imported")
    except Exception as e:
        print(f"❌ quantlab.evaluation - Error: {e}")
        return False
    
    try:
        from quantlab.pipelines.training_pipeline import TrainingPipeline
        print("✅ quantlab.pipelines.training_pipeline - TrainingPipeline imported")
    except Exception as e:
        print(f"❌ quantlab.pipelines.training_pipeline - Error: {e}")
        return False
    
    print("\n✅ All imports successful!")
    return True

def test_configs():
    """Test config loading."""
    print("\nTesting configs...")
    
    try:
        from quantlab.utils import load_config
        
        data_config = load_config('configs/data_config.yaml')
        print("✅ data_config.yaml loaded")
        
        features_config = load_config('configs/features_config.yaml')
        print("✅ features_config.yaml loaded")
        
        model_config = load_config('configs/model_config.yaml')
        print("✅ model_config.yaml loaded")
        
        return True
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("HBOT Quant Lab - Import Test")
    print("=" * 50)
    
    imports_ok = test_imports()
    configs_ok = test_configs()
    
    print("\n" + "=" * 50)
    if imports_ok and configs_ok:
        print("✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)

