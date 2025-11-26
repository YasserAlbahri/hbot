# 🚀 Quick Start Guide

## الخطوة 1: إعداد البيئة

```bash
cd /home/admin/web/hbot.falnakon.com/public_html/quant_lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## الخطوة 2: تشغيل Pipeline كامل

```python
from quantlab.pipelines.training_pipeline import TrainingPipeline
from quantlab.utils import load_config, setup_logging

# Setup
setup_logging()

# Load configs
data_config = load_config('configs/data_config.yaml')
features_config = load_config('configs/features_config.yaml')
model_config = load_config('configs/model_config.yaml')

# Run pipeline
pipeline = TrainingPipeline(data_config, features_config, model_config)
results = pipeline.run(
    symbol='EURUSD',
    timeframe='15m',
    model_type='xgboost'
)

# View results
print(results['report'])
```

## الخطوة 3: استخدام Notebooks

```bash
jupyter lab notebooks/
```

افتح `01_quick_start.ipynb` للبدء.

## 📊 النتائج

بعد تشغيل Pipeline، ستحصل على:
- ✅ نموذج مدرب (XGBoost/LightGBM)
- ✅ نتائج Backtest
- ✅ مقاييس الأداء (Sharpe, Max DD, Win Rate, etc.)
- ✅ تقرير مفصل
- ✅ رسومات Equity Curve

## 🔍 MLflow UI

لعرض التجارب:
```bash
mlflow ui --backend-store-uri file:./mlruns
```

ثم افتح http://localhost:5000

---

**جاهز للبدء! 🎉**

