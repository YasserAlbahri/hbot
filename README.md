# HBOT Quantitative Trading Research Lab

## 🎯 الهدف

مختبر بحثي متقدم للتداول الكمي بالذكاء الاصطناعي. هذا المشروع هو الأساس العلمي الذي سيُبنى عليه نظام HBOT الكامل.

## 📁 هيكل المشروع

```
quant_lab/
├── data/
│   ├── raw/                # البيانات الخام
│   ├── processed/          # البيانات المعالجة
│   └── features/           # الميزات الجاهزة
├── notebooks/              # Jupyter notebooks للتجارب
├── configs/                # ملفات الإعدادات YAML
├── quantlab/               # الكود الأساسي
│   ├── data_loader.py      # تحميل البيانات
│   ├── feature_engineering.py  # بناء الميزات
│   ├── labeling.py         # Triple-Barrier labeling
│   ├── models.py           # نماذج ML
│   ├── backtest.py         # Backtesting
│   ├── evaluation.py       # تقييم الأداء
│   └── pipelines/          # Pipelines كاملة
├── mlruns/                 # MLflow experiments
└── requirements.txt        # المتطلبات
```

## 🚀 البدء السريع

### 1. إعداد البيئة

```bash
cd /home/admin/web/hbot.falnakon.com/public_html/quant_lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. تشغيل Pipeline كامل

```python
from quantlab.pipelines.training_pipeline import TrainingPipeline
from quantlab.utils import load_config

# Load configs
data_config = load_config('configs/data_config.yaml')
features_config = load_config('configs/features_config.yaml')
model_config = load_config('configs/model_config.yaml')

# Run pipeline
pipeline = TrainingPipeline(data_config, features_config, model_config)
results = pipeline.run(symbol='EURUSD', timeframe='15m')
```

## 📊 المكونات الرئيسية

### 1. Data Loader
- تحميل OHLCV من CSV أو APIs
- تنظيف البيانات
- Resampling للفريمات المختلفة

### 2. Feature Engineering
- مؤشرات فنية (RSI, MACD, ATR, Bollinger Bands)
- خصائص الشموع
- Support & Resistance
- Multi-timeframe features
- Time-based features
- Session features

### 3. Labeling
- Triple-Barrier method (López de Prado)
- Fixed-horizon labels
- Trend-based labels

### 4. Models
- XGBoost
- LightGBM
- Deep Learning (لاحقاً)

### 5. Backtesting
- Vectorized backtesting
- Walk-forward analysis
- Performance metrics

### 6. Evaluation
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Win Rate
- Profit Factor
- Visualizations

## 🔧 الإعدادات

جميع الإعدادات في مجلد `configs/`:
- `data_config.yaml`: إعدادات البيانات
- `features_config.yaml`: إعدادات الميزات
- `model_config.yaml`: إعدادات النماذج

## 📈 MLflow

جميع التجارب تُسجل في MLflow:
```bash
mlflow ui --backend-store-uri file:./mlruns
```

## 📝 ملاحظات

- المشروع مصمم ليكون Research Lab منفصل عن Django
- يمكن دمجه لاحقاً مع نظام HBOT الكامل
- جميع المكونات قابلة للتوسع والتخصيص

---

**الإصدار:** 1.0.0  
**التاريخ:** 2025-01-27

