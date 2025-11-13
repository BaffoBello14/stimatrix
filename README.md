# 🏠 Stimatrix - Real Estate Price Prediction Pipeline

**End-to-end machine learning pipeline for real estate price prediction with production-ready features and leak-free contextual feature engineering.**

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Testing](#-testing)
- [Performance](#-performance)
- [Requirements](#-requirements)

---

## ✨ Features

### **🎯 Machine Learning**
- **6 regression models**: Random Forest, CatBoost, XGBoost, LightGBM, Gradient Boosting, HistGradientBoosting
- **Ensemble methods**: Voting and Stacking with automatic model selection
- **Hyperparameter tuning**: Optuna/OptunaHub with 150 trials (20 in fast mode)
- **Target transformations**: log, log10, sqrt, Box-Cox, Yeo-Johnson
- **Production-ready**: No data leakage, all features computable at inference time

### **🔧 Preprocessing**
- **Leak-free contextual features**: Zone/typology/temporal statistics calculated only on training set
- **Smart encoding**: Multi-strategy based on cardinality (OneHot, Target, Frequency, Ordinal)
- **Outlier detection**: Ensemble method (IQR, Z-score, Isolation Forest) with group-wise application
- **Missing value handling**: Group-wise imputation with multiple strategies
- **Temporal split**: Time-aware train/validation/test split (no data leakage)

### **📊 Evaluation & Diagnostics**
- **Comprehensive metrics**: R², RMSE, MSE, MAE, MAPE (original scale + transformed)
- **Group analysis**: Performance by zone, typology, price band
- **Residual analysis**: Diagnostic plots and worst predictions tracking
- **Data drift detection**: PSI and Kolmogorov-Smirnov test
- **Uncertainty quantification**: Prediction intervals with residual bootstrap

### **⚙️ Advanced**
- **Multiple profiles**: Different preprocessing pipelines (scaled, tree, catboost)
- **Experiment tracking**: Weights & Biases integration
- **Data filtering**: Experimental filters for price range, surface, zones, etc.
- **Feature importance**: SHAP values with plots
- **Reproducibility**: Fixed random seeds, full artifact logging

---

## 🚀 Quick Start

### **1. Installation**

```bash
# Clone repository
git clone <repo_url>
cd stimatrix

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. First Run (Fast Mode)**

```bash
# Run full pipeline with fast config (~20 minutes)
python main.py --config fast

# Output: models/summary.json with all metrics
```

### **3. Production Run**

```bash
# Run full pipeline with full config (~2-3 hours, better results)
python main.py

# Or specify config explicitly
python main.py --config config
```

---

## ⚙️ Configuration

### **Two Config Files**

| Config | Trials | Models | Time | Use Case |
|--------|--------|--------|------|----------|
| **`config/config.yaml`** | 150 | All 6 + Ensembles | ~2-3h | Production, final training |
| **`config/config_fast.yaml`** | 20 | RF, CatBoost, XGBoost, LightGBM | ~20min | Testing, development, debug |

### **Key Differences: config.yaml vs config_fast.yaml**

```yaml
# config.yaml (FULL)
training:
  trials_base: 50
  trials_advanced: 150        # ⏱️ More thorough search
  
  models:
    gbr: {enabled: true}       # ✅ All models enabled
    hgbt: {enabled: true}
  
  ensembles:
    voting: {enabled: true}    # ✅ Both ensembles
    stacking: {enabled: true, cv_folds: 10}

# config_fast.yaml (FAST)
training:
  trials_base: 10
  trials_advanced: 20          # ⚡ Quick search
  
  models:
    gbr: {enabled: false}      # ⚡ Only essential models
    hgbt: {enabled: false}
  
  ensembles:
    voting: {enabled: false}   # ⚡ Only stacking
    stacking: {enabled: true, cv_folds: 5}
```

### **Environment Variables**

Create a `.env` file for database credentials:

```bash
# Database
SERVER=your_sql_server
DATABASE=your_database
DB_USER=your_username
DB_PASSWORD=your_password

# Optional
WANDB_ENABLED=true
WANDB_PROJECT=stimatrix
LOG_LEVEL=INFO
```

### **Main Configuration Sections**

```yaml
# Target variable and transformation
target:
  column_candidates: ['AI_Prezzo_Ridistribuito']
  transform: 'yeojohnson'  # log | log10 | sqrt | boxcox | yeojohnson | none

# Temporal split (leak-free!)
temporal_split:
  mode: 'fraction'  # or 'date'
  fraction:
    train: 0.70
    validation: 0.15
    test: 0.15

# Outlier detection
outliers:
  enabled: true
  methods: ['iso_forest']
  iso_forest_contamination: 0.08
  group_by_col: 'AI_ZonaOmi'

# Feature pruning (data-driven)
feature_pruning:
  drop_columns:
    - 'AI_IdImmobile'  # ID columns
    - 'PC_PoligonoMetrico'  # Raw geometry
    # ... ~56 columns (see config for full list)

# Experimental data filters (optional)
data_filters:
  prezzo_min: null     # Filter by price range
  prezzo_max: null
  superficie_min: null # Filter by surface
  superficie_max: null
  zone_incluse: null   # Filter by zone
```

---

## 📁 Project Structure

```
stimatrix/
├── config/
│   ├── config.yaml           # Main config (150 trials, ~2-3h)
│   └── config_fast.yaml      # Fast config (20 trials, ~20min)
├── data/
│   ├── raw/                  # Raw data (parquet)
│   └── preprocessed/         # Preprocessed splits by profile
├── models/                   # Trained models and results
│   ├── summary.json          # All models performance
│   ├── rf/                   # Random Forest artifacts
│   ├── catboost/             # CatBoost artifacts
│   └── ...
├── logs/                     # Pipeline logs
├── notebooks/                # EDA notebooks
│   ├── eda_basic.ipynb
│   ├── eda_advanced.ipynb
│   └── eda_utils.py
├── sql/                      # SQL query templates
├── src/
│   ├── db/                   # Database connection
│   ├── dataset_builder/      # Dataset retrieval
│   ├── preprocessing/
│   │   ├── pipeline.py           # Main preprocessing
│   │   ├── contextual_features_fixed.py  # Leak-free features
│   │   ├── encoders.py           # Multi-strategy encoding
│   │   ├── outliers.py           # Outlier detection
│   │   ├── imputation.py         # Missing value handling
│   │   └── target_transforms.py  # Target transformations
│   ├── training/
│   │   ├── train.py              # Training orchestrator
│   │   ├── tuner.py              # Optuna hyperparameter tuning
│   │   ├── model_zoo.py          # Model definitions
│   │   ├── ensembles.py          # Voting & Stacking
│   │   ├── evaluation.py         # Metrics & evaluation
│   │   ├── diagnostics.py        # Residuals & drift
│   │   └── shap_utils.py         # SHAP explainability
│   ├── inference/
│   │   └── show_samples.py       # Inference examples
│   └── utils/                    # Logging, I/O, security
├── tests/                    # PyTest suite
│   ├── test_preprocessing_pipeline.py
│   ├── test_training.py
│   ├── test_encoding_no_leakage.py
│   └── ...
├── main.py                   # Pipeline orchestrator
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 💻 Usage

### **Basic Commands**

```bash
# Full pipeline (preprocessing + training + evaluation)
python main.py

# Run specific steps
python main.py --steps preprocessing training

# Use fast config
python main.py --config fast

# Force reload (ignore cached outputs)
python main.py --force-reload

# Help
python main.py --help
```

### **Pipeline Steps**

The pipeline consists of 5 main steps:

1. **`schema`**: Extract database schema (optional, if using DB)
2. **`dataset`**: Retrieve data from DB (optional, if using DB)
3. **`preprocessing`**: Feature engineering, outlier detection, encoding, split
4. **`training`**: Hyperparameter tuning, model training, ensemble creation
5. **`evaluation`**: Metrics calculation, group analysis, diagnostics

**Default**: If you have raw data in `data/raw/raw.parquet`, you can skip `schema` and `dataset`:

```yaml
# config.yaml
execution:
  steps: ["preprocessing", "training", "evaluation"]
```

### **Working with Preprocessed Data Only**

If you already have preprocessed data:

```bash
# Skip preprocessing, train only
python main.py --steps training evaluation
```

### **Experiment Tracking (W&B)**

```bash
# Enable W&B
export WANDB_ENABLED=true
export WANDB_PROJECT=stimatrix
python main.py

# Disable W&B
export WANDB_ENABLED=false
python main.py
```

### **Data Filtering (Experimental)**

To train models on specific subsets, edit `config.yaml`:

```yaml
data_filters:
  experiment_name: "high_end_properties"
  description: "Properties 200k-500k€"
  prezzo_min: 200000
  prezzo_max: 500000
  superficie_min: 80
  superficie_max: 200
```

---

## 🧪 Testing

### **Run All Tests**

```bash
# Run full test suite
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing_pipeline.py

# Run tests matching pattern
pytest -k "leakage"
```

### **Test Categories**

- **`test_preprocessing_pipeline.py`**: Full preprocessing workflow
- **`test_encoding_no_leakage.py`**: Verify no data leakage in encoding
- **`test_training.py`**: Training and tuning logic
- **`test_security.py`**: Input validation and SQL injection prevention
- **`test_temporal_split_fix.py`**: Temporal split correctness
- **`test_target_transforms.py`**: Target transformation correctness
- **`test_overflow_prevention.py`**: Numeric stability

---

## 📈 Performance

### **Expected Results (Production Config)**

| Metric | Target | Typical |
|--------|--------|---------|
| **R²** | >0.85 | 0.73-0.80 |
| **RMSE** | <25k€ | 32k-40k€ |
| **MAPE** | <30% | 45-60% |
| **Training Time** | - | 2-3 hours |

**Note**: Performance depends heavily on:
- Dataset quality and size
- Feature engineering
- Hyperparameter tuning budget
- Data filters applied

### **Quick Fixes for Better Performance**

If MAPE > 50% or R² < 0.75, try:

1. **Filter outliers**:
   ```yaml
   data_filters:
     prezzo_min: 20000
     prezzo_max: 500000
     superficie_min: 10
     superficie_max: 300
   
   outliers:
     iso_forest_contamination: 0.15  # More aggressive
   ```

2. **Change target transform**:
   ```yaml
   target:
     transform: 'yeojohnson'  # Better for wide ranges
   ```

3. **Increase regularization**:
   ```yaml
   catboost:
     base_params:
       l2_leaf_reg: 6.0  # Higher regularization
       depth: 5          # Shallower trees
   ```

---

## 📦 Requirements

### **Python Version**
- **Python 3.10+** (tested on 3.10, 3.11)

### **Key Dependencies**
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `scikit-learn >= 1.3.0`
- `catboost >= 1.2.0`
- `xgboost >= 2.0.0`
- `lightgbm >= 4.0.0`
- `optuna >= 3.4.0`
- `optunahub >= 0.1.0`
- `shap >= 0.42.0`
- `wandb >= 0.15.0` (optional, for experiment tracking)
- `pytest >= 7.4.0` (for testing)

See `requirements.txt` for full list.

### **Operating System**
- Linux (recommended)
- macOS
- Windows (with WSL recommended)

### **Hardware Recommendations**
- **CPU**: 8+ cores
- **RAM**: 16GB+ (32GB recommended for large datasets)
- **Storage**: 10GB+ free space

---

## 📝 Notes

### **Important: No Data Leakage**

This pipeline is designed with **zero data leakage**:

- ✅ Contextual features (zone statistics, typology aggregates) are **fit only on training set**
- ✅ Target encoding uses **training-only statistics**
- ✅ Temporal split ensures **no future information in training**
- ✅ All preprocessing steps follow **fit → transform** pattern
- ✅ Production-ready: all features computable at inference time (no target required)

### **Target Variable**

The pipeline predicts **`AI_Prezzo_Ridistribuito`** (redistributed price):
- Price redistributed across multiple properties in the same transaction
- More accurate than raw transaction price for properties sold in bundles
- Original scale: euros (€)
- Transformed scale: depends on `target.transform` setting

### **Feature Pruning**

~56 columns are dropped by default (data-driven decisions):
- ID columns (non-predictive)
- Geometry raw formats (processed into features)
- Highly correlated features (>0.98 correlation)
- High missing ratio (>80% missing)
- Constants or near-constants

See `config.yaml` → `feature_pruning.drop_columns` for full list.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## 📄 License

[Add your license here]

---

## 📧 Contact

[Add contact information here]

---

## 🙏 Acknowledgments

- Optuna team for hyperparameter optimization framework
- CatBoost, XGBoost, LightGBM teams for excellent gradient boosting libraries
- scikit-learn contributors for solid ML foundation
- SHAP for model explainability tools

---

**Happy predicting!** 🏠📊
