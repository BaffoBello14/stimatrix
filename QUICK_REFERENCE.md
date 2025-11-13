# ⚡ QUICK REFERENCE - Stimatrix

**Comandi più usati per quick reference.**

---

## 🚀 COMANDI PRINCIPALI

```bash
# Run completo (production, ~2-3 ore)
python main.py

# Run rapido (dev/test, ~20 minuti)
python main.py --config fast

# Solo preprocessing
python main.py --steps preprocessing

# Solo training (se già preprocessato)
python main.py --steps training evaluation

# Force reload (ignora cache)
python main.py --force-reload

# Help
python main.py --help
```

---

## ⚙️ FILE CONFIG

| File | Uso | Trial | Tempo | Modelli |
|------|-----|-------|-------|---------|
| **`config/config.yaml`** | Production | 150 | 2-3h | Tutti 6 + Ensemble |
| **`config/config_fast.yaml`** | Dev/Test | 20 | 20min | 4 principali |

---

## 📊 OUTPUT

```bash
# Metriche finali
models/summary.json

# Best model artifacts
models/catboost/
models/rf/
models/xgboost/
...

# Logs
logs/pipeline.log           # (config.yaml)
logs/pipeline_fast.log      # (config_fast.yaml)

# Preprocessed data
data/preprocessed/X_train_tree.parquet
data/preprocessed/X_train_catboost.parquet
...
```

---

## 🧪 TESTING

```bash
# Run tutti i test
pytest

# Con coverage
pytest --cov=src --cov-report=html

# Test specifico
pytest tests/test_preprocessing_pipeline.py

# Test pattern
pytest -k "leakage"
```

---

## 🔧 QUICK FIXES PERFORMANCE

### **Se MAPE > 50% o R² < 0.75:**

**1. Filtra outlier prezzi**
```yaml
# config/config.yaml
data_filters:
  anno_min: 2022
  prezzo_min: 20000  # Aggiungi filtro prezzo
  prezzo_max: 500000
  zone_escluse: ['E1', 'E2', 'E3', 'R1']
  tipologie_escluse: ['4']

outliers:
  iso_forest_contamination: 0.15  # Da 0.08
```

**2. Cambia trasformazione target**
```yaml
target:
  transform: 'yeojohnson'  # Invece di 'log'
```

**3. Aumenta regularization**
```yaml
catboost:
  base_params:
    l2_leaf_reg: 6.0  # Da 3.0
    depth: 5          # Da 6
```

---

## 🌍 VARIABILI AMBIENTE

```bash
# Database (se usi step schema/dataset)
export SERVER=your_server
export DATABASE=your_database
export DB_USER=your_user
export DB_PASSWORD=your_password

# W&B (opzionale)
export WANDB_ENABLED=true
export WANDB_PROJECT=stimatrix

# Logging
export LOG_LEVEL=INFO
```

Oppure crea `.env`:
```bash
SERVER=your_server
DATABASE=your_database
DB_USER=your_user
DB_PASSWORD=your_password
WANDB_ENABLED=true
```

---

## 📈 METRICHE TARGET

| Metrica | Attuale | Target Q1 | Target Q2 | Production |
|---------|---------|-----------|-----------|------------|
| R² | 0.74 | >0.80 | >0.85 | >0.90 |
| RMSE | 37k€ | <30k€ | <25k€ | <20k€ |
| MAPE | 58% | <40% | <30% | <20% |

---

## 📝 PASSI PIPELINE

1. **`schema`** - Estrae schema DB (opzionale)
2. **`dataset`** - Recupera dati da DB (opzionale)
3. **`preprocessing`** - Feature engineering, split, encoding
4. **`training`** - Tuning, training, ensemble
5. **`evaluation`** - Metriche, diagnostics, plots

**Default se hai raw data**: `["preprocessing", "training", "evaluation"]`

---

## 🎯 TROUBLESHOOTING VELOCE

**Pipeline non parte?**
```bash
# Check dipendenze
pip install -r requirements.txt

# Check raw data
ls data/raw/raw.parquet

# Check config
python -c "from utils.config import load_config; print(load_config('config/config.yaml'))"
```

**Training lentissimo?**
```bash
# Usa config fast
python main.py --config fast

# O riduci trial manualmente in config.yaml:
# trials_advanced: 50  # Invece di 150
```

**Out of memory?**
```bash
# Riduci batch size / disable SHAP
# In config.yaml:
shap:
  enabled: false
```

**MAPE troppo alto?**
- Vedi sezione "Quick Fixes Performance" sopra
- Controlla outlier in `models/*/worst_predictions.csv`
- Controlla group metrics per zone problematiche

---

## 🔗 LINK UTILI

- **README completo**: `README.md`
- **Config principale**: `config/config.yaml`
- **Config rapido**: `config/config_fast.yaml`
- **Tests**: `tests/`
- **Notebooks EDA**: `notebooks/`

---

**Questo è tutto! Happy coding!** 🚀
