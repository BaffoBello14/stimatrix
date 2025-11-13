# 📝 TODO - Future Improvements (Non Urgent)

**Last Updated**: 2025-11-12

---

## 🗄️ DOCUMENTAZIONE STORICA

### **Consolidare file di documentazione legacy**

**Current State**:
```
/workspace/
  ├── LEAKAGE_FIX.md                 # Data leakage fix documentation
  ├── LEAKAGE_SUMMARY.txt             # Visual recap leakage
  ├── PRODUCTION_READY_FEATURES.md    # Production readiness
  ├── MODIFICHE_APPLICATE.md          # Latest changes (AI_Prezzo_MQ removal + data_filters)
  ├── DATA_DRIVEN_ANALYSIS.md         # Feature pruning analysis
  └── CLEANUP_SUMMARY.md              # This cleanup session
```

**Proposal**:
- ✅ Creare cartella `docs/history/` o `docs/legacy/`
- ✅ Spostare tutti i file storici (LEAKAGE_*, PRODUCTION_*, MODIFICHE_*, DATA_DRIVEN_*)
- ✅ Creare un unico `CHANGELOG.md` nella root con sommario cronologico
- ✅ Mantenere solo `README.md` e `CHANGELOG.md` nella root

**Benefit**: Root più pulita, storico organizzato

---

## 🔧 REFACTORING CODICE

### **1. Rimozione Magic Numbers**

**Occorrenze da verificare:**
```python
# src/preprocessing/transformers.py
def remove_highly_correlated(X: pd.DataFrame, threshold: float = 0.98):
    # ⚠️ Threshold hardcoded, anche se passato come parametro il default è magic

# src/preprocessing/transformers.py
def drop_non_descriptive(X: pd.DataFrame, na_threshold: float = 0.98):
    # ⚠️ Come sopra
```

**Proposal**: Definire costanti globali in `src/utils/constants.py`:
```python
DEFAULT_CORRELATION_THRESHOLD = 0.98
DEFAULT_NA_THRESHOLD = 0.98
DEFAULT_OUTLIER_CONTAMINATION = 0.05
```

**Benefit**: Più facile modificare defaults, meno duplicazione

---

### **2. Type Hints Completi**

**Current**: Alcuni file hanno type hints parziali

**Proposal**: Aggiungere type hints completi a tutti i file, soprattutto:
- `src/preprocessing/encoders.py`
- `src/preprocessing/feature_extractors.py`
- `src/training/diagnostics.py`

**Tool**: `mypy --strict` per validazione

**Benefit**: Type safety, IDE autocomplete migliore

---

### **3. Docstrings Consistenti**

**Current**: Mix di docstring styles (Google, NumPy, plain text)

**Proposal**: Standardizzare su Google style:
```python
def my_function(arg1: str, arg2: int) -> bool:
    """
    Brief description.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When...
    """
```

**Tool**: `pydocstyle` per validazione

**Benefit**: Documentazione uniforme, migliore generazione docs

---

## 🧪 TESTING

### **1. Unit Tests per Contextual Features**

**Current**: `src/preprocessing/contextual_features_fixed.py` non ha test dedicati

**Proposal**: Creare `tests/test_contextual_features.py`:
```python
def test_fit_contextual_features():
    # Test che fit calcoli statistiche solo su train
    pass

def test_transform_contextual_features_no_leakage():
    # Test che transform non usi target dell'istanza
    pass

def test_production_ready_features():
    # Test che tutte le feature siano calcolabili senza target
    pass
```

**Benefit**: Garantisce no regressione su leakage fix

---

### **2. Integration Tests per Pipeline Completa**

**Current**: Test esistenti sono per singoli componenti

**Proposal**: Creare `tests/test_pipeline_integration.py`:
```python
def test_full_pipeline_run():
    # Test che pipeline completa funzioni end-to-end
    pass

def test_pipeline_with_data_filters():
    # Test nuovi data filters
    pass

def test_pipeline_reproducibility():
    # Test che due run con stesso seed diano stesso risultato
    pass
```

**Benefit**: Catch breaking changes in interazioni tra componenti

---

## 🚀 PERFORMANCE

### **1. Profiling Pipeline Bottlenecks**

**Tool**: `cProfile` o `py-spy`

**Areas to Profile**:
- Feature extraction (geometry, JSON parsing)
- Encoding (target encoding su high-cardinality)
- Contextual features aggregation

**Proposal**: Identificare e ottimizzare top 3 bottleneck

---

### **2. Parallelize Feature Extraction**

**Current**: Sequential processing di geometry/JSON features

**Proposal**: Usare `multiprocessing` o `joblib.Parallel` per:
- Geometry parsing (row-level)
- JSON parsing (row-level)
- Aggregazioni contextual (group-level)

**Benefit**: Potenziale speedup 2-4x su large datasets

---

## 📊 MONITORING & OBSERVABILITY

### **1. Structured Logging**

**Current**: Plain text logs

**Proposal**: Passare a structured logging (JSON):
```python
logger.info("Training completed", extra={
    "model": "catboost",
    "rmse": 8911.23,
    "mape": 2.68,
    "duration_seconds": 145.2
})
```

**Tool**: `python-json-logger`

**Benefit**: Migliore parsing/aggregazione logs, integrazione con monitoring tools

---

### **2. Metrics Dashboard**

**Current**: Solo W&B per tracking

**Proposal**: Aggiungere export a Prometheus/Grafana per metriche real-time:
- Training time per model
- RMSE/MAPE trends
- Dataset size over time
- Feature count per profile

**Benefit**: Monitoring centralizzato, alerting su regressioni

---

## 🔐 SECURITY & BEST PRACTICES

### **1. Input Validation**

**Proposal**: Aggiungere validazione esplicita degli input in entrypoints:
- `main.py`: Validare esistenza file config
- `src/preprocessing/pipeline.py`: Validare schema del raw DataFrame
- `src/training/train.py`: Validare esistenza preprocessed files

**Tool**: `pydantic` per validazione config

---

### **2. Error Handling**

**Current**: Mix di try/except con logging generico

**Proposal**: Definire custom exceptions:
```python
# src/utils/exceptions.py
class DataValidationError(Exception): pass
class ModelTrainingError(Exception): pass
class ConfigurationError(Exception): pass
```

**Benefit**: Error handling più granulare, migliore debugging

---

## 📦 DEPENDENCY MANAGEMENT

### **1. Pin Exact Versions**

**Current**: `requirements.txt` ha alcune versioni pinned, altre no

**Proposal**: Usare `pip-tools` per lockfile:
```bash
# requirements.in (high-level deps)
pandas>=2.0.0
scikit-learn>=1.3.0

# requirements.txt (pinned versions via pip-compile)
pandas==2.1.4
scikit-learn==1.3.2
...
```

**Benefit**: Reproducibility, avoid surprises da minor version bumps

---

### **2. Dependency Audit**

**Proposal**: Eseguire periodicamente:
```bash
pip list --outdated
pip-audit  # Security vulnerabilities
```

**Benefit**: Aggiornamenti sicuri, patch security issues

---

## 🎨 CODE QUALITY

### **1. Pre-commit Hooks**

**Proposal**: Setup `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
```

**Benefit**: Code quality automatica, catch issues prima di commit

---

### **2. Coverage Target**

**Current**: Alcuni test esistenti ma no tracking coverage

**Proposal**: Setup `pytest-cov`:
```bash
pytest --cov=src --cov-report=html --cov-report=term
```

**Target**: ≥80% coverage per file critici (pipeline, train, evaluation)

---

## 🌟 FEATURE IDEAS (LOW PRIORITY)

### **1. Auto-Feature Engineering**

**Idea**: Integrare `featuretools` o `tsfresh` per auto-generazione feature

**Benefit**: Scoprire interaction features non ovvie

**Risk**: Overfitting, interpretability loss

---

### **2. AutoML Integration**

**Idea**: Aggiungere supporto per `FLAML`, `AutoGluon`, o `TPOT`

**Benefit**: Potenzialmente migliori hyperparameters/architetture

**Risk**: Loss of control, training time

---

### **3. Model Compression**

**Idea**: Dopo training, applicare pruning/quantization per inference più veloce

**Tool**: `ONNX Runtime`, model distillation

**Benefit**: Deploy più leggero, latency ridotta

---

## ⏱️ PRIORITÀ

### **🔥 High (Next Sprint)**
1. ✅ Unit tests contextual features
2. ✅ Consolidare documentazione legacy
3. ✅ Pre-commit hooks setup

### **🔸 Medium (Prossimi 2-3 mesi)**
1. ✅ Type hints completi + mypy strict
2. ✅ Profiling + ottimizzazioni performance
3. ✅ Structured logging

### **🔹 Low (Nice to Have)**
1. ✅ Auto-feature engineering exploration
2. ✅ AutoML integration
3. ✅ Metrics dashboard

---

## 📝 NOTE

- Questo file è un **wishlist**, non urgente
- Ogni item dovrebbe avere un issue dedicato se implementato
- Priorità possono cambiare in base a feedback utente/business needs

**Remember**: "Perfect is the enemy of good" - il codice attuale funziona bene! ✨
