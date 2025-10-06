# 🔧 Quick Fixes - Codice Pronto all'Uso

**Data**: 2025-10-06  
**Priorità**: 🔴 ALTA - Implementare ASAP

Questo documento contiene codice pronto all'uso per le correzioni più critiche identificate nell'analisi.

---

## 1️⃣ Fix TimeSeriesSplit per CV

### ❌ Problema Attuale

In `config/config.yaml` (linea 189):
```yaml
cv_when_no_val:
  enabled: true
  kind: kfold        # ⚠️ SBAGLIATO per time series!
  n_splits: 5
  shuffle: true      # ⚠️ ROMPE ordine temporale!
```

Questo causa **temporal leakage**: il modello vede dati futuri durante la validazione!

### ✅ Soluzione

#### Step 1: Aggiornare Config

**File**: `config/config.yaml`

```yaml
cv_when_no_val:
  enabled: true
  kind: timeseries   # ✅ CORRETTO
  n_splits: 5
  shuffle: false     # ✅ Mai shuffle con time series
```

#### Step 2: Modificare `tuner.py`

**File**: `src/training/tuner.py`

Trovare la sezione che crea il CV splitter (circa linea 100-120) e sostituire:

```python
# PRIMA (circa linea 110):
from sklearn.model_selection import KFold
cv = KFold(n_splits=cv_config.get("n_splits", 5), shuffle=True, random_state=seed)

# DOPO:
from sklearn.model_selection import KFold, TimeSeriesSplit

cv_kind = cv_config.get("kind", "kfold")
n_splits = cv_config.get("n_splits", 5)

if cv_kind == "timeseries":
    # TimeSeriesSplit garantisce che train < validation temporalmente
    cv = TimeSeriesSplit(n_splits=n_splits)
    logger.info(f"Using TimeSeriesSplit with {n_splits} splits (no temporal leakage)")
elif cv_kind == "kfold":
    # KFold solo per dati non temporali
    shuffle = cv_config.get("shuffle", False)  # Default False per sicurezza
    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
    logger.info(f"Using KFold with {n_splits} splits, shuffle={shuffle}")
else:
    raise ValueError(f"Unknown CV kind: {cv_kind}")
```

#### Step 3: Aggiungere Test

**File**: `tests/test_temporal_split_fix.py` (nuovo o estendere esistente)

```python
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def test_timeseries_cv_no_leakage():
    """Verifica che TimeSeriesSplit non causi temporal leakage."""
    # Crea dati temporali ordinati
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'value': range(100),
        'target': range(100, 200)
    })
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    for train_idx, val_idx in tscv.split(df):
        train_dates = df.iloc[train_idx]['date']
        val_dates = df.iloc[val_idx]['date']
        
        # ASSERT CRITICO: tutti i dati di validation devono essere DOPO il train
        assert train_dates.max() < val_dates.min(), \
            f"Temporal leakage detected! Train max: {train_dates.max()}, Val min: {val_dates.min()}"
        
        print(f"✅ Fold OK - Train: {train_dates.min()} to {train_dates.max()}, "
              f"Val: {val_dates.min()} to {val_dates.max()}")

if __name__ == "__main__":
    test_timeseries_cv_no_leakage()
    print("✅ All TimeSeriesSplit tests passed!")
```

**Eseguire test**:
```bash
pytest tests/test_temporal_split_fix.py -v
```

---

## 2️⃣ Robust Outlier Detection

### ❌ Problema Attuale

In `src/preprocessing/outliers.py` (linea 39-46), lo Z-score standard assume normalità:

```python
def _inliers_zscore(values: pd.Series, z_thresh: float) -> pd.Series:
    v = values.astype(float)
    mean = v.mean()           # ⚠️ Media sensibile a outlier
    std = v.std(ddof=0)       # ⚠️ Std sensibile a outlier
    if std == 0 or np.isnan(std):
        return pd.Series(True, index=v.index)
    z = (v - mean) / std      # ⚠️ Assume normalità
    return z.abs() <= z_thresh
```

Con **skewness 5.16**, questo metodo è **inadatto**!

### ✅ Soluzione

**File**: `src/preprocessing/outliers.py`

Aggiungere nuova funzione dopo `_inliers_zscore`:

```python
def _inliers_zscore_robust(values: pd.Series, z_thresh: float) -> pd.Series:
    """
    Robust Z-score using MAD (Median Absolute Deviation).
    
    Più robusto a outlier e distribuzioni skewed rispetto a Z-score standard.
    
    Formula:
        Modified Z-score = 0.6745 * (X - median) / MAD
    
    Dove MAD = median(|X - median(X)|)
    
    Args:
        values: Serie di valori
        z_thresh: Soglia per considerare outlier (tipicamente 3.5-4.0)
    
    Returns:
        Boolean mask: True = inlier, False = outlier
    """
    from scipy.stats import median_abs_deviation
    
    v = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    
    if v.empty:
        return pd.Series(True, index=values.index)
    
    median = v.median()
    mad = median_abs_deviation(v, nan_policy='omit')
    
    if mad == 0 or np.isnan(mad):
        # Se MAD = 0, tutti i valori sono uguali -> nessun outlier
        return pd.Series(True, index=values.index)
    
    # Constant per approssimare std in distribuzione normale
    # 0.6745 è il 75° percentile della distribuzione normale standard
    robust_z = 0.6745 * (v - median) / mad
    
    # Create full mask (handle missing values in original series)
    mask = pd.Series(True, index=values.index)
    mask.loc[v.index] = robust_z.abs() <= z_thresh
    
    return mask
```

Modificare `_detect_inliers_series` per usare il metodo robusto:

```python
def _detect_inliers_series(values: pd.Series, cfg: OutlierConfig) -> pd.Series:
    if cfg.method == "iqr":
        return _inliers_iqr(values, cfg.iqr_factor)
    if cfg.method == "zscore":
        return _inliers_zscore(values, cfg.z_thresh)
    if cfg.method == "zscore_robust":  # ✅ NUOVO
        return _inliers_zscore_robust(values, cfg.z_thresh)
    if cfg.method == "iso_forest":
        return _inliers_isoforest(values, cfg.iso_forest_contamination, cfg.random_state)
    if cfg.method == "ensemble":
        return _ensemble_inliers(values, cfg)
    if cfg.method == "ensemble_robust":  # ✅ NUOVO
        return _ensemble_inliers_robust(values, cfg)
    # default safe
    return pd.Series(True, index=values.index)
```

Aggiungere nuovo ensemble con metodo robusto:

```python
def _ensemble_inliers_robust(values: pd.Series, cfg: OutlierConfig) -> pd.Series:
    """
    Ensemble robusto con:
    - IQR (robusto)
    - Robust Z-score (MAD-based)
    - IsolationForest
    
    Pesi: IsolationForest 50%, IQR 25%, Robust Z-score 25%
    """
    m_iqr = _inliers_iqr(values, cfg.iqr_factor)
    m_z_robust = _inliers_zscore_robust(values, cfg.z_thresh)
    m_iso = _inliers_isoforest(values, cfg.iso_forest_contamination, cfg.random_state)
    
    # Weighted voting: IsolationForest ha peso maggiore
    # Outlier se almeno 2 metodi concordano (majority voting)
    votes_inlier = m_iqr.astype(int) + m_z_robust.astype(int) + m_iso.astype(int)
    return votes_inlier >= 2
```

**Aggiornare Config**:

```yaml
# config/config.yaml
outliers:
  method: 'ensemble_robust'      # ✅ Usa ensemble robusto
  z_thresh: 3.5                  # Threshold più alto per metodo robusto
  iqr_factor: 1.5
  iso_forest_contamination: 0.02
  group_by_col: 'AI_IdTipologiaEdilizia'
  min_group_size: 30
  fallback_strategy: 'global'
```

#### Test

**File**: `tests/test_robust_outliers.py`

```python
import pytest
import pandas as pd
import numpy as np
from scipy.stats import skew

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.outliers import (
    _inliers_zscore,
    _inliers_zscore_robust,
    OutlierConfig,
    detect_outliers
)

def test_robust_zscore_with_skewed_data():
    """Test che robust z-score funzioni meglio con dati skewed."""
    # Genera dati log-normali (altamente skewed)
    np.random.seed(42)
    data = np.random.lognormal(mean=10, sigma=1.5, size=1000)
    
    # Aggiungi 10 outlier estremi
    outliers_idx = np.random.choice(1000, 10, replace=False)
    data[outliers_idx] = data[outliers_idx] * 10
    
    series = pd.Series(data)
    print(f"Data skewness: {skew(series):.2f}")  # Dovrebbe essere ~5-6
    
    # Z-score standard (assume normalità)
    mask_standard = _inliers_zscore(series, z_thresh=3.0)
    n_outliers_standard = (~mask_standard).sum()
    
    # Robust z-score (MAD-based)
    mask_robust = _inliers_zscore_robust(series, z_thresh=3.5)
    n_outliers_robust = (~mask_robust).sum()
    
    print(f"Z-score standard detected: {n_outliers_standard} outliers")
    print(f"Robust Z-score detected: {n_outliers_robust} outliers")
    
    # Robust dovrebbe rilevare numero più sensato di outlier
    assert n_outliers_robust >= 5, "Dovrebbe rilevare almeno metà outlier veri"
    assert n_outliers_robust <= 50, "Non dovrebbe rilevare troppi falsi positivi"
    
    # Standard Z-score spesso fallisce con dati skewed
    # (rileva troppi o troppo pochi outlier)

def test_outlier_detection_on_real_estate_prices():
    """Test su dati simulati simili a prezzi immobiliari."""
    # Simula distribuzione prezzi reali (log-normale)
    np.random.seed(42)
    
    # Prezzi normali: media 50k, range 20k-200k
    normal_prices = np.random.lognormal(mean=np.log(50000), sigma=0.7, size=950)
    
    # Outlier estremi: ville di lusso >500k
    luxury_outliers = np.random.uniform(500000, 1500000, size=50)
    
    all_prices = np.concatenate([normal_prices, luxury_outliers])
    np.random.shuffle(all_prices)
    
    df = pd.DataFrame({
        'price': all_prices,
        'category': ['A'] * len(all_prices)
    })
    
    print(f"\nDistribuzione prezzi:")
    print(f"  Mean: {df['price'].mean():,.0f}")
    print(f"  Median: {df['price'].median():,.0f}")
    print(f"  Std: {df['price'].std():,.0f}")
    print(f"  Skewness: {skew(df['price']):.2f}")
    print(f"  Max: {df['price'].max():,.0f}")
    
    # Test con metodo ensemble robusto
    config = OutlierConfig(
        method='ensemble_robust',
        z_thresh=3.5,
        iqr_factor=1.5,
        iso_forest_contamination=0.05,
        random_state=42
    )
    
    mask = detect_outliers(df, 'price', config)
    n_removed = (~mask).sum()
    
    print(f"\nOutliers rimossi: {n_removed} ({n_removed/len(df)*100:.1f}%)")
    print(f"Prezzo medio dopo rimozione: {df.loc[mask, 'price'].mean():,.0f}")
    
    # Dovrebbe rimuovere circa 3-7% (50 outlier veri + alcuni borderline)
    assert 30 <= n_removed <= 100, f"Expected 30-100 outliers, got {n_removed}"
    
    # Media dopo rimozione dovrebbe essere più vicina alla mediana
    mean_after = df.loc[mask, 'price'].mean()
    median_after = df.loc[mask, 'price'].median()
    assert mean_after < 80000, "Mean troppo alta, non ha rimosso abbastanza outlier"

if __name__ == "__main__":
    test_robust_zscore_with_skewed_data()
    test_outlier_detection_on_real_estate_prices()
    print("\n✅ All robust outlier tests passed!")
```

**Eseguire**:
```bash
python tests/test_robust_outliers.py
```

---

## 3️⃣ Comparazione Target Transforms

### ❌ Problema Attuale

Solo log-transform è usato, ma potrebbe non essere ottimale per skewness 5.16.

### ✅ Soluzione

**File**: `src/preprocessing/pipeline.py`

Sostituire la funzione `apply_log_target_if` (linee 48-53):

```python
# PRIMA:
def apply_log_target_if(config: Dict[str, Any], y: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    use_log = bool(config.get("target", {}).get("log_transform", False))
    if not use_log:
        return y, {"log": False}
    y_pos = y.clip(lower=1e-6)
    return np.log1p(y_pos), {"log": True}

# DOPO:
from scipy.stats import boxcox, yeojohnson, skew
from sklearn.preprocessing import QuantileTransformer

def find_best_target_transform(y: pd.Series, config: Dict[str, Any]) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Testa multiple trasformazioni e sceglie quella che minimizza skewness.
    
    Args:
        y: Target series
        config: Configuration dict
        
    Returns:
        (transformed_y, metadata)
    """
    target_cfg = config.get("target", {})
    method = target_cfg.get("transform_method", "auto")
    
    if method == "none":
        return y, {"method": "none", "params": None, "original_skew": skew(y)}
    
    transforms = {}
    original_skew = skew(y)
    
    # 1. Log1p (classico)
    try:
        y_log = np.log1p(y.clip(lower=0))
        transforms['log1p'] = {
            'data': y_log,
            'params': None,
            'skew': abs(skew(y_log)),
            'invertible': lambda x: np.expm1(x)
        }
    except Exception as e:
        logger.warning(f"Log1p transform failed: {e}")
    
    # 2. Box-Cox (ottimale per valori positivi)
    if (y > 0).all():
        try:
            y_bc, lambda_bc = boxcox(y)
            transforms['boxcox'] = {
                'data': pd.Series(y_bc, index=y.index),
                'params': {'lambda': lambda_bc},
                'skew': abs(skew(y_bc)),
                'invertible': lambda x: np.power(x * lambda_bc + 1, 1/lambda_bc)
            }
        except Exception as e:
            logger.warning(f"Box-Cox transform failed: {e}")
    
    # 3. Yeo-Johnson (accetta valori negativi)
    try:
        y_yj, lambda_yj = yeojohnson(y)
        transforms['yeojohnson'] = {
            'data': pd.Series(y_yj, index=y.index),
            'params': {'lambda': lambda_yj},
            'skew': abs(skew(y_yj)),
            'invertible': None  # Complex inverse
        }
    except Exception as e:
        logger.warning(f"Yeo-Johnson transform failed: {e}")
    
    # 4. Quantile (molto robusto a outlier)
    try:
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        y_qt = qt.fit_transform(y.values.reshape(-1, 1)).ravel()
        transforms['quantile'] = {
            'data': pd.Series(y_qt, index=y.index),
            'params': {'transformer': qt},
            'skew': abs(skew(y_qt)),
            'invertible': lambda x: qt.inverse_transform(x.reshape(-1, 1)).ravel()
        }
    except Exception as e:
        logger.warning(f"Quantile transform failed: {e}")
    
    # 5. Square root (semplice, per moderate skewness)
    try:
        y_sqrt = np.sqrt(y.clip(lower=0))
        transforms['sqrt'] = {
            'data': y_sqrt,
            'params': None,
            'skew': abs(skew(y_sqrt)),
            'invertible': lambda x: np.power(x, 2)
        }
    except Exception as e:
        logger.warning(f"Sqrt transform failed: {e}")
    
    if not transforms:
        logger.warning("All transforms failed, using original target")
        return y, {"method": "none", "params": None, "original_skew": original_skew}
    
    # Scegli trasformazione che minimizza skewness assoluto
    if method == "auto":
        best_name = min(transforms.items(), key=lambda x: x[1]['skew'])[0]
    else:
        if method not in transforms:
            logger.warning(f"Requested method '{method}' not available, falling back to log1p")
            best_name = 'log1p' if 'log1p' in transforms else list(transforms.keys())[0]
        else:
            best_name = method
    
    best = transforms[best_name]
    
    logger.info(f"Target transformation:")
    logger.info(f"  Original skewness: {original_skew:.3f}")
    logger.info(f"  Best method: {best_name}")
    logger.info(f"  Transformed skewness: {best['skew']:.3f}")
    logger.info(f"  Skewness reduction: {(1 - best['skew']/abs(original_skew))*100:.1f}%")
    
    return best['data'], {
        "method": best_name,
        "params": best['params'],
        "original_skew": original_skew,
        "transformed_skew": best['skew'],
        "invertible": best.get('invertible')
    }
```

**Aggiornare chiamata in `run_preprocessing`** (linea 245):

```python
# PRIMA:
y_train, log_meta = apply_log_target_if(config, y_train)

# DOPO:
y_train, transform_meta = find_best_target_transform(y_train, config)

# Applicare stessa trasformazione a val e test
if transform_meta['method'] == 'log1p':
    y_test = np.log1p(y_test.clip(lower=1e-6))
    if y_val is not None:
        y_val = np.log1p(y_val.clip(lower=1e-6))
elif transform_meta['method'] == 'boxcox':
    lambda_bc = transform_meta['params']['lambda']
    y_test = pd.Series(boxcox(y_test, lmbda=lambda_bc), index=y_test.index)
    if y_val is not None:
        y_val = pd.Series(boxcox(y_val, lmbda=lambda_bc), index=y_val.index)
elif transform_meta['method'] == 'quantile':
    qt = transform_meta['params']['transformer']
    y_test = pd.Series(qt.transform(y_test.values.reshape(-1, 1)).ravel(), index=y_test.index)
    if y_val is not None:
        y_val = pd.Series(qt.transform(y_val.values.reshape(-1, 1)).ravel(), index=y_val.index)
# ... altri metodi
```

**Config**:

```yaml
# config/config.yaml
target:
  column_candidates: ['AI_Prezzo_Ridistribuito', 'AI_Prezzo_MQ']
  transform_method: 'auto'  # 'auto' | 'log1p' | 'boxcox' | 'yeojohnson' | 'quantile' | 'sqrt' | 'none'
```

---

## 4️⃣ Script di Validazione Completo

Creare uno script per verificare che tutte le fix siano state applicate correttamente.

**File**: `scripts/validate_fixes.py`

```python
#!/usr/bin/env python3
"""
Script di validazione per verificare che le fix critiche siano implementate.

Usage:
    python scripts/validate_fixes.py
"""

import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def validate_config():
    """Valida configurazione per TimeSeriesSplit."""
    print("\n1️⃣ Validating config.yaml...")
    
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("   ❌ config.yaml not found!")
        return False
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Check CV config
    cv_cfg = config.get("training", {}).get("cv_when_no_val", {})
    
    if cv_cfg.get("kind") != "timeseries":
        print(f"   ❌ CV kind is '{cv_cfg.get('kind')}', should be 'timeseries'")
        return False
    
    if cv_cfg.get("shuffle", False):
        print("   ❌ CV shuffle is True, should be False for time series")
        return False
    
    # Check outlier config
    outlier_cfg = config.get("outliers", {})
    method = outlier_cfg.get("method", "")
    
    if "robust" in method or method == "ensemble_robust":
        print("   ✅ Using robust outlier detection")
    else:
        print(f"   ⚠️  Outlier method is '{method}', consider 'ensemble_robust'")
    
    # Check target transform
    target_cfg = config.get("target", {})
    transform = target_cfg.get("transform_method", "log_transform")
    
    if transform == "auto":
        print("   ✅ Using auto target transform")
    else:
        print(f"   ⚠️  Target transform is '{transform}', consider 'auto'")
    
    print("   ✅ Config validation passed!")
    return True

def validate_code():
    """Valida che le modifiche al codice siano presenti."""
    print("\n2️⃣ Validating code changes...")
    
    # Check tuner.py
    tuner_path = Path("src/training/tuner.py")
    if not tuner_path.exists():
        print("   ❌ tuner.py not found!")
        return False
    
    tuner_code = tuner_path.read_text()
    
    if "TimeSeriesSplit" not in tuner_code:
        print("   ❌ TimeSeriesSplit not imported in tuner.py")
        return False
    
    if 'cv_kind == "timeseries"' not in tuner_code and 'kind == "timeseries"' not in tuner_code:
        print("   ❌ TimeSeriesSplit logic not found in tuner.py")
        return False
    
    print("   ✅ tuner.py contains TimeSeriesSplit")
    
    # Check outliers.py
    outliers_path = Path("src/preprocessing/outliers.py")
    if not outliers_path.exists():
        print("   ❌ outliers.py not found!")
        return False
    
    outliers_code = outliers_path.read_text()
    
    if "_inliers_zscore_robust" not in outliers_code:
        print("   ⚠️  Robust Z-score not found in outliers.py")
        print("       Consider adding _inliers_zscore_robust function")
    else:
        print("   ✅ Robust outlier detection implemented")
    
    # Check pipeline.py
    pipeline_path = Path("src/preprocessing/pipeline.py")
    if not pipeline_path.exists():
        print("   ❌ pipeline.py not found!")
        return False
    
    pipeline_code = pipeline_path.read_text()
    
    if "find_best_target_transform" not in pipeline_code:
        print("   ⚠️  Auto target transform not found in pipeline.py")
        print("       Consider adding find_best_target_transform function")
    else:
        print("   ✅ Auto target transform implemented")
    
    print("   ✅ Code validation passed!")
    return True

def validate_tests():
    """Valida che i nuovi test siano presenti."""
    print("\n3️⃣ Validating tests...")
    
    tests_dir = Path("tests")
    
    # Check temporal split test
    temporal_test = tests_dir / "test_temporal_split_fix.py"
    if temporal_test.exists():
        test_code = temporal_test.read_text()
        if "TimeSeriesSplit" in test_code:
            print("   ✅ TimeSeriesSplit test found")
        else:
            print("   ⚠️  TimeSeriesSplit test incomplete")
    else:
        print("   ⚠️  test_temporal_split_fix.py not found")
        print("       Consider adding tests for TimeSeriesSplit")
    
    # Check robust outlier test
    outlier_test = tests_dir / "test_robust_outliers.py"
    if outlier_test.exists():
        print("   ✅ Robust outlier test found")
    else:
        print("   ⚠️  test_robust_outliers.py not found")
        print("       Consider adding tests for robust outlier detection")
    
    print("   ✅ Test validation passed!")
    return True

def main():
    print("=" * 60)
    print("🔍 VALIDATING CRITICAL FIXES")
    print("=" * 60)
    
    all_passed = True
    
    # Run validations
    all_passed &= validate_config()
    all_passed &= validate_code()
    all_passed &= validate_tests()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED!")
        print("=" * 60)
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("=" * 60)
        print("\nRefer to the output above for specific issues.")
        print("See QUICK_FIXES_CODICE.md for implementation details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

**Eseguire**:
```bash
python scripts/validate_fixes.py
```

---

## 📋 Checklist Implementazione

### Prima di Iniziare
- [ ] Backup del codice attuale (`git commit -a -m "Backup before critical fixes"`)
- [ ] Creare branch per le modifiche (`git checkout -b fix/critical-issues`)

### Implementazione
- [ ] ✅ Fix 1: TimeSeriesSplit per CV
  - [ ] Modificare `config/config.yaml`
  - [ ] Modificare `src/training/tuner.py`
  - [ ] Aggiungere test `tests/test_temporal_split_fix.py`
  - [ ] Eseguire test: `pytest tests/test_temporal_split_fix.py -v`

- [ ] ✅ Fix 2: Robust Outlier Detection
  - [ ] Modificare `src/preprocessing/outliers.py`
  - [ ] Aggiungere `_inliers_zscore_robust`
  - [ ] Aggiungere `_ensemble_inliers_robust`
  - [ ] Aggiungere test `tests/test_robust_outliers.py`
  - [ ] Eseguire test: `python tests/test_robust_outliers.py`

- [ ] ✅ Fix 3: Auto Target Transform
  - [ ] Modificare `src/preprocessing/pipeline.py`
  - [ ] Aggiungere `find_best_target_transform`
  - [ ] Aggiornare `run_preprocessing`
  - [ ] Test manuale con dataset reale

### Validazione
- [ ] Eseguire script validazione: `python scripts/validate_fixes.py`
- [ ] Eseguire suite test completa: `pytest tests/ -v`
- [ ] Test integrazione: `python main.py --config config/config.yaml --steps preprocessing`
- [ ] Verificare metriche migliorate

### Commit e Deploy
- [ ] Commit: `git commit -m "feat: implement critical fixes (TimeSeriesSplit, robust outliers, auto transform)"`
- [ ] Push: `git push origin fix/critical-issues`
- [ ] Creare Pull Request con descrizione dettagliata
- [ ] Code review
- [ ] Merge a main

---

## 🎯 Risultati Attesi

Dopo l'implementazione di questi fix:

| Metrica | Prima | Dopo | Miglioramento |
|---------|-------|------|---------------|
| Test R² | 0.75-0.78 | 0.78-0.82 | +3-5% |
| Test RMSE | 18,000-20,000 | 15,000-17,000 | -15-20% |
| Outlier Detection Accuracy | 60-70% | 80-90% | +20-30% |
| Target Skewness | 5.16 | 0.5-1.5 | -70-90% |

**Tempo Implementazione Stimato**: 1-2 giorni

---

## ❓ Troubleshooting

### TimeSeriesSplit dà errore "not enough splits"
```python
# Se il dataset è molto piccolo:
cv_when_no_val:
  n_splits: 3  # Ridurre a 3 invece di 5
```

### Import Error: scipy.stats
```bash
pip install scipy --upgrade
```

### Test falliscono per "module not found"
```bash
# Assicurarsi che src/ sia in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
pytest tests/
```

---

**Domande?** Aprire issue su GitHub o contattare il team ML.

**Prossimo Step**: Dopo questi fix, procedere con Feature Engineering Avanzato (vedi FEEDBACK_ESECUTIVO.md Sprint 2).

