# 🔒 Data Leakage Audit Report

**Data**: 2025-11-13  
**Progetto**: Real Estate Price Prediction ML Pipeline  
**Revisore**: AI Code Auditor

---

## 📋 Executive Summary

Ho condotto un'analisi approfondita del codebase per identificare potenziali problemi di data leakage. Il codice presenta **una buona struttura generale** con diversi pattern anti-leakage implementati correttamente. Tuttavia, ho identificato **alcuni punti critici** che richiedono attenzione.

### ✅ **Punti di Forza**

1. **Temporal Split corretto**: Lo split temporale avviene PRIMA di qualsiasi feature engineering
2. **Fit/Transform pattern**: Implementato correttamente per encoder, imputer, scaler
3. **Test coverage**: Esistono test specifici per il data leakage nell'encoding
4. **Documentazione**: Commenti espliciti su "LEAK-FREE" in punti critici

### ⚠️ **Problemi Identificati**

- **CRITICO**: Feature contestuali rimosse ma potrebbero tornare (righe commentate)
- **MEDIO**: Potenziale leakage nel tuning quando non c'è validation set
- **BASSO**: Outlier detection usa tutto il training set (corretto, ma da monitorare)

---

## 🔍 Analisi Dettagliata

### 1. ✅ **Temporal Split - CORRETTO**

**File**: `src/preprocessing/pipeline.py` (linee 377-394)

```python
# Temporal split FIRST to avoid leakage (contextual features AFTER split!)
split_cfg = TemporalSplitConfig(...)
train_df, val_df, test_df = temporal_split_3way(Xy_full, split_cfg)
```

**Verifica**:
- ✅ Split avviene PRIMA di imputation, encoding, scaling
- ✅ Mantiene ordine cronologico (no shuffle)
- ✅ Usa frazione configurabile o data fissa
- ✅ Validation set opzionale

**Raccomandazione**: ✨ **Nessuna azione richiesta**

---

### 2. ✅ **Encoding - CORRETTO**

**File**: `src/preprocessing/encoders.py`

**Verifica**:
- ✅ `plan_encodings()`: Pianifica solo su train
- ✅ `fit_apply_encoders()`: Fit solo su train (linea 113-255)
- ✅ `transform_with_encoders()`: Applica encoder fittati (linea 258-346)
- ✅ Gestione corretta categorie unseen in test (handle_unknown='ignore')
- ✅ Target encoding con smoothing per evitare overfitting

**Test Coverage**: `tests/test_encoding_no_leakage.py` (270 linee, 9 test)

**Raccomandazione**: ✨ **Nessuna azione richiesta**

---

### 3. ✅ **Imputation - CORRETTO**

**File**: `src/preprocessing/imputation.py`

**Verifica**:
- ✅ `fit_imputers()`: Calcola statistiche solo su train (linea 135-136)
- ✅ `transform_with_imputers()`: Applica statistiche pre-calcolate (linea 139-140)
- ✅ Group-by imputation usa statistiche del train
- ✅ Fallback a statistiche globali per gruppi non visti

**Raccomandazione**: ✨ **Nessuna azione richiesta**

---

### 4. ⚠️ **Contextual Features - PROBLEMATICO (RISOLTO MA DA MONITORARE)**

**File**: `src/preprocessing/contextual_features.py`

#### 🟢 **Corretto ora**:
```python
# Fit ONLY on train, transform all splits
stats = fit_contextual_features(train_df, target_col=target_col)
train_out = transform_contextual_features(train_df, stats, ...)
val_out = transform_contextual_features(val_df, stats, ...)
test_out = transform_contextual_features(test_df, stats, ...)
```

#### 🔴 **Codice commentato problematico** (linee 161-165, 186, 208-209, 226):

Ho trovato **codice commentato** che conteneva feature problematiche:

```python
# ❌ REMOVED: Derived features that require target instance (not usable in production)
# - price_vs_zone_mean_ratio
# - price_vs_zone_median_ratio
# - price_zone_zscore
# - price_zone_iqr_position
# - price_zone_range_position
# ❌ REMOVED: price_vs_type_zone_mean (requires target instance)
# ❌ REMOVED: prezzo_mq (requires target instance)
# ❌ REMOVED: prezzo_mq_vs_zone (requires target instance)
# ❌ REMOVED: price_vs_temporal_mean (requires target instance)
```

**Problema**: Queste feature richiedono il **target dell'istanza corrente** per essere calcolate, causando:
1. **Data leakage**: Il modello "vede" il target durante il training
2. **Inutilizzabilità in produzione**: Non possiamo calcolare queste feature senza conoscere il prezzo

#### ✅ **Feature mantenute (corrette)**:
```python
# ✅ KEEP: type_zone_rarity (uses count, not target instance)
df['type_zone_rarity'] = 1.0 / (df['type_zone_count'] + 1)

# ✅ KEEP: surface ratios (no target needed)
df['surface_vs_zone_mean'] = df[surface_col] / (df['zone_surface_mean'] + 1e-8)
df['surface_vs_type_zone_mean'] = df[surface_col] / (df['type_zone_surface_mean'] + 1e-8)
```

**Raccomandazione**: 
- ✅ **Codice attuale è corretto**
- ⚠️ **ELIMINARE il codice commentato** per evitare reintroduzioni accidentali
- 📝 **Documentare chiaramente** quali feature sono permesse e quali no

---

### 5. ✅ **Target Transformation - CORRETTO**

**File**: `src/preprocessing/target_transforms.py`

**Verifica**:
- ✅ Box-Cox/Yeo-Johnson: Lambda stimato su train, applicato a test (linee 454-479)
- ✅ Log transform: Stessi parametri per train/test
- ✅ Inverse transform corretto per predictions

**File**: `src/preprocessing/pipeline.py` (linee 446-492)

```python
# Apply target transformation (fit on train, transform test/val with same params)
y_train, transform_metadata = apply_target_transform_from_config(config, y_train)

# For Box-Cox/Yeo-Johnson: use lambda fitted on train for test/val
if transform_type == "boxcox":
    lambda_val = float(transform_metadata.get("lambda"))
    y_test = boxcox_transform(y_test.to_numpy(), lambda_val, shift)
```

**Raccomandazione**: ✨ **Nessuna azione richiesta**

---

### 6. ✅ **Scaling e PCA - CORRETTO**

**File**: `src/preprocessing/transformers.py` (linee 149-179)

**Verifica**:
- ✅ Scaler fit solo su train (linea 163, 167)
- ✅ PCA fit solo su train (linea 172)
- ✅ Transform applicato a test con oggetti fittati
- ✅ Winsorization bounds calcolati solo su train (linee 131-136)

**Raccomandazione**: ✨ **Nessuna azione richiesta**

---

### 7. ✅ **Outlier Detection - CORRETTO**

**File**: `src/preprocessing/pipeline.py` (linee 411-429)

```python
# Outlier detection ONLY on train target (optionally per category)
before = len(train_df)
inliers_mask = detect_outliers(train_df, target_col, out_cfg)
train_df = train_df.loc[inliers_mask].copy()
```

**Verifica**:
- ✅ Applicato SOLO al training set
- ✅ Test set non modificato (corretto!)
- ✅ Group-by outlier detection per categoria
- ✅ Random state configurabile per IsolationForest

**Nota**: Questo è il comportamento corretto. Gli outlier vengono rimossi solo dal training per migliorare il fitting, ma il test set rimane intatto per una valutazione realistica.

**Raccomandazione**: ✨ **Nessuna azione richiesta**

---

### 8. ⚠️ **Tuning - POTENZIALE LEAKAGE (MEDIO)**

**File**: `src/training/tuner.py` (linee 176-195)

#### 🟡 **Scenario problematico**: Quando non c'è validation set

```python
if X_val is None or y_val is None:
    # Use temporal split instead of random split to avoid data leakage
    # Maintain chronological order for time-series data
    split_point = int(len(X_train) * tuning_split_fraction)
    X_tr = X_train.iloc[:split_point]
    X_va = X_train.iloc[split_point:]
```

**Problema**: 
- Il codice assume che `X_train` sia già ordinato temporalmente
- Se per qualche motivo l'ordine venisse perso (e.g., shuffle accidentale), questo diventerebbe uno split random

**Impatto**: MEDIO
- ✅ Commento esplicito che dice "maintain chronological order"
- ⚠️ Non c'è verifica che i dati siano effettivamente ordinati
- ⚠️ Lo split è semplice (primi N vs ultimi M), senza controllare la temporal key

**Raccomandazione**: 
```python
# SUGGERIMENTO: Aggiungere verifica esplicita
if 'TemporalKey' in X_train.columns:
    assert X_train['TemporalKey'].is_monotonic_increasing, \
        "X_train must be sorted by TemporalKey for temporal split in tuning"
```

---

### 9. ✅ **Cross-Validation - CORRETTO CON RISERVA**

**File**: `src/training/tuner.py` (linee 105-175)

**Verifica**:
- ✅ `TimeSeriesSplit` disponibile per dati temporali
- ✅ KFold con shuffle configurabile
- ⚠️ KFold con shuffle=True potrebbe causare leakage temporale

**Raccomandazione**: 
- Per dati time-series: **Usare sempre `TimeSeriesSplit`**
- KFold shuffle: **Disabilitare** per dati temporali
- Aggiungere warning nel codice se si usa KFold con shuffle su dati temporali

---

### 10. ✅ **Training Finale - CORRETTO**

**File**: `src/training/train.py`

**Verifica**:
- ✅ Dati caricati già processati e splittati (linea 176)
- ✅ Encoder persistiti e riusati per test (linee 646-654)
- ✅ Smearing factor calcolato solo su train per log transform (linee 369-376)
- ✅ Group metrics calcolati su original scale (linee 491-572)

**Raccomandazione**: ✨ **Nessuna azione richiesta**

---

## 🎯 Raccomandazioni Prioritizzate

### 🔴 **CRITICO** (da fare subito)

#### 1. Rimuovere codice commentato problematico
**File**: `src/preprocessing/contextual_features.py`

```python
# ❌ ELIMINARE le righe 161-165, 186, 208-209, 226
# Rimuovere completamente per evitare reintroduzioni accidentali
```

**Azione**:
```bash
# Linee da eliminare:
# - 161-165: price_vs_zone_* features
# - 186: price_vs_type_zone_mean
# - 208-209: prezzo_mq features
# - 226: price_vs_temporal_mean
```

---

### 🟡 **IMPORTANTE** (da fare presto)

#### 2. Aggiungere verifica ordine temporale nel tuning
**File**: `src/training/tuner.py` (dopo linea 177)

```python
# Add temporal order verification
if hasattr(X_train, 'columns') and 'TemporalKey' in X_train.columns:
    if not X_train['TemporalKey'].is_monotonic_increasing:
        raise ValueError(
            "X_train must be sorted by TemporalKey for temporal split in tuning. "
            "Detected non-monotonic TemporalKey sequence."
        )
```

#### 3. Documentare regole per contextual features
**File**: `docs/CONTEXTUAL_FEATURES_GUIDELINES.md` (nuovo)

```markdown
# Contextual Features Guidelines

## ✅ Allowed Features (LEAK-FREE)
- Aggregated statistics from TRAINING data only
- Features that don't require target of current instance
- Examples:
  - zone_price_mean, zone_price_median (from train)
  - type_zone_count, type_zone_rarity
  - surface_vs_zone_mean (ratio of surfaces, no prices)

## ❌ Prohibited Features (CAUSE LEAKAGE)
- Any feature requiring target of current instance
- Examples:
  - price_vs_zone_mean (needs current price!)
  - prezzo_mq (needs current price!)
  - price_zone_zscore (needs current price!)

## 🔑 Golden Rule
If you can't calculate the feature in production WITHOUT knowing 
the target price, then it's LEAKAGE.
```

---

### 🟢 **BUONA PRATICA** (miglioramenti)

#### 4. Aggiungere test per contextual features
**File**: `tests/test_contextual_features_no_leakage.py` (nuovo)

```python
def test_contextual_features_no_target_leakage():
    """Test che le contextual features non usino il target dell'istanza corrente."""
    train = pd.DataFrame({
        'AI_ZonaOmi': ['A', 'A', 'B', 'B'],
        'AI_Prezzo_Ridistribuito': [100, 200, 150, 250],
        'AI_Superficie': [50, 100, 75, 125],
    })
    
    stats = fit_contextual_features(train)
    transformed = transform_contextual_features(train, stats)
    
    # Verify no features use current row's target
    assert 'price_vs_zone_mean' not in transformed.columns
    assert 'prezzo_mq' not in transformed.columns
    
    # Verify allowed features exist
    assert 'zone_price_mean' in transformed.columns
    assert 'surface_vs_zone_mean' in transformed.columns
```

#### 5. Aggiungere monitoring per ordine temporale
**File**: `src/preprocessing/pipeline.py` (dopo temporal split)

```python
# Verify temporal order is preserved
if 'TemporalKey' in train_df.columns:
    assert train_df['TemporalKey'].is_monotonic_increasing, \
        "Train set lost temporal order after split!"
    logger.info(f"✅ Temporal order verified: {train_df['TemporalKey'].min()} → {train_df['TemporalKey'].max()}")
```

#### 6. Warning per KFold con shuffle su time-series
**File**: `src/training/tuner.py` (linea 112)

```python
if kind == "kfold":
    if shuffle:
        logger.warning(
            "⚠️  Using KFold with shuffle=True on time-series data may cause "
            "temporal leakage. Consider using TimeSeriesSplit instead."
        )
    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
```

---

## 🧪 Test Coverage Attuali

### ✅ **Test esistenti**
1. `test_encoding_no_leakage.py`: 9 test per encoding
2. `test_preprocessing_pipeline.py`: Test generali preprocessing
3. `test_temporal_split_fix.py`: Test per temporal split
4. `test_target_transforms.py`: Test per trasformazioni target

### ⚠️ **Test mancanti**
1. Contextual features (nessun test specifico per leakage)
2. Tuning temporal split verification
3. Integration test end-to-end per leakage

---

## 📊 Matrice di Rischio

| Componente | Rischio Leakage | Implementazione | Test Coverage | Priorità Fix |
|------------|----------------|-----------------|---------------|--------------|
| Temporal Split | ✅ Basso | ✅ Corretto | ✅ Testato | - |
| Encoding | ✅ Basso | ✅ Corretto | ✅ Testato | - |
| Imputation | ✅ Basso | ✅ Corretto | ⚠️ Parziale | 🟢 Bassa |
| Contextual Features | ⚠️ Medio | ⚠️ Codice commentato | ❌ Non testato | 🔴 Alta |
| Target Transform | ✅ Basso | ✅ Corretto | ✅ Testato | - |
| Scaling/PCA | ✅ Basso | ✅ Corretto | ⚠️ Parziale | 🟢 Bassa |
| Outlier Detection | ✅ Basso | ✅ Corretto | ⚠️ Parziale | - |
| Tuning Split | ⚠️ Medio | ⚠️ No verifica ordine | ❌ Non testato | 🟡 Media |
| Cross-Validation | ⚠️ Medio | ⚠️ KFold shuffle | ⚠️ Parziale | 🟡 Media |
| Training Finale | ✅ Basso | ✅ Corretto | ✅ Testato | - |

---

## 🔍 Checklist Anti-Leakage

### Prima dello Split
- [ ] ✅ Nessuna feature derivata prima dello split temporale
- [ ] ✅ Nessuna aggregazione su tutto il dataset prima dello split
- [ ] ✅ Data filters applicati prima dello split (ok per sperimentazione)

### Durante il Preprocessing
- [ ] ✅ Imputer fittato solo su train
- [ ] ✅ Encoder fittato solo su train
- [ ] ✅ Scaler fittato solo su train
- [ ] ✅ PCA fittato solo su train
- [ ] ✅ Outlier detection solo su train

### Feature Engineering
- [ ] ✅ Contextual stats calcolate solo su train
- [ ] ⚠️ **Rimuovere** feature che usano target dell'istanza corrente
- [ ] ✅ Nessuna feature "dal futuro"

### Training & Tuning
- [ ] ✅ Validation split mantiene ordine temporale
- [ ] ⚠️ **Verificare** ordine temporale nel tuning split
- [ ] ⚠️ **Preferire** TimeSeriesSplit a KFold per time-series

### Evaluation
- [ ] ✅ Test set mai usato per fit/tuning
- [ ] ✅ Metriche calcolate su original scale
- [ ] ✅ Group metrics non leakano informazioni

---

## 📈 Metriche di Qualità del Codice

### Code Quality Score: **8.5/10** 🎯

**Breakdown**:
- Architecture: 9/10 ✅
- Test Coverage: 7/10 ⚠️
- Documentation: 8/10 ✅
- Anti-Leakage Patterns: 9/10 ✅
- Code Cleanliness: 7/10 ⚠️ (codice commentato)

---

## 🎓 Best Practices Applicate

### ✅ **Cosa il progetto fa bene**

1. **Separation of Concerns**: Preprocessing completamente separato da training
2. **Fit/Transform Pattern**: Implementato coerentemente in tutto il codebase
3. **Temporal Awareness**: Split temporale corretto per time-series
4. **Artifacts Persistence**: Encoder, scaler, imputer salvati per inference
5. **Test-Driven**: Test specifici per data leakage
6. **Documentation**: Commenti espliciti su leak-free sections

### 🎯 **Cosa può migliorare**

1. **Code Hygiene**: Rimuovere codice commentato problematico
2. **Defensive Programming**: Aggiungere assert per verificare assumptions
3. **Test Coverage**: Più test per contextual features
4. **Warning System**: Alert quando si usano pattern rischiosi (KFold shuffle)

---

## 🚀 Action Plan

### Fase 1: Immediate (questa settimana)
1. ✅ Completare questo audit report
2. 🔴 Rimuovere codice commentato in `contextual_features.py`
3. 🟡 Aggiungere verifica ordine temporale in `tuner.py`

### Fase 2: Short-term (prossime 2 settimane)
4. 🟡 Creare `CONTEXTUAL_FEATURES_GUIDELINES.md`
5. 🟢 Aggiungere test per contextual features
6. 🟢 Aggiungere warning per KFold shuffle

### Fase 3: Long-term (prossimo mese)
7. 🟢 Integration test end-to-end per leakage
8. 🟢 Monitoring temporale in pipeline
9. 🟢 Documentazione completa anti-leakage patterns

---

## 📝 Conclusioni

### Stato Generale: **BUONO** ✅

Il progetto dimostra una **solida comprensione** dei rischi di data leakage e implementa correttamente la maggior parte dei pattern anti-leakage. I problemi identificati sono **gestibili** e principalmente riguardano:

1. **Code hygiene** (codice commentato)
2. **Defensive programming** (verifiche esplicite)
3. **Test coverage** (alcune aree non testate)

### Rischio Complessivo: **BASSO-MEDIO** 🟡

- **Non ci sono leakage attivi** nel codice in produzione
- Il codice commentato rappresenta un **rischio potenziale** se reintrodotto
- Alcuni pattern potrebbero beneficiare di **verifiche più stringenti**

### Prossimi Passi

1. **Immediate**: Applicare fix critici (rimozione codice commentato)
2. **Short-term**: Migliorare test coverage e documentazione
3. **Long-term**: Implementare monitoring e defensive programming

---

**Report generato da**: AI Code Auditor  
**Data**: 2025-11-13  
**Versione**: 1.0  
**Stato**: ✅ Review Completo

---

## 📚 Risorse Aggiuntive

### Letture Consigliate
- [Preventing Data Leakage in ML](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [Target Encoding Best Practices](https://maxhalford.github.io/blog/target-encoding/)

### Tools
- `sklearn.model_selection.TimeSeriesSplit`: Corretto per time-series
- `category_encoders.TargetEncoder`: Con smoothing per evitare overfitting
- Optuna: Per tuning con temporal-aware splits

---

**Fine Report** 🏁
