# 🔧 Preprocessing Fixes A & B

## 🎯 **Problema Identificato**

Durante l'analisi del preprocessing, sono stati identificati **due problemi critici** che causavano:
1. **Profile TREE**: Troppo aggressivo (157 features finali, troppe poche!)
2. **Profile CATBOOST**: Nessun controllo su extreme high-cardinality (rischio overfitting)

---

## 📊 **Analisi del Problema Originale**

### **Profile TREE - Situazione Precedente**:

```
Start: ~455 features

1. Encoding:
   - Drop 16 high-cardinality (>200)          → -16
   - OHE 83 cols → 245 binary features        → +162 (245-83)
   - Target/Frequency/Ordinal encoding        → 0 (same cols)

2. Drop non-descriptive:                      → -58

3. Correlation Pruning (❌ PROBLEMA):
   X_tr_num = X_tr.select_dtypes(include=[np.number])
   # ❌ INCLUDE tutte le OHE features (245 binaries)!
   remove_highly_correlated(X_tr_num, 0.98)   → -157 dropped
   # ❌ Molte OHE features vengono droppate!

RISULTATO: 157 features finali (TROPPO POCHE!)
```

### **Profile CATBOOST - Situazione Precedente**:

```
Start: ~455 features

1. Nessun encoding:
   - Mantiene TUTTE le categoriche native     → 0

2. Drop non-descriptive:                      → -58

3. Correlation Pruning (solo numeriche):
   X_tr_num = X_tr.select_dtypes(include=[np.number])
   # ✅ Solo vere numeriche (categoriche sono OBJECT)
   remove_highly_correlated(X_tr_num, 0.98)   → -61 dropped

4. ❌ PROBLEMA: Nessun controllo extreme high-cardinality!
   - Colonne con >1000 unique vengono mantenute
   - Rischio: overfitting, slowdown, memory overhead

RISULTATO: 351 features finali (troppe categoriche raw)
```

---

## ✅ **Fix A: Escludi OHE dal Correlation Pruning (Profile TREE)**

### **Problema**:
- Le OHE features sono **binarie (0/1)** → numeriche
- Entrano nel correlation pruning pensato per continue numerics
- Vengono droppate se correlazione >0.98 con altre features
- **Perdi informazione categorica importante**!

**Esempio**:
```
Categoria "AI_ZonaOmi": 9 zone → OHE crea 9 colonne binarie
Se 2 zone hanno distribuzioni simili → corr >0.98 → UNA VIENE DROPPATA
Ma erano DUE ZONE DIVERSE! Informazione persa.
```

### **Soluzione Implementata**:

**File**: `src/preprocessing/pipeline.py` (linee 788-817)

```python
# FIX A: Correlation pruning ONLY on continuous numeric features (exclude OHE)
# OHE features are binary (0/1) and should NOT be pruned based on correlation
# They represent categorical information and high correlation is expected

# Identify OHE columns (pattern: __ohe_ or ohe_)
ohe_cols = [c for c in X_tr.columns if '__ohe_' in c or c.startswith('ohe_')]
logger.info(f"[tree] Identified {len(ohe_cols)} OHE features to preserve")

# Correlation pruning ONLY on numeric features that are NOT OHE
X_tr_num = X_tr.select_dtypes(include=[np.number]).drop(columns=ohe_cols, errors="ignore")
X_tr_num_pruned, dropped_corr = remove_highly_correlated(X_tr_num, threshold=corr_thr)
logger.info(f"[tree] Pruning correlazioni numeriche (excluding OHE): {len(dropped_corr)} dropped")

# Recombine: pruned numeric + ALL OHE + other encoded features
X_tr_final = pd.concat([X_tr_num_pruned, X_tr[ohe_cols], X_tr_other], axis=1)
```

### **Impatto Atteso**:

**Prima**:
```
Features dopo OHE: ~400 (245 OHE + numeriche originali)
Correlation pruning: -157 (INCLUDE molte OHE)
Risultato: 157 features finali
```

**Dopo Fix A**:
```
Features dopo OHE: ~400 (245 OHE + numeriche originali)
OHE escluse da pruning: 245 preservate ✅
Correlation pruning: ~60-80 (solo vere numeriche)
Risultato: ~280-320 features finali (+100-160 rispetto a prima!)
```

---

## ✅ **Fix B: Drop Extreme High-Cardinality (Profile CATBOOST)**

### **Problema**:
- CatBoost gestisce bene high-cardinality, **MA non ALL-cardinality**
- Colonne con >500-1000 unique values causano:
  1. **Overfitting**: Memorizzazione invece di generalizzazione
  2. **Slowdown**: Training più lento
  3. **Memory overhead**: Consumo RAM aumentato
  4. **Poor generalization**: Scarsa performance su unseen categories

### **Soluzione Implementata**:

**File**: `src/preprocessing/pipeline.py` (linee 848-870)

```python
# FIX B: Drop extreme high-cardinality categorical features
# Even CatBoost has limits - extreme cardinality (>500-1000) can cause:
# 1. Overfitting (memorization instead of generalization)
# 2. Training slowdown and memory overhead
# 3. Poor performance on unseen categories

extreme_card_threshold = int(config.get("encoding", {}).get("catboost_max_cardinality", 500))
cat_cols = X_tr.select_dtypes(include=['object', 'category']).columns
extreme_high_card = []

for col in cat_cols:
    nunique = X_tr[col].nunique(dropna=True)
    if nunique > extreme_card_threshold:
        extreme_high_card.append(col)
        logger.warning(
            f"[catboost] Dropping extreme high-cardinality column: {col} "
            f"({nunique} unique values > {extreme_card_threshold} threshold)"
        )

if extreme_high_card:
    X_tr = X_tr.drop(columns=extreme_high_card)
    X_te = X_te.drop(columns=extreme_high_card, errors="ignore")
    if X_va is not None:
        X_va = X_va.drop(columns=extreme_high_card, errors="ignore")
    logger.info(f"[catboost] Dropped {len(extreme_high_card)} extreme high-cardinality columns")
```

### **Config Aggiunta**:

**Files**: `config/config.yaml`, `config/config_fast.yaml`

```yaml
encoding:
  catboost_max_cardinality: 500  # FIX B: Max cardinality for CatBoost (drop if >500)
```

**Valore configurabile**:
- `500`: Default conservativo (buon bilanciamento)
- `1000`: Più permissivo (per dataset molto grandi)
- `300`: Più aggressivo (per dataset piccoli o ridurre overfitting)

### **Impatto Atteso**:

**Prima**:
```
Categoriche native: ~100 colonne (alcune con >1000 unique)
Nessun drop: tutte mantenute
Risultato: 351 features (alcune problematiche)
```

**Dopo Fix B**:
```
Categoriche native: ~100 colonne
Drop extreme (>500): ~5-10 colonne rimosse ✅
Risultato: ~340-345 features (più pulite)
```

---

## 📊 **Confronto Prima/Dopo**

### **Profile TREE**:

| Metric | Prima Fix A | Dopo Fix A | Delta |
|--------|-------------|------------|-------|
| Features start | ~455 | ~455 | 0 |
| Drop high-card | -16 | -16 | 0 |
| OHE expansion | +162 | +162 | 0 |
| Drop non-desc | -58 | -58 | 0 |
| **Correlation pruning** | **-157** ❌ | **~70** ✅ | **+87** |
| **RISULTATO** | **157** ❌ | **~300** ✅ | **+143** |

### **Profile CATBOOST**:

| Metric | Prima Fix B | Dopo Fix B | Delta |
|--------|-------------|------------|-------|
| Features start | ~455 | ~455 | 0 |
| **Drop extreme high-card** | **0** ❌ | **~10** ✅ | **-10** |
| Drop non-desc | -58 | -58 | 0 |
| Correlation pruning | -61 | -61 | 0 |
| **RISULTATO** | **351** | **~340** ✅ | **-11** |

### **Differenza Finale**:

- **Prima**: Tree=157, CatBoost=351 → **Δ = 194 features** ❌ (troppo grande!)
- **Dopo**: Tree=~300, CatBoost=~340 → **Δ = ~40 features** ✅ (ragionevole!)

---

## 🎯 **Vantaggi dei Fix**

### **Fix A (OHE preservate)**:
- ✅ **+140-160 features** nel profile TREE
- ✅ **Mantiene informazione categorica** (zone, tipologie, ecc.)
- ✅ **Migliore performance** attesa per XGBoost/LightGBM/RF
- ✅ **Correlation pruning** solo su vere numeriche (corretto!)

### **Fix B (Drop extreme high-card)**:
- ✅ **Previene overfitting** su colonne quasi-uniche
- ✅ **Training più veloce** (meno overhead)
- ✅ **Migliore generalizzazione** su dati non visti
- ✅ **Riduce memory footprint**

### **Combinati**:
- ✅ **Profili più bilanciati** (~300 vs ~340 invece di 157 vs 351)
- ✅ **Ensemble possibili** su stesso profile senza mismatch estremi
- ✅ **Preprocessing più corretto** e robusto

---

## 🧪 **Testing e Verifica**

### **Come Testare**:

```bash
# Esegui preprocessing completo
python main.py --mode preprocessing --config config/config_fast.yaml

# Verifica nei log:
# 1. Fix A:
#    "[tree] Identified 245 OHE features to preserve"
#    "[tree] Pruning correlazioni numeriche (excluding OHE): 70 dropped"
#
# 2. Fix B:
#    "[catboost] Dropping extreme high-cardinality column: X (1234 unique)"
#    "[catboost] Dropped 8 extreme high-cardinality columns"
```

### **Verifiche nei File Salvati**:

```python
import pandas as pd

# Profile TREE
X_train_tree = pd.read_parquet("data/preprocessed/X_train_tree.parquet")
print(f"Tree features: {X_train_tree.shape[1]}")  # Atteso: ~280-320

# Profile CATBOOST
X_train_catboost = pd.read_parquet("data/preprocessed/X_train_catboost.parquet")
print(f"CatBoost features: {X_train_catboost.shape[1]}")  # Atteso: ~340-345

# Differenza
print(f"Delta: {abs(X_train_catboost.shape[1] - X_train_tree.shape[1])}")  # Atteso: ~40-60
```

---

## ⚙️ **Configurazione**

### **Parametro Aggiunto**:

```yaml
# config/config.yaml e config/config_fast.yaml
encoding:
  catboost_max_cardinality: 500  # Soglia per extreme high-cardinality
```

### **Tuning del Parametro**:

**Linee guida**:
- **Dataset piccolo (<5000 righe)**: `300-400` (più conservativo)
- **Dataset medio (5000-50000 righe)**: `500-700` (default)
- **Dataset grande (>50000 righe)**: `700-1000` (più permissivo)

**Trade-off**:
- **Valore basso** (300): Meno overfitting, più generalizzazione, meno informazione
- **Valore alto** (1000): Più informazione, rischio overfitting, training più lento

---

## 🔍 **Pattern OHE Identificati**

Il Fix A identifica OHE features con questi pattern:
- `feature__ohe_category` (sklearn OneHotEncoder con sparse=False)
- `ohe_feature_category` (altri encoder)

Se usi un encoder diverso con pattern diverso, aggiorna la riga 794:
```python
ohe_cols = [c for c in X_tr.columns if '__ohe_' in c or c.startswith('ohe_') or 'YOUR_PATTERN' in c]
```

---

## 📚 **Riferimenti**

### **Codice Modificato**:
- `src/preprocessing/pipeline.py` (linee 788-817, 848-870)
- `config/config.yaml` (linea 122)
- `config/config_fast.yaml` (linea 115)

### **Related Issues**:
- Profile TREE: Troppo aggressivo (157 features)
- Profile CATBOOST: Nessun controllo extreme high-cardinality
- Ensemble: Mismatch troppo grande (194 features differenza)

---

## ✅ **Checklist Verifica**

Dopo il training, verifica:

- [ ] Log: "[tree] Identified N OHE features to preserve"
- [ ] Log: "[tree] Pruning correlazioni numeriche (excluding OHE): M dropped"
- [ ] Log: "[catboost] Dropped K extreme high-cardinality columns"
- [ ] Tree features: ~280-320 (non più 157!)
- [ ] CatBoost features: ~340-345 (non più 351)
- [ ] Delta: ~40-60 features (non più 194!)
- [ ] Performance: Test R² invariato o migliorato
- [ ] Training time: Leggermente più veloce (meno overfitting)

---

**Data**: 2025-11-13  
**Branch**: `cursor/code-review-for-data-leakage-e943`  
**Status**: ✅ Implemented & Documented
