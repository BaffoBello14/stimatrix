# ⚡ QUICK FIXES IMMEDIATI - Da Applicare ORA

**Obiettivo**: Migliorare performance da MAPE 58% → <50% in 30 minuti

---

## 🎯 MODIFICHE DA FARE

### **1. Filtro Outlier Più Aggressivo**

**File**: `config/config_optimized.yaml`

**Trova questa sezione:**
```yaml
# Data filtering for experimentation
data_filters:
  # enabled: false  # Se false, nessun filtro applicato
  experiment_name: null
  description: null
  
  # Temporal filtering
  anno_min: null
  anno_max: null
```

**Modifica in:**
```yaml
# Data filtering for experimentation
data_filters:
  enabled: true  # ✅ ATTIVATO
  experiment_name: "outlier_removal_v1"
  description: "Rimozione outlier estremi: prezzo 20k-500k, superficie 10-300mq"
  
  # Temporal filtering
  anno_min: null
  anno_max: null
  
  # 🔥 NUOVO: Price filtering (rimuove outlier estremi)
  prezzo_min: 20000   # Rimuovi prezzi <20k€ (errori/outlier)
  prezzo_max: 500000  # Rimuovi prezzi >500k€ (outlier lusso)
  
  # 🔥 NUOVO: Superficie filtering
  superficie_min: 10  # Rimuovi <10mq (errori)
  superficie_max: 300 # Rimuovi >300mq (outlier)
```

---

### **2. Aumentare Contamination Outlier Detection**

**File**: `config/config_optimized.yaml`

**Trova:**
```yaml
outliers:
  enabled: true
  methods: ['iso_forest']
  iqr_multiplier: 2.5
  z_score_threshold: 3.5
  iso_forest_contamination: 0.08  # ✅ AUMENTATO da 0.05
```

**Modifica in:**
```yaml
outliers:
  enabled: true
  methods: ['iso_forest']
  iqr_multiplier: 2.0          # ✅ RIDOTTO da 2.5 (più aggressivo)
  z_score_threshold: 3.0       # ✅ RIDOTTO da 3.5 (più aggressivo)
  iso_forest_contamination: 0.15  # ✅ AUMENTATO da 0.08 (più aggressivo)
```

---

### **3. Cambio Trasformazione Target: log → yeojohnson**

**File**: `config/config_optimized.yaml`

**Trova:**
```yaml
target:
  column_candidates: ['AI_Prezzo_Ridistribuito']
  transform: 'log'
```

**Modifica in:**
```yaml
target:
  column_candidates: ['AI_Prezzo_Ridistribuito']
  transform: 'yeojohnson'  # ✅ CAMBIATO da 'log' → gestisce meglio range estremi
```

---

### **4. Aumentare Regularization CatBoost**

**File**: `config/config_optimized.yaml`

**Trova:**
```yaml
catboost:
  enabled: true
  base_params:
    iterations: 1500
    depth: 6
    learning_rate: 0.05
    l2_leaf_reg: 3.0  # ✅ AUMENTATO da 1.0
```

**Modifica in:**
```yaml
catboost:
  enabled: true
  base_params:
    iterations: 1500
    depth: 5          # ✅ RIDOTTO da 6 (meno overfit)
    learning_rate: 0.05
    l2_leaf_reg: 6.0  # ✅ AUMENTATO da 3.0 (più regularization)
```

**E anche nel search_space:**

**Trova:**
```yaml
  search_space:
    depth:
      low: 4
      high: 8
    l2_leaf_reg:
      low: 1.0
      high: 5.0
```

**Modifica in:**
```yaml
  search_space:
    depth:
      low: 3          # ✅ RIDOTTO da 4
      high: 7         # ✅ RIDOTTO da 8
    l2_leaf_reg:
      low: 3.0        # ✅ AUMENTATO da 1.0
      high: 10.0      # ✅ AUMENTATO da 5.0
```

---

### **5. Ridurre Overfit XGBoost**

**File**: `config/config_optimized.yaml`

**Trova:**
```yaml
xgboost:
  enabled: true
  base_params:
    n_estimators: 1500
    max_depth: 5
```

**Modifica in:**
```yaml
xgboost:
  enabled: true
  base_params:
    n_estimators: 1000  # ✅ RIDOTTO da 1500 (early stop)
    max_depth: 4        # ✅ RIDOTTO da 5 (meno overfit)
```

---

### **6. Aumentare Min Samples Leaf RF**

**File**: `config/config_optimized.yaml`

**Trova:**
```yaml
rf:
  enabled: true
  base_params:
    n_estimators: 500
    max_depth: 20
    min_samples_leaf: 5
```

**Modifica in:**
```yaml
rf:
  enabled: true
  base_params:
    n_estimators: 500
    max_depth: 15       # ✅ RIDOTTO da 20
    min_samples_leaf: 10  # ✅ AUMENTATO da 5 (più regularization)
```

---

## 📋 CHECKLIST APPLICAZIONE

```bash
# 1. Backup config attuale
cp config/config_optimized.yaml config/config_optimized_backup.yaml

# 2. Applica modifiche sopra
#    (edita config/config_optimized.yaml manualmente)

# 3. Verifica modifiche
grep "prezzo_min\|prezzo_max\|yeojohnson\|contamination: 0.15\|l2_leaf_reg: 6.0" config/config_optimized.yaml

# Output atteso:
#   prezzo_min: 20000
#   prezzo_max: 500000
#   transform: 'yeojohnson'
#   iso_forest_contamination: 0.15
#   l2_leaf_reg: 6.0

# 4. Esegui training
python run_fixed_training.py

# 5. Confronta risultati
#    Target: MAPE < 50%, R² > 0.75, no outlier estremi
```

---

## 📊 IMPATTO ATTESO

| Modifica | Impatto su MAPE | Impatto su R² | Impatto Overfit |
|----------|-----------------|---------------|-----------------|
| Filtro prezzo 20k-500k | **-8 to -12%** | **+0.03 to +0.05** | Neutro |
| Filtro superficie 10-300mq | **-3 to -5%** | **+0.02 to +0.03** | Neutro |
| Contamination 0.15 | **-5 to -8%** | **+0.02 to +0.04** | **-0.02 gap R²** |
| Transform yeojohnson | **-3 to -6%** | **+0.02 to +0.03** | Neutro |
| L2_reg CatBoost 6.0 | **-2 to -4%** | **-0.01 to +0.01** | **-0.03 gap R²** |
| Depth reduction | **-1 to -3%** | **-0.01 to 0** | **-0.02 gap R²** |
| **TOTALE** | **-22 to -38%** | **+0.07 to +0.14** | **-0.05 to -0.08 gap** |

### **Risultati Attesi:**
```
PRIMA (Attuale):
  R²:          0.7364
  RMSE:        36,768 €
  MAPE:        58.10%
  Gap R²:      0.1166

DOPO (Quick Fixes):
  R²:          0.80 - 0.85  (✅ +0.07-0.11)
  RMSE:        30,000 - 33,000 €  (✅ -3,000 - -6,000€)
  MAPE:        38% - 45%  (✅ -13% - -20%)
  Gap R²:      0.06 - 0.08  (✅ -0.04 - -0.06)
```

---

## ⏱️ TIMELINE

| Task | Tempo | Persona |
|------|-------|---------|
| Backup config | 1 min | You |
| Applicare modifiche | 10 min | You |
| Verifica config | 2 min | You |
| Eseguire training | 15-30 min | Script |
| Analizzare risultati | 5-10 min | You |
| **TOTALE** | **~35-55 min** | |

---

## 🚨 ATTENZIONE

### **Modifiche Applicate:**

- ✅ **data_filters.enabled = true** → Dataset sarà filtrato!
- ✅ **prezzo_min/max** → Rimuoverà ~5-10% dataset (outlier)
- ✅ **superficie_min/max** → Rimuoverà ~2-5% dataset (outlier)
- ✅ **trasformazione yeojohnson** → Preprocessing diverso

### **Risultato Finale:**

- Dataset più pulito (~10-15% dati rimossi)
- Modello più robusto (meno overfit)
- Performance migliori (MAPE -13% to -20%)

### **Se Risultati Peggiori:**

```bash
# Rollback immediato
cp config/config_optimized_backup.yaml config/config_optimized.yaml
```

---

## 📝 NOTE

- Queste sono **quick fixes** → Non risolvono problema MAPE alto completamente
- Target finale production-ready: **MAPE <20%** → Servirà FASE 2 (feature engineering)
- Dopo quick fixes → Analizzare nuovi worst predictions / group metrics
- Se MAPE scende <45% → Procedere con FASE 2 (vedi `ANALISI_RISULTATI_POST_CLEANUP.md`)

---

## ✅ DOPO IL TRAINING

1. **Controlla MAPE:**
   - Se <45%: ✅ Successo! Procedi FASE 2
   - Se 45-50%: ⚠️ Parziale, rivedere filtri
   - Se >50%: ❌ Rollback, analizzare

2. **Controlla R²:**
   - Se >0.80: ✅ Ottimo!
   - Se 0.75-0.80: ⚠️ Accettabile
   - Se <0.75: ❌ Problema

3. **Controlla Group Metrics:**
   - Zona C3: R² dovrebbe essere >0 (ora -0.32)
   - Price band: R² dovrebbe essere >0 (ora tutti negativi)

4. **Controlla Worst Predictions:**
   - No più errori >500% (ora 5,768%)
   - No più prezzi <5k€ (ora 1,531€)

---

**PRONTI? Via!** 🚀

**Domande?** Apri `ANALISI_RISULTATI_POST_CLEANUP.md` per dettagli tecnici completi.
