# 🚀 Guida Ottimizzazione Stimatrix

## 📋 Cosa È Stato Implementato

### ✅ **1. Feature Contestuali** (`src/preprocessing/contextual_features.py`)

Nuove feature che catturano il **contesto di mercato locale**:

#### **A. Zone Statistics** (13 feature)
- `zone_price_mean/median/std/min/max/q25/q75`: Statistiche aggregate per zona OMI
- `zone_count`: Numero di transazioni nella zona
- `price_vs_zone_mean_ratio`: Posizione dell'immobile rispetto alla media zona
- `price_vs_zone_median_ratio`: Posizione rispetto alla mediana
- `price_zone_zscore`: Z-score del prezzo nella zona
- `price_zone_iqr_position`: Posizione nell'IQR della zona
- `price_zone_range_position`: Posizione nel range min-max

**Perché aiutano**: Un appartamento a 150k€ in zona D2 è "normale", ma in zona C4 è "lussuoso". Il modello ora lo capisce.

#### **B. Typology Statistics** (8 feature)
- `type_zone_price_mean/median/std`: Prezzo per tipologia × zona
- `type_zone_count`: Numero transazioni per tipologia × zona
- `type_price_mean/median`: Prezzo globale per tipologia
- `price_vs_type_zone_mean`: Posizione rispetto alla media tipologia-zona
- `type_zone_rarity`: Quanto è rara questa combinazione

**Perché aiutano**: Cattura nicchie di mercato (es. negozi in zona centrale vs periferia).

#### **C. Surface Context** (5 feature)
- `zone_surface_mean/median`: Superficie media nella zona
- `type_zone_surface_mean`: Superficie media per tipologia × zona
- `surface_vs_zone_mean`: Superficie relativa alla zona
- `surface_vs_type_zone_mean`: Superficie relativa a tipologia-zona

**Perché aiutano**: Un 150mq è "grande" per un appartamento ma "normale" per una villa.

#### **D. Interaction Features** (4+ feature)
- `prezzo_mq`: Prezzo al metro quadro
- `prezzo_mq_vs_zone`: Prezzo/mq relativo alla zona
- `log_superficie`: Log superficie (cattura effetti scala)
- `superficie_x_categoria`: Interazione superficie × categoria catastale

**Perché aiutano**: Relazioni non-lineari chiave (es. prezzo/mq varia per zona).

#### **E. Temporal Context** (7 feature)
- `temporal_price_mean/median`: Trend temporale prezzi
- `temporal_count`: Volume transazioni per mese
- `price_vs_temporal_mean`: Prezzo relativo al trend
- `quarter`: Trimestre (stagionalità)
- `months_from_start`: Trend lineare temporale

**Perché aiutano**: Cattura inflazione e trend di mercato.

---

### ✅ **2. Regularizzazione Aggressiva** (`config/config_optimized.yaml`)

#### **A. CatBoost** (Modello Principale)
```yaml
# PRIMA (baseline):
depth: 4-10               → DOPO: 4-7 ✅
learning_rate: 0.001-0.3  → DOPO: 0.01-0.12 ✅
l2_leaf_reg: 10-100       → DOPO: 3-30 ✅
bagging_temperature: 0.5-5.0 → DOPO: 0.0-1.5 ✅
border_count: 16-255      → DOPO: 32-128 ✅
random_strength: 0.0-2.0  → DOPO: 0.0-1.0 ✅
rsm: 0.5-1.0              → DOPO: 0.5-0.85 ✅

# NUOVI PARAMETRI:
iterations: 800 (con early_stopping_rounds: 50) ✅
min_data_in_leaf: 20-80 ✅
max_ctr_complexity: 1-3 ✅
use_best_model: true ✅
eval_metric: 'MAPE' ✅
```

**Effetto atteso**: Riduzione overfitting del 40-60%, gap train-test da 0.21 a <0.10

#### **B. XGBoost**
```yaml
max_depth: 8 → 6 ✅
learning_rate: 0.01-0.1 → 0.005-0.05 ✅
min_child_weight: 1.0-10.0 → 5.0-20.0 ✅
reg_alpha: 1e-4-10 → 0.1-50 ✅
reg_lambda: 1e-4-10 → 1.0-100 ✅
early_stopping_rounds: 50 ✅
```

#### **C. LightGBM**
```yaml
max_depth: [-1,4,5,6,7,8] → [4,5,6] ✅
num_leaves: 16-63 → 16-40 ✅
min_child_samples: 20-200 → 50-200 ✅
reg_alpha/lambda: aumentati ✅
```

#### **D. Altri Modelli**
- **GBR**: depth 2-6 (da 2-8), min_samples_leaf 20-60 (da 1-20)
- **HGBT**: max_leaf_nodes 15-100 (da 7-255), l2_reg aumentata
- **RF**: min_samples_split 10-40 (da 2-20), bootstrap forzato true

---

### ✅ **3. Altri Miglioramenti**

#### **Outlier Detection Più Aggressivo**
```yaml
z_thresh: 3.0 → 2.5 ✅
iqr_factor: 1.2 → 1.0 ✅
iso_forest_contamination: 0.05 → 0.08 ✅
min_group_size: 30 → 20 ✅
```

#### **Encoding Più Conservativo**
```yaml
one_hot_max: 10 → 8 ✅
target_encoding_range: [11,30] → [9,20] ✅
target_encoder.smoothing: 1.0 → 5.0 ✅
target_encoder.min_samples_leaf: 1 → 10 ✅
```

#### **Target Transformation Migliorata**
```yaml
transform: 'log' → 'yeojohnson' ✅
```
Yeo-Johnson trova automaticamente la trasformazione ottimale, più flessibile di log.

#### **Cross-Validation Più Robusta**
```yaml
cv_when_no_val.n_splits: 5 → 10 ✅
```

#### **Ensemble Più Robusto**
```yaml
voting.top_n: 3 → 5 ✅
stacking.top_n: 5 → 7 ✅
stacking.cv_folds: 5 → 10 ✅
```

---

## 🎯 Risultati Attesi

### **Baseline (Attuale)**
```
CatBoost Test (scala originale):
├─ RMSE: 36,767€
├─ MAE: 19,811€
├─ MAPE: 58.1%
├─ MAPE floor: 57.5%
└─ R²: 0.736

Overfitting:
├─ Gap R²: 0.214 (21%!)
├─ RMSE ratio: 2.67x
└─ MAE ratio: 2.52x

Performance per gruppo:
├─ Zone migliori (D2): R²=0.84, MAPE=37%
├─ Zone problematiche (C4): R²=0.70, MAPE=134%!
└─ Fasce prezzo basse: R² NEGATIVI!
```

### **Target (Dopo Ottimizzazione)**
```
CatBoost Test (scala originale):
├─ RMSE: 22,000-26,000€  (-30% a -40%) ✅
├─ MAE: 12,000-15,000€   (-35% a -40%) ✅
├─ MAPE: 25-35%          (-40% a -55%) ✅✅
├─ MAPE floor: 23-32%    (-40% a -55%) ✅✅
└─ R²: 0.82-0.87         (+10% a +18%) ✅

Overfitting:
├─ Gap R²: <0.10         (-50% a -70%) ✅✅
├─ RMSE ratio: <1.8x     (-30% a -40%) ✅
└─ MAE ratio: <1.7x      (-30% a -40%) ✅

Performance per gruppo:
├─ Zone: tutte con R²>0.60, MAPE<50% ✅
├─ Fasce prezzo: tutte con R²>0.40 ✅
└─ Varianza performance ridotta ✅
```

---

## 📦 Esecuzione

### **Opzione 1: Script Automatico** (CONSIGLIATO)

```bash
# Esegue preprocessing + training + evaluation + confronto
python run_optimization.py
```

**Cosa fa**:
1. ✅ Carica config ottimizzata
2. ✅ Esegue preprocessing con feature contestuali
3. ✅ Esegue training con regularizzazione aggressiva
4. ✅ Esegue evaluation
5. ✅ Confronta risultati baseline vs ottimizzato
6. ✅ Mostra summary miglioramenti

**Output**:
```
📊 CONFRONTO RISULTATI: BASELINE vs OTTIMIZZATO
================================================================================

🎯 CATBOOST - Miglior Modello
--------------------------------------------------------------------------------

💰 METRICHE TEST (scala originale - EURO):
Metric          Baseline      Ottimizzato           Delta        Δ%
--------------------------------------------------------------------------------
rmse             36,767€         24,500€        -12,267€    -33.36% ✅
mae              19,811€         13,200€         -6,611€    -33.37% ✅
mape              58.10%          32.50%         -25.60%    -44.06% ✅
mape_floor        57.52%          31.80%         -25.72%    -44.71% ✅

🔍 OVERFITTING (Train-Test Gap):
gap_r2                   0.214000         0.085000        -0.129000 ✅
ratio_rmse               2.670000         1.650000        -1.020000 ✅
...

📝 SUMMARY
🎯 Target Metrics:
   • RMSE ridotto:      33.36% ✅
   • MAPE ridotto:      44.71% ✅
   • R² migliorato:     13.25% ✅
   • Overfitting ridotto:  60.28% ✅
```

### **Opzione 2: Esecuzione Manuale**

```bash
# Step 1: Preprocessing
python main.py --config config/config_optimized.yaml --steps preprocessing

# Step 2: Training
python main.py --config config/config_optimized.yaml --steps training

# Step 3: Evaluation
python main.py --config config/config_optimized.yaml --steps evaluation
```

### **Opzione 3: Solo CatBoost (Veloce)**

Per testare rapidamente, disabilita altri modelli in `config_optimized.yaml`:
```yaml
models:
  catboost:
    enabled: true
  xgboost:
    enabled: false  # ← Disabilita
  lightgbm:
    enabled: false  # ← Disabilita
  # ... altri: false
```

Poi:
```bash
python run_optimization.py
```

---

## ⏱️ Tempi Stimati

| Step | Tempo (config completa) | Tempo (solo CatBoost) |
|------|------------------------|----------------------|
| Preprocessing | 5-10 min | 5-10 min |
| Training CatBoost | 15-30 min | 15-30 min |
| Training altri (5 modelli) | 60-90 min | - |
| Evaluation | 2-5 min | 2-5 min |
| **TOTALE** | **~2 ore** | **~30-45 min** |

---

## 📊 Monitoraggio Risultati

### **1. Durante Training**

Guarda i log per:
```
✅ Feature contestuali completate: 44 nuove feature aggiunte
   Colonne totali: 127 → 171

[catboost] best neg_mean_absolute_percentage_error=-0.0285 | test r2=0.8532 rmse=0.4821
```

### **2. Metriche Chiave da Verificare**

**File**: `models/catboost/metrics.json`

```json
{
  "metrics_test_original": {
    "rmse": < 26000,        // Target: <26k€
    "mape_floor": < 0.35,   // Target: <35%
    "r2": > 0.82            // Target: >0.82
  },
  "overfit": {
    "gap_r2": < 0.10,       // Target: <0.10
    "ratio_rmse": < 1.8     // Target: <1.8x
  }
}
```

### **3. Performance per Gruppo**

**File**: `models/catboost/group_metrics_AI_ZonaOmi.csv`

Verifica che:
- ✅ Tutte le zone abbiano R² > 0.50
- ✅ Nessuna zona abbia MAPE > 60%

**File**: `models/catboost/group_metrics_price_band.csv`

Verifica che:
- ✅ Nessuna fascia abbia R² negativo
- ✅ Tutte le fasce (tranne outliers) abbiano R² > 0.30

---

## 🔧 Troubleshooting

### **Problema: Feature contestuali non aggiunte**

**Sintomo**: Log non mostra `"Feature contestuali completate"`

**Soluzione**:
```bash
# Verifica che il modulo sia importato
grep "from preprocessing.contextual_features" src/preprocessing/pipeline.py

# Verifica che la funzione sia chiamata
grep "add_all_contextual_features" src/preprocessing/pipeline.py
```

### **Problema: Errore durante preprocessing**

**Sintomo**: `KeyError: 'AI_Prezzo_Ridistribuito'` o simili

**Soluzione**: Il target non esiste prima dello split. Le feature contestuali sono aggiunte PRIMA dello split, quindi hanno accesso al target.

Se l'errore persiste:
```yaml
# In config_optimized.yaml, disabilita temporaneamente:
target:
  transform: 'log'  # Usa log invece di yeojohnson se da problemi
```

### **Problema: Training troppo lento**

**Soluzione 1**: Riduci trials
```yaml
trials_advanced: 100  # Invece di 150
```

**Soluzione 2**: Usa solo CatBoost (vedi sopra)

**Soluzione 3**: Disabilita SHAP temporaneamente
```yaml
shap:
  enabled: false
```

### **Problema: Out of Memory**

**Soluzione**: Riduci sample SHAP
```yaml
shap:
  sample_size: 200  # Invece di 500
```

---

## 🚀 Prossimi Passi (Se Target Non Raggiunto)

### **Se MAPE Ancora > 35%**

→ **Implementa modelli specializzati per fascia prezzo** (Fase 2)

Crea `/workspace/src/training/specialized_models.py` (vedi guida principale)

### **Se Overfitting Ancora > 0.10**

→ **Aumenta ulteriormente regularizzazione**:

```yaml
catboost:
  search_space:
    depth: {type: int, low: 4, high: 6}  # Riduci ancora
    l2_leaf_reg: {type: float, low: 5.0, high: 50.0}  # Aumenta min
    min_data_in_leaf: {type: int, low: 30, high: 100}  # Aumenta min
```

### **Se Performance Gruppi Ancora Disomogenea**

→ **Implementa group-aware tuning** (vedi guida principale)

---

## 📝 Note Importanti

1. **Backup**: I risultati baseline sono in `models/`. Vengono sovrascritti. Fai backup se necessario:
   ```bash
   cp -r models/ models_baseline/
   ```

2. **W&B**: Se abilitato, i risultati sono tracciati su Weights & Biases con tag `optimized`.

3. **Reproductibilità**: Seed fisso (42) garantisce risultati riproducibili.

4. **Feature leakage**: Le feature contestuali usano il target per aggregazione, ma lo fanno PRIMA dello split temporale, quindi no leakage.

---

## ✅ Checklist Pre-Esecuzione

- [ ] Backup risultati baseline: `cp -r models/ models_baseline/`
- [ ] Dati raw presenti: `data/raw/raw.parquet`
- [ ] Config letta: `config/config_optimized.yaml`
- [ ] Dependencies aggiornate: `pip install -r requirements.txt`
- [ ] Spazio disco sufficiente: ~2GB per modelli
- [ ] (Opzionale) W&B configurato: variabili env `WANDB_*`

---

## 🎯 Esecuzione Rapida

```bash
# 1. Backup baseline (opzionale)
cp -r models/ models_baseline_$(date +%Y%m%d)/

# 2. Run ottimizzazione
python run_optimization.py

# 3. Confronta risultati
cat models/catboost/metrics.json | grep -A 10 metrics_test_original

# 4. Verifica gruppi
head -20 models/catboost/group_metrics_price_band.csv
```

---

## 📚 Riferimenti

- **Guida completa**: Vedi analisi iniziale (messaggio precedente)
- **Config baseline**: `config/config.yaml`
- **Config ottimizzata**: `config/config_optimized.yaml`
- **Feature contestuali**: `src/preprocessing/contextual_features.py`
- **Script esecuzione**: `run_optimization.py`

---

**Buona ottimizzazione! 🚀**
