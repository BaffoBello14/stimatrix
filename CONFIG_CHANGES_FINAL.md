# ✅ MODIFICHE FINALI CONFIG - Riepilogo

**Data**: 2025-11-13  
**Modifiche richieste dall'utente applicate**

---

## 🔧 MODIFICHE APPLICATE

### **1. Unificazione Filtri**

**Prima:**
```yaml
data_filters:
  # Tanti filtri sparsi...
  anno_min: null
  mese_min: null
  prezzo_min: null
  superficie_min: null
  locali_min: null
  ...

temporal_filter:  # Sezione separata!
  enabled: true
  min_year: 2022
  exclude_zones: [...]
  exclude_tipologie: [...]
```

**Dopo:**
```yaml
data_filters:
  # Solo i 4 filtri essenziali richiesti
  anno_min: 2022
  anno_max: null
  prezzo_min: null
  prezzo_max: null
  zone_escluse: ['E1', 'E2', 'E3', 'R1']
  tipologie_escluse: ['4']
```

**Vantaggi:**
- ✅ Tutto in un unico blocco
- ✅ Solo filtri essenziali (anno, prezzo, esclusioni)
- ✅ Più chiaro e semplice da usare

---

### **2. Rimozione Commenti Storici**

**Prima:**
```yaml
# Outlier detection - PIÙ AGGRESSIVO
outliers:
  z_thresh: 2.5           # ✅ RIDOTTO da 3.0 - più aggressivo
  iso_forest_contamination: 0.08  # ✅ AUMENTATO da 0.05

# Target configuration - MIGLIORATO
target:
  transform: 'yeojohnson'  # ✅ CAMBIATO da 'log' - trova automaticamente lambda
```

**Dopo:**
```yaml
# Outlier detection
outliers:
  z_thresh: 2.5
  iso_forest_contamination: 0.08

# Target configuration
target:
  transform: 'yeojohnson'  # none | log | log10 | sqrt | boxcox | yeojohnson
```

**Vantaggi:**
- ✅ Config sembra immacolato (non modificato)
- ✅ Commenti con scelte possibili dove utile
- ✅ Più professionale

---

### **3. Commenti con Scelte Possibili**

**Aggiunti in punti chiave:**

```yaml
logging:
  level: INFO  # DEBUG | INFO | WARNING | ERROR | CRITICAL

execution:
  steps: ["preprocessing", "training", "evaluation"]  # all | schema | dataset | preprocessing | training | evaluation

database:
  output_format: 'parquet'  # parquet | csv
  compression: 'snappy'  # snappy | gzip | none

target:
  transform: 'yeojohnson'  # none | log | log10 | sqrt | boxcox | yeojohnson

outliers:
  method: 'ensemble'  # ensemble | iqr | zscore | iso_forest
  fallback_strategy: 'global'  # global | skip

imputation:
  numeric_strategy: 'median'  # median | mean
  categorical_strategy: 'most_frequent'  # most_frequent | constant

temporal_split:
  mode: 'fraction'  # fraction | date

scaling:
  scaler_type: 'standard'  # standard | minmax | robust

training:
  primary_metric: "neg_mean_absolute_percentage_error"  # neg_mape | r2 | neg_rmse | neg_mae
  sampler: "auto"  # auto | tpe | random | cmaes
  
  cv_when_no_val:
    kind: kfold  # kfold | stratified | timeseries

ensembles:
  stacking:
    final_estimator: "ridge"  # ridge | lasso | elasticnet | linear

uncertainty:
  prediction_intervals:
    method: 'residual_bootstrap'  # residual_bootstrap | quantile_regression

evaluation:
  group_metrics:
    price_band:
      method: 'quantile'  # quantile | equal_width | custom

tracking:
  wandb:
    mode: 'online'  # online | offline | disabled
```

**Vantaggi:**
- ✅ Utente vede subito tutte le opzioni disponibili
- ✅ Non deve cercare nella documentazione
- ✅ Riduce errori di configurazione

---

### **4. Nome Run Dinamico**

**Prima:**
```yaml
# config.yaml
tracking:
  wandb:
    name: 'opt_contextual_reg'  # Sempre lo stesso!

# config_fast.yaml
tracking:
  wandb:
    name: 'fast_test'  # Sempre lo stesso!
```

**Dopo:**
```yaml
# config.yaml e config_fast.yaml
tracking:
  wandb:
    # name non più specificato - generato automaticamente
```

**Generazione automatica in `src/utils/wandb_utils.py`:**
```python
# Generate descriptive run name
config_type = "fast" if "fast" in project.lower() else "full"
target_cfg = self.config.get("target", {})
transform = target_cfg.get("transform", "none")
timestamp = time.strftime('%m%d_%H%M')
resolved_name = f"{config_type}_{transform}_{timestamp}"
```

**Esempi nomi run generati:**
- `full_yeojohnson_1113_1445` (config.yaml, trasform yeojohnson, 13 nov ore 14:45)
- `fast_log_1113_0920` (config_fast.yaml, transform log, 13 nov ore 09:20)
- `full_none_1114_1630` (config.yaml, no transform, 14 nov ore 16:30)

**Vantaggi:**
- ✅ Nome sempre diverso per ogni run
- ✅ Identifica subito config type (full vs fast)
- ✅ Vede transform usato
- ✅ Timestamp per ordinamento cronologico

---

### **5. Fix main.py**

**Prima:**
```python
if args.config.strip().lower() == "fast":
    args.config = "config/config_fast_test.yaml"  # File vecchio!
```

**Dopo:**
```python
if args.config.strip().lower() == "fast":
    args.config = "config/config_fast.yaml"  # File corretto
```

**Vantaggi:**
- ✅ `python main.py --config fast` ora funziona correttamente

---

## 📊 CONFRONTO DATA_FILTERS

### **Prima (Complesso):**
```yaml
data_filters:
  experiment_name: "baseline_full"
  description: "Baseline completo - tutti immobili post-2022"
  anno_min: null
  anno_max: null
  mese_min: null
  mese_max: null
  prezzo_min: null
  prezzo_max: null
  prezzo_mq_min: null
  prezzo_mq_max: null
  superficie_min: null
  superficie_max: null
  locali_min: null
  locali_max: null
  piano_min: null
  piano_max: null
  zone_incluse: null
  zone_escluse: null
  tipologie_incluse: null
  tipologie_escluse: null
  max_missing_ratio: null
  remove_outliers_iqr: false
  iqr_factor: 1.5

temporal_filter:  # Sezione separata!
  enabled: true
  min_year: 2022
  min_month: null
  exclude_zones: ['E1', 'E2', 'E3', 'R1']
  exclude_tipologie: ['4']
```

### **Dopo (Semplice):**
```yaml
data_filters:
  # Anno stipula
  anno_min: 2022
  anno_max: null
  
  # Prezzo (€)
  prezzo_min: null
  prezzo_max: null
  
  # Zone OMI da escludere
  zone_escluse: ['E1', 'E2', 'E3', 'R1']
  
  # Tipologie edilizie da escludere
  tipologie_escluse: ['4']  # Ville
```

**Riduzione**: 32 linee → 14 linee (56% più corto!)

---

## 🎯 FILE MODIFICATI

1. **`config/config.yaml`** - Riscritto completamente
2. **`config/config_fast.yaml`** - Riscritto completamente
3. **`src/utils/wandb_utils.py`** - Nome run dinamico
4. **`main.py`** - Fix riferimento config_fast.yaml

---

## ✅ VERIFICA

```bash
# Verifica config puliti
grep -n "✅\|RIDOTTO\|AUMENTATO\|CAMBIATO" config/config.yaml
# → Nessun risultato (pulito!)

# Verifica nome run dinamico
grep -n "opt_contextual_reg\|fast_test" config/*.yaml
# → Nessun risultato (nomi hardcoded rimossi!)

# Verifica filtri unificati
grep -n "temporal_filter:" config/config.yaml
# → Nessun risultato (unificato in data_filters!)

# Test shorthand fast
python main.py --config fast --help
# → Dovrebbe caricare config_fast.yaml
```

---

## 🚀 ESEMPI USO

### **Esempio 1: Training base**
```bash
python main.py
# → Config: config.yaml
# → Filtri: anno≥2022, zone E1/E2/E3/R1 escluse, tipologia 4 esclusa
# → Run name: full_yeojohnson_1113_1445
```

### **Esempio 2: Training fast**
```bash
python main.py --config fast
# → Config: config_fast.yaml
# → Filtri: stessi del base
# → Run name: fast_yeojohnson_1113_1450
```

### **Esempio 3: Training con filtri prezzo**
Modifica `config/config.yaml`:
```yaml
data_filters:
  anno_min: 2022
  prezzo_min: 50000   # Aggiungi filtro
  prezzo_max: 300000  # Aggiungi filtro
  zone_escluse: ['E1', 'E2', 'E3', 'R1']
  tipologie_escluse: ['4']
```

```bash
python main.py
# → Run name: full_yeojohnson_1113_1500
# → Dataset filtrato: solo 50k-300k€
```

---

## 💡 VANTAGGI FINALI

### **Usabilità:**
- ✅ Filtri tutti in un posto
- ✅ Solo i parametri essenziali (anno, prezzo, esclusioni)
- ✅ Commenti con scelte possibili (meno errori)

### **Manutenibilità:**
- ✅ Config pulito (no storia modifiche)
- ✅ Nomi run sempre univoci e descrittivi
- ✅ Più facile capire cosa fa ogni opzione

### **Professionalità:**
- ✅ Config sembra immacolato (non "patchato")
- ✅ Nomi run significativi
- ✅ Documentazione inline completa

---

**Tutte le modifiche richieste sono state applicate!** ✅
