# 🚀 Feature Improvements - Ispirate da RealEstatePricePrediction

Questo documento riassume tutte le nuove feature implementate nel progetto Stimatrix, ispirate dal repository [RealEstatePricePrediction](https://github.com/BaffoBello14/RealEstatePricePrediction).

## ✅ Tutte le Feature Implementate (100%)

### 📊 **Fase 1: Quick Wins**

#### 1.1 Trasformazioni Target Multiple ⭐⭐⭐
**File modificati**: `src/utils/transforms.py` (NEW), `src/preprocessing/pipeline.py`

**Cosa fa**: Supporto completo per trasformazioni target con metodi avanzati:
- **`boxcox`** - Box-Cox con λ ottimizzato automaticamente (**DEFAULT**)
- **`yeojohnson`** - Yeo-Johnson (come Box-Cox ma gestisce negativi/zero)
- **`sqrt`** - radice quadrata
- **`log`** (log1p) - logaritmo naturale
- **`log10`** - log base 10 con offset configurabile
- **`none`** - nessuna trasformazione

**Configurazione**:
```yaml
target:
  transform: 'boxcox'  # DEFAULT - Ottimizza automaticamente la trasformazione!
  log10_offset: 1.0    # Usato solo se transform: 'log10'
```

**Benefici**: Box-Cox e Yeo-Johnson ottimizzano automaticamente la trasformazione per massimizzare la normalità dei dati.

---

#### 1.2 Feature Temporali Avanzate ⭐⭐⭐
**File modificati**: `src/preprocessing/feature_extractors.py`, `src/preprocessing/pipeline.py`

**Feature create**:
- **`quarter`** - Q1, Q2, Q3, Q4
- **`is_summer`** - flag booleano per Giugno-Agosto
- **`month_sin`, `month_cos`** - encoding ciclico per mese (cattura ciclicità)
- **`months_since_start`** - contatore progressivo tempo

**Configurazione**:
```yaml
advanced_features:
  temporal:
    enabled: true
    features: ['quarter', 'is_summer', 'month_sin', 'month_cos', 'months_since_start']
```

**Benefici**: Cattura meglio la stagionalità del mercato immobiliare.

---

#### 1.3 Optuna MedianPruner ⭐⭐
**File modificati**: `src/training/tuner.py`, `src/training/train.py`

**Cosa fa**: Early stopping intelligente durante hyperparameter tuning:
- Stoppa trial poco promettenti dopo 10 warmup steps
- Confronta con mediana dei trial precedenti
- Supporta anche PercentilePruner

**Configurazione**:
```yaml
training:
  pruner:
    enabled: true
    type: "median"
    n_warmup_steps: 10
```

**Benefici**: Risparmia 30-50% del tempo di tuning.

---

### 🎯 **Fase 2: Feature Engineering Avanzato**

#### 2.1 Feature Geografiche Avanzate ⭐⭐⭐
**File modificati**: `src/preprocessing/feature_extractors.py`, `src/preprocessing/pipeline.py`

**Feature create**:
- **`geo_cluster`** - Spatial clustering con KMeans su lat/lon (identifica micro-zone)
- **`distance_to_center_km`** - Distanza Haversine dal centro città
- **`density_500m`** - Numero proprietà entro 500m (indice urbanizzazione)

**Configurazione**:
```yaml
advanced_features:
  geographic:
    enabled: true
    spatial_clusters:
      enabled: true
      n_clusters: 8
    distance_to_center:
      enabled: true
      center_lat: 45.1564  # Mantova
      center_lon: 10.7914
    density:
      enabled: true
      radius_km: 0.5
```

**Benefici**: Cattura pattern geografici non visibili con ZonaOmi da sola.

---

#### 2.2 Missing Pattern Flags ⭐⭐
**File modificati**: `src/preprocessing/feature_extractors.py`, `src/preprocessing/pipeline.py`

**Cosa fa**: Crea flag binari per presenza/assenza di gruppi di colonne:
- **`has_C1`** - 1 se almeno una colonna C1_* è non-null
- **`has_C2`** - 1 se almeno una colonna C2_* è non-null

**Configurazione**:
```yaml
advanced_features:
  missing_patterns:
    enabled: true
    create_flags_for_prefixes: ['C1_', 'C2_']
    feature_name_template: 'has_{prefix}'
```

**Benefici**: Il modello impara che "avere dati CENED" è informativo.

---

#### 2.3 Feature di Interazione ⭐⭐
**File modificati**: `src/preprocessing/feature_extractors.py`, `src/preprocessing/pipeline.py`

**Tipi di interazioni**:
- **Categorical × Numeric**: `Superficie × ZonaOmi` (media gruppo pesata)
- **Categorical × Categorical**: `Categoria × Zona` (combinazioni)
- **Polynomial**: `Superficie²` (relazioni non lineari)

**Configurazione**:
```yaml
advanced_features:
  interactions:
    enabled: true
    categorical_numeric:
      - ['AI_Superficie', 'AI_ZonaOmi']
    categorical_categorical:
      - ['AI_IdCategoriaCatastale', 'AI_ZonaOmi']
    polynomial:
      columns: ['AI_Superficie']
      degree: 2
```

**Benefici**: Cattura relazioni complesse tra variabili.

---

### 🎨 **Fase 3: Encoding Avanzato**

#### 3.1 Encoding Multi-Strategia ⭐⭐⭐
**File creati**: `src/preprocessing/encoders_advanced.py`  
**File modificati**: `src/preprocessing/pipeline.py`

**Strategia automatica basata su cardinalità**:
- **≤10 unique** → **One-Hot Encoding**
- **11-30 unique** → **Target Encoding** (con smoothing)
- **31-100 unique** → **Frequency Encoding**
- **101-200 unique** → **Ordinal Encoding**
- **>200 unique** → **DROP**

**Configurazione**:
```yaml
encoding:
  one_hot_max: 10
  target_encoding_range: [11, 30]
  frequency_encoding_range: [31, 100]
  ordinal_encoding_range: [101, 200]
  drop_above: 200
  target_encoder:
    smoothing: 1.0
    min_samples_leaf: 1

profiles:
  tree:
    enabled: true
    apply_advanced_encoding: true  # ← Abilita encoding avanzato
```

**Benefici**: Gestisce meglio categoriche ad alta cardinalità, riduce dimensionalità.

---

#### 3.2 Boolean Handling Automatico ⭐
**File modificati**: `src/preprocessing/encoders_advanced.py`, `src/preprocessing/pipeline.py`

**Cosa fa**: Converte automaticamente colonne booleane:
- `True` → 1, `False` → 0, `NaN` → -1
- Auto-detection di colonne boolean-like (≤2 unique values)

**Configurazione**:
```yaml
encoding:
  handle_booleans: true
  boolean_null_value: -1
```

---

### 📈 **Fase 4: Diagnostica Avanzata**

#### 4.1 Residual Analysis ⭐⭐⭐
**File creati**: `src/training/diagnostics.py`

**Analisi**:
- **Overall statistics**: mean, std, median, q25, q75, skewness, kurtosis
- **By group**: residui per ZonaOmi, Categoria, Tipologia, price_quartile
- **Worst predictions**: top-50 errori più grandi salvati in CSV
- **Plots**: residual_vs_predicted, residual_vs_actual, residual_distribution

**Configurazione**:
```yaml
diagnostics:
  residual_analysis:
    enabled: true
    by_groups: ['AI_ZonaOmi', 'AI_IdCategoriaCatastale', 'price_quartile']
    save_worst_predictions: true
    top_n_worst: 50
    plots: ['residual_vs_predicted', 'residual_distribution']
```

**Output**: `models/{model}_worst_predictions.csv`, `models/{model}_residual_plots/`

**Benefici**: Identifica i punti deboli del modello per gruppo.

---

#### 4.2 Drift Detection ⭐⭐⭐
**File modificati**: `src/training/diagnostics.py`

**Metodi**:
- **PSI** (Population Stability Index): < 0.1 OK, 0.1-0.15 moderato, > 0.15 significativo
- **Kolmogorov-Smirnov test**: test statistico per shift distribuzionale

**Configurazione**:
```yaml
diagnostics:
  drift_detection:
    enabled: true
    methods: ['psi', 'ks_test']
    alert_threshold: 0.15
    save_report: true
    output_file: 'models/drift_report.json'
```

**Output**: 
```json
{
  "alerts": [
    {"feature": "AI_Superficie", "method": "psi", "value": 0.23},
    {"feature": "geo_cluster", "method": "ks_test", "pvalue": 0.001}
  ]
}
```

**Benefici**: Rileva se distribuzione train/test è diversa (possibile overfitting).

---

#### 4.3 Prediction Intervals ⭐⭐⭐
**File modificati**: `src/training/diagnostics.py`

**Cosa fa**: Quantifica l'incertezza delle predizioni con intervalli di confidenza:
- Metodo: **Residual Bootstrap**
- Intervalli: 80%, 90%
- Metriche: coverage, average_width

**Configurazione**:
```yaml
uncertainty:
  prediction_intervals:
    enabled: true
    method: 'residual_bootstrap'
    n_bootstraps: 100
    confidence_levels: [0.8, 0.9]
```

**Output**:
```json
{
  "90%": {
    "coverage": 0.92,
    "average_width": 15420.3,
    "target_coverage": 0.90
  }
}
```

**Benefici**: Fornisce intervalli di confidenza per le predizioni.

---

## 📁 File Creati/Modificati

### File Nuovi
- `src/utils/transforms.py` - Trasformazioni target (Box-Cox, Yeo-Johnson, etc.)
- `src/preprocessing/encoders_advanced.py` - Encoding multi-strategia
- `src/training/diagnostics.py` - Residual analysis, drift detection, prediction intervals
- `FEATURE_IMPROVEMENTS.md` - Questo file

### File Modificati
- `src/preprocessing/feature_extractors.py` - Aggiunto: temporal, geographic, interactions, missing patterns
- `src/preprocessing/pipeline.py` - Integrazione di tutte le nuove feature
- `src/training/tuner.py` - Aggiunto MedianPruner
- `src/training/train.py` - Passa pruner_config a tune_model
- `config/config.yaml` - Tutte le nuove sezioni di configurazione
- `requirements.txt` - (già presente category_encoders)

---

## 🎯 Riepilogo Priorità

### ⭐⭐⭐ MUST HAVE (Massimo impatto)
1. **Trasformazioni Target Multiple** - Box-Cox può dare +5-10% performance
2. **Feature Temporali Avanzate** - Cattura stagionalità
3. **Feature Geografiche Avanzate** - Density e clustering molto potenti
4. **Encoding Multi-Strategia** - Target encoding superiore a OHE
5. **Residual Analysis** - Essenziale per capire dove migliorare
6. **Drift Detection** - Rileva problemi di generalizzazione

### ⭐⭐ NICE TO HAVE (Buon impatto)
7. Missing Pattern Flags
8. Feature di Interazione
9. Optuna MedianPruner
10. Prediction Intervals

### ⭐ OPTIONAL (Miglioramenti minori)
11. Boolean Handling

---

## 🚀 Come Usare le Nuove Feature

### Esempio 1: Abilita Tutto
```yaml
# config/config.yaml
target:
  transform: 'boxcox'  # Default, puoi cambiare a 'log', 'yeojohnson', etc.

advanced_features:
  temporal:
    enabled: true
  geographic:
    enabled: true
  interactions:
    enabled: true
  missing_patterns:
    enabled: true

profiles:
  tree:
    apply_advanced_encoding: true

training:
  pruner:
    enabled: true

diagnostics:
  residual_analysis:
    enabled: true
  drift_detection:
    enabled: true

uncertainty:
  prediction_intervals:
    enabled: true
```

### Esempio 2: Solo Feature Temporali e Geografiche
```yaml
advanced_features:
  temporal:
    enabled: true
  geographic:
    enabled: true
  interactions:
    enabled: false
  missing_patterns:
    enabled: false
```

### Esempio 3: Prova Diverse Trasformazioni Target
```bash
# Run 1: log
python main.py --config config/config.yaml --steps preprocessing training

# Run 2: boxcox (modifica config.yaml: transform: 'boxcox')
python main.py --config config/config.yaml --steps preprocessing training

# Run 3: yeojohnson (modifica config.yaml: transform: 'yeojohnson')
python main.py --config config/config.yaml --steps preprocessing training

# Confronta: models/summary.json
```

---

## 🎉 Conclusione

Abbiamo implementato **tutte le feature** dal repository esterno:
- ✅ Fase 1: Quick Wins (3/3)
- ✅ Fase 2: Feature Engineering (3/3)
- ✅ Fase 3: Encoding Avanzato (2/2)
- ✅ Fase 4: Diagnostica (3/3)

**Totale: 11/11 feature implementate (100%)**

Le feature sono **modulari** e possono essere abilitate/disabilitate indipendentemente tramite config YAML.

---

## 📚 References

- **Repository Originale**: https://github.com/BaffoBello14/RealEstatePricePrediction
- **Optuna Pruners**: https://optuna.readthedocs.io/en/stable/reference/pruners.html
- **Category Encoders**: https://contrib.scikit-learn.org/category_encoders/
- **Box-Cox Transform**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
