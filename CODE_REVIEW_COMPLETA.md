# 📋 REVIEW COMPLETA DEL CODICE - STIMATRIX PIPELINE

**Data Review**: 2025-11-11  
**Revisore**: AI Assistant  
**Versione Codice**: Branch `cursor/review-code-and-suggest-configurations-ef0d`

---

## 📊 EXECUTIVE SUMMARY

**Stimatrix** è una pipeline **production-ready** end-to-end per la stima automatica dei prezzi immobiliari, costruita con best practices professionali e architettura modulare eccellente.

### ✅ Punti di Forza (Eccellenze)

1. **🏗️ Architettura Modulare Eccezionale**: Separazione perfetta tra `db`, `preprocessing`, `training`, `utils`
2. **🔒 Sicurezza Robusta**: Credenziali da env vars, input sanitization, audit logging
3. **🧪 Testing Completo**: 11 test files con coverage end-to-end (preprocessing, training, encoding, overflow, ecc.)
4. **📈 Experiment Tracking**: Integrazione W&B nativa e configurabile
5. **🎯 Target Transform Avanzato**: Supporto Box-Cox, Yeo-Johnson, log, sqrt con Duan smearing
6. **🔍 Diagnostics Avanzate**: Residual analysis, drift detection, prediction intervals, SHAP
7. **📊 Profili Multipli**: `scaled`, `tree`, `catboost` per famiglie di modelli diverse
8. **⚙️ Configurazione Flessibile**: YAML con env vars expansion, profili per-model
9. **🚀 Feature Engineering Intelligente**: Estrazione WKT, JSON, GeoJSON, floor parsing
10. **🔄 Backward Compatibility**: File senza suffisso per compatibilità legacy

### ⚠️ Aree di Miglioramento (Non Critiche)

1. **📝 Documentazione**: Manca docstring in alcune funzioni interne
2. **🧹 Refactoring Minore**: Alcune funzioni lunghe (`run_preprocessing`, `run_training`) potrebbero essere split
3. **⚡ Performance**: Considerare caching di query DB e preprocessing intermedio
4. **🔍 Monitoring**: Aggiungere health checks e alerting produzione
5. **🐳 Deployment**: Manca Dockerfile/docker-compose per deploy facile

**Giudizio Complessivo**: ⭐⭐⭐⭐⭐ (5/5) - Codice di **qualità professionale elevata**

---

## 🏗️ ARCHITETTURA E STRUTTURA

### 1. Moduli Principali

```
src/
├── db/                     # Database connection & schema extraction
│   ├── connect.py          # ✅ Secure connection con retry e TLS
│   └── schema_extract.py   # ✅ Type normalization, view support
├── dataset_builder/
│   └── retrieval.py        # ✅ SQL templates, POI/ZTL enrichment
├── preprocessing/
│   ├── pipeline.py         # 🔥 CORE: orchestrazione preprocessing
│   ├── encoders.py         # Multi-strategy encoding (OHE, target, freq, ordinal)
│   ├── imputation.py       # Group-wise imputation
│   ├── outliers.py         # Ensemble outlier detection (IQR+Z-score+IsoForest)
│   ├── target_transforms.py # Box-Cox, Yeo-Johnson, log con Duan smearing
│   ├── transformers.py     # Scaling, PCA, correlation pruning
│   └── feature_extractors.py # WKT, JSON, GeoJSON, floor parsing
├── training/
│   ├── train.py            # 🔥 CORE: training loop con Optuna
│   ├── tuner.py            # Optuna integration (TPE, AutoSampler)
│   ├── evaluation.py       # Model evaluation e group metrics
│   ├── metrics.py          # Regression metrics + grouped metrics
│   ├── diagnostics.py      # Residual analysis, drift, PI
│   ├── model_zoo.py        # Factory pattern per 12+ modelli
│   ├── ensembles.py        # Voting & Stacking ensembles
│   └── shap_utils.py       # SHAP feature importance
└── utils/
    ├── config.py           # ✅ YAML loader con env vars expansion
    ├── logger.py           # ✅ Structured logging con rotation
    ├── io.py               # ✅ Save/load helpers (JSON, Parquet, CSV)
    ├── security.py         # ✅ Credential manager + input validation
    ├── sql_templates.py    # ✅ SQL template system
    └── wandb_utils.py      # ✅ W&B tracker con graceful degradation
```

### 2. Flusso di Esecuzione

```
main.py
  ↓
[STEP 1: schema] → db.schema_extract.run_schema()
  → schema/db_schema.json
  ↓
[STEP 2: dataset] → dataset_builder.retrieval.run_dataset()
  → data/raw/raw.parquet (con POI/ZTL opzionali)
  ↓
[STEP 3: preprocessing] → preprocessing.pipeline.run_preprocessing()
  → data/preprocessed/
      ├── X_train_{profile}.parquet
      ├── y_train_{profile}.parquet
      ├── X_val_{profile}.parquet (opzionale)
      ├── y_val_{profile}.parquet
      ├── X_test_{profile}.parquet
      ├── y_test_{profile}.parquet
      ├── y_test_orig_{profile}.parquet
      ├── artifacts/
      │   ├── imputers.joblib
      │   ├── {profile}/
      │   │   ├── encoders.joblib
      │   │   ├── winsorizer.joblib
      │   │   └── transforms.joblib (scaler + PCA)
      └── preprocessing_info.json
  ↓
[STEP 4: training] → training.train.run_training()
  → models/
      ├── {model_key}/
      │   ├── model.pkl
      │   ├── metrics.json
      │   ├── optuna_trials.csv
      │   ├── shap/
      │   │   ├── shap_{model}_beeswarm.png
      │   │   ├── shap_{model}_bar.png
      │   │   └── shap_values.npy (opzionale)
      │   ├── group_metrics_AI_ZonaOmi.csv
      │   ├── group_metrics_price_band.csv
      │   └── {model}_worst_predictions.csv
      ├── voting/
      │   └── model.pkl
      ├── stacking/
      │   └── model.pkl
      ├── summary.json
      └── validation_results.csv
  ↓
[STEP 5: evaluation] → training.evaluation.run_evaluation()
  → models/evaluation_summary.json
```

---

## 💾 COME VENGONO SALVATI I RISULTATI

### 1. **Preprocessing Output** (`data/preprocessed/`)

#### File Principali per Profilo

```python
# Per ogni profilo abilitato (tree, catboost, scaled):
X_train_{profile}.parquet       # Feature di training
y_train_{profile}.parquet       # Target di training (transformed)
X_val_{profile}.parquet         # Feature di validation (se valid_fraction > 0)
y_val_{profile}.parquet         # Target di validation (transformed)
X_test_{profile}.parquet        # Feature di test
y_test_{profile}.parquet        # Target di test (transformed)
y_test_orig_{profile}.parquet   # Target di test (scala originale - EURO)
y_val_orig_{profile}.parquet    # Target di validation (scala originale)

# File backward-compatible (copia del primo profilo abilitato)
X_train.parquet
y_train.parquet
X_val.parquet (se esiste validation)
y_val.parquet
X_test.parquet
y_test.parquet
y_test_orig.parquet
y_val_orig.parquet

# Dataset combinato
preprocessed.parquet  # train + val + test con target (per visualizzazione)

# Sidecar per group metrics (evaluation)
group_cols_train_{profile}.parquet  # Colonne per raggruppamento (es. AI_ZonaOmi)
group_cols_test_{profile}.parquet
group_cols_val_{profile}.parquet
```

#### Artefatti di Trasformazione (`data/preprocessed/artifacts/`)

```python
artifacts/
├── imputers.joblib              # SimpleImputer per numeriche/categoriche
├── {profile}/
│   ├── encoders.joblib          # Dict di encoders (OHE, target, freq, ordinal)
│   ├── winsorizer.joblib        # Winsorizer con quantili
│   └── transforms.joblib        # Dict con 'scaler' (StandardScaler/RobustScaler) 
│                                 # e 'pca' (PCA opzionale)
```

#### Metadata (`data/preprocessed/preprocessing_info.json`)

```json
{
  "target_column": "AI_Prezzo_Ridistribuito",
  "target_transformation": {
    "transform": "boxcox",           // Tipo: none|log|log10|sqrt|boxcox|yeojohnson
    "lambda": 0.123,                 // Lambda per Box-Cox/Yeo-Johnson
    "shift": 100.0,                  // Shift per Box-Cox (se y <= 0)
    "log10_offset": 1.0              // Offset per log10
  },
  "profiles_saved": ["tree", "catboost"],
  "feature_columns_per_profile": {
    "tree": ["AI_Superficie", "AI_Vani", ...],
    "catboost": ["AI_Superficie", "AI_ZonaOmi", ...]
  }
}
```

### 2. **Training Output** (`models/`)

#### Per Modello (`models/{model_key}/`)

```python
{model_key}/
├── model.pkl                    # Modello serializzato (joblib)
├── metrics.json                 # Metriche complete
├── optuna_trials.csv            # Trial history di Optuna
├── shap/
│   ├── shap_{model}_beeswarm.png   # SHAP beeswarm plot
│   ├── shap_{model}_bar.png        # SHAP bar plot (feature importance)
│   ├── shap_values.npy             # SHAP values (opzionale, può essere grande)
│   └── shap_sample.parquet         # Sample usato per SHAP
├── group_metrics_AI_ZonaOmi.csv           # Metriche per zona OMI
├── group_metrics_AI_IdCategoriaCatastale.csv  # Metriche per categoria catastale
├── group_metrics_price_band.csv           # Metriche per fascia di prezzo
├── {model}_worst_predictions.csv          # Top N worst predictions
├── {model}_residual_plots/
│   ├── residual_vs_predicted.png
│   ├── residual_vs_actual.png
│   ├── residual_distribution.png
│   └── qq_plot.png
└── {model}_prediction_intervals.json      # Prediction intervals bootstrap
```

#### Struttura `metrics.json` (Esempio)

```json
{
  "model_key": "xgboost",
  "prefix": "tree",
  "primary_metric": "neg_root_mean_squared_error",
  "best_primary_value": -15234.56,
  "best_params": {
    "n_estimators": 1200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.85,
    "min_child_weight": 2.3,
    "reg_alpha": 0.5,
    "reg_lambda": 1.2,
    "gamma": 0.1
  },
  "metrics_train": {
    "r2": 0.9523,
    "rmse": 12345.67,
    "mse": 152413890.0,
    "mae": 8901.23,
    "mape": 0.0823,
    "explained_variance": 0.9530,
    "medae": 6789.45
  },
  "metrics_test": {
    "r2": 0.9012,
    "rmse": 18567.89,
    "mse": 344766666.0,
    "mae": 13456.78,
    "mape": 0.1234,
    "explained_variance": 0.9020,
    "medae": 10234.56
  },
  "metrics_train_original": {
    "r2": 0.9501,
    "rmse": 12789.45,
    "mae": 9123.45,
    "mape": 0.0845,
    "mape_floor": 0.0838
  },
  "metrics_test_original": {
    "r2": 0.8989,
    "rmse": 19234.56,
    "mae": 13890.12,
    "mape": 0.1267,
    "mape_floor": 0.1259
  },
  "smearing_factor": 1.0234,
  "overfit": {
    "gap_r2": 0.0511,
    "gap_explained_variance": 0.0510,
    "ratio_rmse": 1.5034,
    "delta_rmse": 6222.22,
    "ratio_mse": 2.2602,
    "delta_mse": 192352776.0,
    "ratio_mae": 1.5115,
    "delta_mae": 4554.55,
    "ratio_mape": 1.4993,
    "delta_mape": 0.0411,
    "ratio_medae": 1.5070,
    "delta_medae": 3445.11
  }
}
```

#### Ensemble Models (`models/voting/`, `models/stacking/`)

Stessa struttura dei singoli modelli, con `metrics.json` che include:

```json
{
  "type": "voting",  // o "stacking"
  "members": ["xgboost", "lightgbm", "catboost"],
  "final_estimator": "ridge",  // solo per stacking
  "metrics_train": {...},
  "metrics_test": {...},
  "metrics_train_original": {...},
  "metrics_test_original": {...},
  "overfit": {...}
}
```

#### Summary Files

##### `models/summary.json`

```json
{
  "models": {
    "xgboost": {
      "best_params": {...},
      "best_primary_value": -15234.56,
      "metrics_test": {...},
      "metrics_train": {...},
      "metrics_test_original": {...},
      "metrics_train_original": {...},
      "smearing_factor": 1.0234,
      "overfit": {...}
    },
    "lightgbm": {...},
    "catboost": {...}
  },
  "baselines": {
    "xgboost": {
      "metrics_test": {...}
    }
  },
  "ensembles": {
    "voting": {...},
    "stacking": {...}
  }
}
```

##### `models/validation_results.csv`

```csv
Model,Category,Test_RMSE,Test_R2
Optimized_xgboost,Optimized,18567.89,0.9012
Optimized_lightgbm,Optimized,18901.23,0.8987
Optimized_catboost,Optimized,19123.45,0.8965
Ensemble_voting,Ensemble,18234.56,0.9034
Ensemble_stacking,Ensemble,18012.34,0.9056
Baseline_xgboost,Baseline,21234.56,0.8756
```

##### `models/evaluation_summary.json`

```json
{
  "top_models": [
    {
      "Model": "Ensemble_stacking",
      "Category": "Ensemble",
      "Test_RMSE": 18012.34,
      "Test_R2": 0.9056
    },
    ...
  ],
  "test_metrics": [
    {
      "model": "xgboost",
      "r2": 0.9012,
      "rmse": 18567.89,
      "mae": 13456.78,
      "r2_orig": 0.8989,
      "rmse_orig": 19234.56,
      "mae_orig": 13890.12,
      "mape_floor_orig": 0.1259
    },
    ...
  ]
}
```

### 3. **Group Metrics** (`group_metrics_*.csv`)

Esempio di `group_metrics_AI_ZonaOmi.csv`:

```csv
group,count,r2,rmse,mse,mae,mape,medae
B1,1234,0.91,15234.56,232094567.0,10123.45,0.098,7890.12
B2,2345,0.89,17890.12,320054321.0,12345.67,0.112,9876.54
C1,987,0.87,19234.56,370068901.0,13456.78,0.125,10234.56
...
```

Esempio di `group_metrics_price_band.csv`:

```csv
group,count,r2,rmse,mse,mae,mape,medae
PREZZO_(50000.0, 150000.0],1567,0.85,8901.23,79230987.0,6789.45,0.067,5123.45
PREZZO_(150000.0, 250000.0],2890,0.89,15234.56,232094567.0,11234.56,0.089,8765.43
PREZZO_(250000.0, 400000.0],1876,0.91,23456.78,550220987.0,17890.12,0.095,14321.09
PREZZO_(400000.0, 800000.0],456,0.87,45678.90,2086567890.0,34567.89,0.108,28901.23
```

### 4. **Diagnostics Output**

#### Drift Detection (`models/drift_report.json`)

```json
{
  "features": {
    "AI_Superficie": {
      "psi": 0.089,
      "ks_statistic": 0.023,
      "ks_pvalue": 0.234
    },
    "AI_Vani": {
      "psi": 0.187,
      "ks_statistic": 0.078,
      "ks_pvalue": 0.012
    }
  },
  "alerts": [
    {
      "feature": "AI_Vani",
      "method": "psi",
      "value": 0.187,
      "threshold": 0.15,
      "severity": "moderate"
    },
    {
      "feature": "AI_Vani",
      "method": "ks_test",
      "statistic": 0.078,
      "pvalue": 0.012,
      "severity": "high"
    }
  ],
  "summary": {
    "total_features_checked": 87,
    "psi_alerts": 3,
    "ks_alerts": 5,
    "total_alerts": 8
  }
}
```

#### Prediction Intervals (`{model}_prediction_intervals.json`)

```json
{
  "80%": {
    "coverage": 0.823,
    "average_width": 34567.89,
    "average_width_pct": 21.34,
    "target_coverage": 0.8
  },
  "90%": {
    "coverage": 0.912,
    "average_width": 45678.90,
    "average_width_pct": 28.12,
    "target_coverage": 0.9
  }
}
```

---

## 📊 SIGNIFICATO DEI RISULTATI

### 1. **Metriche di Regressione**

#### Metriche su Scala Trasformata (`metrics_test`)
- **R²**: % di varianza spiegata (0-1, meglio se vicino a 1)
- **RMSE**: Root Mean Squared Error - errore medio quadratico (più basso è meglio)
- **MSE**: Mean Squared Error - RMSE al quadrato
- **MAE**: Mean Absolute Error - errore medio assoluto (più robusto a outlier)
- **MAPE**: Mean Absolute Percentage Error - errore % medio
- **MedAE**: Median Absolute Error - mediana dell'errore assoluto (robusto)
- **Explained Variance**: Varianza spiegata (simile a R²)

#### Metriche su Scala Originale (`metrics_test_original`)
**⚠️ IMPORTANTE**: Queste sono le metriche "reali" in EURO per interpretazione business!

- **R² original**: Performance su scala EURO (es. 0.8989 = 89.89% varianza spiegata)
- **RMSE original**: Errore medio in EURO (es. 19234.56€ = errore medio ±19k€)
- **MAE original**: Errore assoluto medio in EURO
- **MAPE_floor**: MAPE con floor per evitare divisioni per zero su valori piccoli

**Interpretazione Pratica**:
```
RMSE = 19234.56€
→ In media, le predizioni sbagliano di ±19k€
→ Su un immobile da 200k€, errore ~9.6%
→ Su un immobile da 500k€, errore ~3.8%
```

### 2. **Diagnostiche di Overfitting**

#### Gap Metrics (train - test)
```json
"gap_r2": 0.0511
```
- **Interpretazione**: Il modello performa 5.11% peggio su test rispetto a train
- **Soglia OK**: < 0.05 (5%)
- **Moderato**: 0.05 - 0.10
- **Alto**: > 0.10 → modello troppo overfit

#### Ratio Metrics (test / train)
```json
"ratio_rmse": 1.5034
```
- **Interpretazione**: L'errore su test è 50% più alto che su train
- **Soglia OK**: 1.0 - 1.2 (20% di degradazione)
- **Moderato**: 1.2 - 1.5
- **Alto**: > 1.5 → modello troppo overfit

### 3. **SHAP Feature Importance**

I grafici SHAP mostrano:
- **Beeswarm plot**: Contributo di ogni feature per ogni predizione
  - Colore = valore feature (rosso alto, blu basso)
  - Asse X = SHAP value (impatto sulla predizione)
  - Feature ordinate per importanza
  
- **Bar plot**: Importanza media assoluta per feature
  - Più alta la barra = più importante la feature

**Esempio Interpretazione**:
```
AI_Superficie: SHAP = +0.3 (in scala log)
→ Un aumento di superficie contribuisce positivamente al prezzo
→ È la feature più importante per il modello
```

### 4. **Group Metrics**

Permettono di identificare **bias geografici/categoriali**:

```csv
group,count,r2,rmse,mae
Zona_B1,1234,0.91,15234.56,10123.45
Zona_C3,987,0.67,35678.90,25123.45
```

**Interpretazione**:
- Zona B1: ottima performance (R² = 0.91, RMSE = 15k€)
- Zona C3: scarsa performance (R² = 0.67, RMSE = 35k€)
- **Azione**: Investigare Zona C3 (dati mancanti? outlier? feature mancanti?)

### 5. **Price Band Metrics**

Mostrano performance per **fascia di prezzo**:

```csv
group,r2,rmse,mape
PREZZO_(50k, 150k],0.85,8901.23,0.067
PREZZO_(400k, 800k],0.87,45678.90,0.108
```

**Interpretazione**:
- Fasce basse: R² simile, ma RMSE più basso (errore assoluto minore)
- Fasce alte: R² simile, ma RMSE più alto (errore assoluto maggiore)
- MAPE cresce con il prezzo → il modello sbaglia % più alta sugli immobili costosi

### 6. **Drift Detection**

**PSI (Population Stability Index)**:
- < 0.1: Nessun drift significativo
- 0.1 - 0.15: Drift moderato
- \> 0.15: Drift significativo → **modello da ritrainare**

**KS Test (Kolmogorov-Smirnov)**:
- p < 0.05: Distribuzione train vs test significativamente diversa
- **Azione**: Verificare se il test set è rappresentativo

**Esempio Pratico**:
```json
"AI_Vani": {
  "psi": 0.187,      → DRIFT ALTO!
  "ks_pvalue": 0.012 → DISTRIBUZIONE DIVERSA!
}
```
→ La distribuzione del numero di vani è cambiata tra train e test
→ Possibile causa: cambiamento temporale, bias nel sampling
→ Azione: Ritrainare con dati più recenti o investigare il motivo

### 7. **Prediction Intervals**

Quantificano l'**incertezza** della predizione:

```json
"90%": {
  "coverage": 0.912,          → 91.2% dei valori veri cadono nell'intervallo
  "average_width": 45678.90,  → Intervallo medio di ±45k€
  "target_coverage": 0.9      → Target era 90%
}
```

**Interpretazione**:
- Coverage vicino al target (91.2% vs 90%) → intervalli ben calibrati
- Average width = incertezza media del modello
- Usare per comunicare la **confidenza** della stima al cliente

**Esempio Business**:
```
Predizione: 200k€
Intervallo 90%: [175k€, 225k€]
→ "Siamo confidenti al 90% che il prezzo reale sia tra 175k e 225k"
```

### 8. **Worst Predictions**

File `{model}_worst_predictions.csv` contiene i record con errori più grandi:

```csv
true,predicted,residual,abs_residual,pct_error
450000.0,280000.0,-170000.0,170000.0,37.78
```

**Uso**:
1. Identificare pattern comuni nei worst predictions
2. Verificare se ci sono outlier non rilevati
3. Cercare feature mancanti (es. "tutti gli errori grandi sono in zona X")
4. Migliorare il modello per questi casi specifici

---

## 🧪 CONFIGURAZIONI DA PROVARE (ESPERIMENTI)

### 🎯 Baseline: Configurazione Attuale

**File**: `config/config.yaml`

**Setup**:
```yaml
target:
  transform: boxcox
database:
  use_poi: true
  use_ztl: true
  selected_aliases: ['A', 'AI', 'PC', 'ISC', 'II', 'PC_OZ', 'OZ', 'OV', 'C1', 'C2']
profiles:
  tree: enabled: true
  catboost: enabled: true
training:
  models: [rf, gbr, hgbt, xgboost, lightgbm, catboost]
  trials: 100
```

**Metriche Attese** (baseline per confronti):
- Test R² (original): ~0.89-0.91
- Test RMSE (original): ~18k-20k€
- Test MAPE: ~12-13%

---

### 📋 ESPERIMENTI CONSIGLIATI

#### **CATEGORIA A: ABLATION STUDIES (Rimuovere Features)**

---

### 🧪 **Esperimento A1: Senza Trasformazione Target**

**Obiettivo**: Valutare impatto della trasformazione Box-Cox sul target

**Config**: `config/config_no_transform.yaml`

```yaml
target:
  transform: none  # ← CAMBIO PRINCIPALE

# Resto identico a config.yaml
database:
  use_poi: true
  use_ztl: true
profiles:
  tree: {enabled: true}
  catboost: {enabled: true}
training:
  models: {rf: {enabled: true}, xgboost: {enabled: true}, lightgbm: {enabled: true}, catboost: {enabled: true}}
```

**Ipotesi**:
- Target senza trasformazione → modelli meno performanti (R² più basso)
- RMSE potrebbe essere più alto su scala originale
- Residui potrebbero essere asimmetrici (code lunghe)

**Metriche da Confrontare**:
- R² original: atteso < 0.85 (peggiore di baseline)
- RMSE original: atteso > 22k€
- Residual distribution: più skewed

**Come Eseguire**:
```bash
python main.py --config config/config_no_transform.yaml --steps preprocessing training evaluation
```

---

### 🧪 **Esperimento A2: Senza POI (Points of Interest)**

**Obiettivo**: Valutare impatto delle feature POI (scuole, ospedali, fermate, ecc.)

**Config**: `config/config_no_poi.yaml`

```yaml
database:
  use_poi: false  # ← CAMBIO PRINCIPALE
  use_ztl: true
  selected_aliases: ['A', 'AI', 'PC', 'ISC', 'II', 'PC_OZ', 'OZ', 'OV', 'C1', 'C2']

target:
  transform: boxcox

profiles:
  tree: {enabled: true}
  catboost: {enabled: true}
training:
  models: {rf: {enabled: true}, xgboost: {enabled: true}, lightgbm: {enabled: true}, catboost: {enabled: true}}
```

**Ipotesi**:
- POI aggiungono informazione sul contesto geografico/urbanistico
- Rimuoverli → calo di performance (specialmente in zone urbane)
- Group metrics per zona potrebbero mostrare bias maggiore

**Metriche da Confrontare**:
- R² original: atteso ~0.87-0.88 (calo ~1-2%)
- RMSE original: atteso ~20-22k€ (aumento ~5-10%)
- Group metrics per zona: verificare se alcune zone peggiorano molto

**SHAP Analysis**:
- Nella configurazione baseline, verificare feature importance di POI
- Se POI sono importanti → esperimento conferma il loro valore

---

### 🧪 **Esperimento A3: Senza ZTL (Zone a Traffico Limitato)**

**Obiettivo**: Valutare impatto della feature binaria ZTL

**Config**: `config/config_no_ztl.yaml`

```yaml
database:
  use_poi: true
  use_ztl: false  # ← CAMBIO PRINCIPALE
  selected_aliases: ['A', 'AI', 'PC', 'ISC', 'II', 'PC_OZ', 'OZ', 'OV', 'C1', 'C2']

target:
  transform: boxcox

profiles:
  tree: {enabled: true}
  catboost: {enabled: true}
training:
  models: {rf: {enabled: true}, xgboost: {enabled: true}, lightgbm: {enabled: true}, catboost: {enabled: true}}
```

**Ipotesi**:
- ZTL ha impatto minore rispetto a POI (è una feature binaria singola)
- Calo di performance atteso: < 0.5%
- Se calo è alto → ZTL è un proxy importante per "centro città"

**Metriche da Confrontare**:
- R² original: atteso ~0.89-0.90 (calo minimo)
- RMSE original: atteso ~18.5-19.5k€

---

### 🧪 **Esperimento A4: Senza CENED (Certificati Energetici)**

**Obiettivo**: Valutare impatto delle view `attiimmobili_cened1` e `attiimmobili_cened2`

**Config**: `config/config_no_cened.yaml`

```yaml
database:
  use_poi: true
  use_ztl: true
  selected_aliases: ['A', 'AI', 'PC', 'ISC', 'II', 'PC_OZ', 'OZ', 'OV']  # ← RIMOSSI C1, C2

target:
  transform: boxcox

profiles:
  tree: {enabled: true}
  catboost: {enabled: true}
training:
  models: {rf: {enabled: true}, xgboost: {enabled: true}, lightgbm: {enabled: true}, catboost: {enabled: true}}
```

**Ipotesi**:
- CENED contiene info su efficienza energetica (importante per valutazione immobile)
- Rimuoverlo → calo performance, specialmente su immobili moderni
- Group metrics per categoria catastale potrebbero mostrare bias

**Metriche da Confrontare**:
- R² original: atteso ~0.88-0.89 (calo ~0.5-1%)
- Worst predictions: verificare se aumentano gli errori su immobili con certificazione

**Nota**: Verificare con SHAP nella baseline quante feature CENED sono importanti

---

### 🧪 **Esperimento A5: Senza POI + ZTL + CENED (Solo Dati Base)**

**Obiettivo**: Baseline "minimalista" - solo dati immobiliari base

**Config**: `config/config_minimal.yaml`

```yaml
database:
  use_poi: false  # ← RIMOSSO
  use_ztl: false  # ← RIMOSSO
  selected_aliases: ['A', 'AI', 'PC', 'ISC', 'II', 'PC_OZ', 'OZ', 'OV']  # ← RIMOSSI C1, C2

target:
  transform: boxcox

profiles:
  tree: {enabled: true}
  catboost: {enabled: true}
training:
  models: {rf: {enabled: true}, xgboost: {enabled: true}, lightgbm: {enabled: true}, catboost: {enabled: true}}
```

**Ipotesi**:
- Configurazione più semplice, meno overfitting potenziale
- Ma: performance peggiore per mancanza di contesto
- Utile per capire il "valore incrementale" delle feature aggiunte

**Metriche Attese**:
- R² original: atteso ~0.85-0.87 (calo ~3-5%)
- RMSE original: atteso ~22-25k€
- Overfitting gap potrebbe ridursi (meno feature = meno overfitting)

---

#### **CATEGORIA B: VARIAZIONI TARGET**

---

### 🧪 **Esperimento B1: Target = AI_Prezzo_MQ (Prezzo al Metro Quadro)**

**Obiettivo**: Predire prezzo al m² invece di prezzo totale

**Config**: `config/config_target_mq.yaml`

```yaml
target:
  column_candidates: ['AI_Prezzo_MQ']  # ← CAMBIO PRINCIPALE
  transform: boxcox

database:
  use_poi: true
  use_ztl: true
  selected_aliases: ['A', 'AI', 'PC', 'ISC', 'II', 'PC_OZ', 'OZ', 'OV', 'C1', 'C2']

feature_pruning:
  include_ai_superficie: false  # ← IMPORTANTE: rimuovere superficie dalle feature!

profiles:
  tree: {enabled: true}
  catboost: {enabled: true}
training:
  models: {rf: {enabled: true}, xgboost: {enabled: true}, lightgbm: {enabled: true}, catboost: {enabled: true}}
```

**Ipotesi**:
- Prezzo/m² potrebbe essere più uniforme tra immobili (meno varianza)
- R² potrebbe essere più alto (target normalizzato per dimensione)
- MAPE potrebbe migliorare

**Metriche da Confrontare**:
- R² original: atteso ~0.91-0.93 (potenziale miglioramento)
- RMSE in €/m²: difficile confrontare con baseline (scale diverse)
- MAPE: atteso ~10-11% (miglioramento)

**Nota**: Per confronto apples-to-apples, moltiplicare predizioni per superficie e confrontare RMSE totale

---

### 🧪 **Esperimento B2: Target con Log Transform (invece di Box-Cox)**

**Obiettivo**: Confrontare log semplice vs Box-Cox parametrico

**Config**: `config/config_log_transform.yaml`

```yaml
target:
  column_candidates: ['AI_Prezzo_Ridistribuito']
  transform: log  # ← CAMBIO PRINCIPALE (log1p)

database:
  use_poi: true
  use_ztl: true
  selected_aliases: ['A', 'AI', 'PC', 'ISC', 'II', 'PC_OZ', 'OZ', 'OV', 'C1', 'C2']

profiles:
  tree: {enabled: true}
  catboost: {enabled: true}
training:
  models: {rf: {enabled: true}, xgboost: {enabled: true}, lightgbm: {enabled: true}, catboost: {enabled: true}}
```

**Ipotesi**:
- Log è più semplice e interpretabile
- Box-Cox è più flessibile (trova lambda ottimale)
- Differenza attesa: < 0.5% in R²

**Metriche da Confrontare**:
- R² original: atteso ~0.89-0.90 (simile o leggermente peggiore)
- Smearing factor: confrontare con Box-Cox
- Residual distribution: verificare normalità

---

### 🧪 **Esperimento B3: Target con Yeo-Johnson (alternativa a Box-Cox)**

**Obiettivo**: Yeo-Johnson supporta valori negativi (più robusto)

**Config**: `config/config_yeojohnson.yaml`

```yaml
target:
  column_candidates: ['AI_Prezzo_Ridistribuito']
  transform: yeojohnson  # ← CAMBIO PRINCIPALE

database:
  use_poi: true
  use_ztl: true
  selected_aliases: ['A', 'AI', 'PC', 'ISC', 'II', 'PC_OZ', 'OZ', 'OV', 'C1', 'C2']

profiles:
  tree: {enabled: true}
  catboost: {enabled: true}
training:
  models: {rf: {enabled: true}, xgboost: {enabled: true}, lightgbm: {enabled: true}, catboost: {enabled: true}}
```

**Ipotesi**:
- Performance simile a Box-Cox
- Più robusto se ci sono outlier negativi post-preprocessing

**Metriche da Confrontare**:
- R² original: atteso ~0.89-0.91 (simile)
- Lambda: confrontare con Box-Cox lambda

---

#### **CATEGORIA C: VARIAZIONI PREPROCESSING**

---

### 🧪 **Esperimento C1: Outlier Detection più Aggressivo**

**Obiettivo**: Rimuovere più outlier dal training set

**Config**: `config/config_outlier_aggressive.yaml`

```yaml
outliers:
  method: ensemble
  z_thresh: 2.5       # ← Più stretto (default: 3.0)
  iqr_factor: 1.0     # ← Più stretto (default: 1.5)
  iso_forest_contamination: 0.05  # ← Più alto (default: 0.02)
  group_by_col: 'AI_IdTipologiaEdilizia'
  min_group_size: 30

target:
  transform: boxcox

database:
  use_poi: true
  use_ztl: true

profiles:
  tree: {enabled: true}
  catboost: {enabled: true}
training:
  models: {rf: {enabled: true}, xgboost: {enabled: true}, lightgbm: {enabled: true}, catboost: {enabled: true}}
```

**Ipotesi**:
- Training set più pulito → modelli più semplici
- R² train potrebbe calare (meno dati)
- R² test potrebbe migliorare (meno noise)
- Overfitting gap potrebbe ridursi

**Metriche da Confrontare**:
- % outlier rimossi: atteso ~5-8% (vs ~2% baseline)
- R² train: atteso ~0.94 (calo da ~0.95)
- R² test: atteso ~0.90-0.92 (potenziale miglioramento)
- Overfitting ratio_rmse: atteso ~1.3-1.4 (miglioramento da ~1.5)

---

### 🧪 **Esperimento C2: Senza Winsorization**

**Obiettivo**: Valutare impatto del clipping dei quantili estremi

**Config**: `config/config_no_winsor.yaml`

```yaml
winsorization:
  enabled: false  # ← CAMBIO PRINCIPALE

target:
  transform: boxcox

database:
  use_poi: true
  use_ztl: true

profiles:
  tree:
    enabled: true
    winsorization:
      enabled: false  # ← Assicurarsi che sia disabilitato anche a livello profilo
  catboost:
    enabled: true

training:
  models: {rf: {enabled: true}, xgboost: {enabled: true}, lightgbm: {enabled: true}, catboost: {enabled: true}}
```

**Ipotesi**:
- Senza winsor → feature numeriche con valori estremi
- Modelli tree-based: impatto minimo (gestiscono bene outlier)
- Profilo `scaled` (se abilitato): impatto maggiore su scaling

**Metriche da Confrontare**:
- R² test: atteso simile (~0.89-0.90)
- Worst predictions: potrebbero aumentare (più sensibilità a estremi)

---

### 🧪 **Esperimento C3: Split Temporale più Conservativo (Test più Recente)**

**Obiettivo**: Test set più recente per simulare predizione su dati nuovi

**Config**: `config/config_temporal_recent.yaml`

```yaml
temporal_split:
  mode: fraction
  fraction:
    train: 0.6    # ← CAMBIO: meno training (default: 0.7)
    valid: 0.2
    # test: 0.2 (automatico)

target:
  transform: boxcox

database:
  use_poi: true
  use_ztl: true

profiles:
  tree: {enabled: true}
  catboost: {enabled: true}
training:
  models: {rf: {enabled: true}, xgboost: {enabled: true}, lightgbm: {enabled: true}, catboost: {enabled: true}}
```

**Ipotesi**:
- Test set più recente → potenziale drift temporale
- R² test potrebbe calare (dati più difficili)
- Drift detection dovrebbe segnalare più features

**Metriche da Confrontare**:
- R² test: atteso ~0.87-0.89 (potenziale calo)
- Drift alerts: atteso > 5 (vs ~3 baseline)
- Group metrics: verificare se bias temporale esiste

---

### 🧪 **Esperimento C4: Soglia Correlazione più Bassa (Mantenere più Feature)**

**Obiettivo**: Evitare pruning aggressivo di feature correlate

**Config**: `config/config_low_corr_threshold.yaml`

```yaml
correlation:
  numeric_threshold: 0.99  # ← Più permissivo (default: 0.95)

target:
  transform: boxcox

database:
  use_poi: true
  use_ztl: true

profiles:
  tree:
    enabled: true
    correlation:
      numeric_threshold: 0.99
  catboost:
    enabled: true
    correlation:
      numeric_threshold: 0.99

training:
  models: {rf: {enabled: true}, xgboost: {enabled: true}, lightgbm: {enabled: true}, catboost: {enabled: true}}
```

**Ipotesi**:
- Più feature → modelli più complessi
- Potenziale overfitting aumentato
- R² train potrebbe salire, R² test potrebbe calare

**Metriche da Confrontare**:
- Feature count: atteso +10-20 feature
- R² train: atteso ~0.96 (aumento)
- R² test: atteso ~0.88-0.89 (potenziale calo)
- Overfitting gap: atteso aumento

---

#### **CATEGORIA D: VARIAZIONI MODELLI**

---

### 🧪 **Esperimento D1: Solo Ensemble (No Modelli Singoli)**

**Obiettivo**: Verificare se ensemble battono sempre i singoli

**Config**: `config/config_ensemble_only.yaml`

```yaml
target:
  transform: boxcox

database:
  use_poi: true
  use_ztl: true

profiles:
  tree: {enabled: true}
  catboost: {enabled: true}

training:
  models:
    # Train tutti i modelli ma focus su ensemble
    rf: {enabled: true, trials: 50}
    xgboost: {enabled: true, trials: 50}
    lightgbm: {enabled: true, trials: 50}
    catboost: {enabled: true, trials: 50}
    hgbt: {enabled: true, trials: 50}
  ensembles:
    voting:
      enabled: true
      top_n: 5  # ← Usare tutti i top 5
      tune_weights: true
    stacking:
      enabled: true
      top_n: 5
      final_estimator: ridge
      cv_folds: 5
```

**Ipotesi**:
- Ensemble quasi sempre migliori dei singoli
- Stacking > Voting (meta-learner impara combinazione ottimale)

**Metriche da Confrontare**:
- Confrontare R² di voting, stacking vs best single model
- Atteso: stacking R² ~0.91-0.92 (migliore dei singoli)

---

### 🧪 **Esperimento D2: Solo XGBoost (Modello Singolo Best)**

**Obiettivo**: Training intensivo su un solo modello

**Config**: `config/config_xgboost_only.yaml`

```yaml
target:
  transform: boxcox

database:
  use_poi: true
  use_ztl: true

profiles:
  tree: {enabled: true}

training:
  trials_advanced: 500  # ← Molti più trial
  models:
    xgboost:
      enabled: true
      profile: tree
      trials: 500  # ← Search space esaustivo
      base_params: {}
      search_space:
        n_estimators: {type: int, low: 500, high: 2000}
        max_depth: {type: int, low: 3, high: 10}
        learning_rate: {type: float, low: 0.005, high: 0.1, log: true}
        subsample: {type: float, low: 0.5, high: 1.0}
        colsample_bytree: {type: float, low: 0.5, high: 1.0}
        min_child_weight: {type: float, low: 1.0, high: 15.0, log: true}
        reg_alpha: {type: float, low: 1e-5, high: 10.0, log: true}
        reg_lambda: {type: float, low: 1e-5, high: 10.0, log: true}
        gamma: {type: float, low: 1e-5, high: 10.0, log: true}
  ensembles:
    voting: {enabled: false}
    stacking: {enabled: false}
```

**Ipotesi**:
- Tuning intensivo può far emergere configurazioni migliori
- R² test atteso: ~0.90-0.92
- Tempo: ~2-3 ore (vs ~30min baseline)

**Metriche da Confrontare**:
- R² vs baseline XGBoost con 100 trial
- Verificare se best_params sono molto diversi

---

### 🧪 **Esperimento D3: Modelli Lineari vs Tree-Based**

**Obiettivo**: Confrontare performance di modelli lineari (con profilo `scaled`)

**Config**: `config/config_linear_models.yaml`

```yaml
target:
  transform: boxcox

database:
  use_poi: true
  use_ztl: true

profiles:
  scaled:  # ← Profilo per modelli lineari
    enabled: true
    output_prefix: scaled
    winsorization: {enabled: true, lower_quantile: 0.01, upper_quantile: 0.99}
    scaling: {scaler_type: standard}
    pca: {enabled: true, n_components: 0.95}

training:
  models:
    ridge:
      enabled: true
      profile: scaled
      trials: 100
    lasso:
      enabled: true
      profile: scaled
      trials: 100
    elasticnet:
      enabled: true
      profile: scaled
      trials: 100
    # Per confronto
    xgboost:
      enabled: true
      profile: tree
      trials: 100
```

**Ipotesi**:
- Modelli lineari: meno performanti ma più interpretabili
- R² atteso: ~0.80-0.85 (vs ~0.90 tree-based)
- Più veloci da trainare

**Metriche da Confrontare**:
- R² test: confrontare ridge/lasso/elasticnet vs xgboost
- Training time
- Interpretabilità: coefficienti lineari vs SHAP

---

### 🧪 **Esperimento D4: CatBoost con Iterazioni Elevate**

**Obiettivo**: Sfruttare appieno la potenza di CatBoost

**Config**: `config/config_catboost_heavy.yaml`

```yaml
target:
  transform: boxcox

database:
  use_poi: true
  use_ztl: true

profiles:
  catboost:
    enabled: true
    output_prefix: catboost

training:
  models:
    catboost:
      enabled: true
      profile: catboost
      trials: 200  # ← Più trial
      base_params:
        allow_writing_files: false
        iterations: 2000  # ← Baseline alto
      search_space:
        iterations: {type: int, low: 1000, high: 3000}  # ← Search space alto
        depth: {type: int, low: 4, high: 10}
        learning_rate: {type: float, low: 0.001, high: 0.1, log: true}
        l2_leaf_reg: {type: float, low: 1.0, high: 20.0}
        bagging_temperature: {type: float, low: 0.0, high: 5.0}
        border_count: {type: int, low: 32, high: 255}
  ensembles:
    voting: {enabled: false}
    stacking: {enabled: false}
```

**Ipotesi**:
- CatBoost con più iterazioni può migliorare performance
- R² test atteso: ~0.91-0.93
- Rischio overfitting con troppe iterazioni

**Metriche da Confrontare**:
- R² vs baseline CatBoost
- Overfitting gap: monitorare che non esploda

---

#### **CATEGORIA E: SPLIT E VALIDAZIONE**

---

### 🧪 **Esperimento E1: Cross-Validation K-Fold (No Validation Set)**

**Obiettivo**: Usare tutto il train set con CV invece di hold-out validation

**Config**: `config/config_kfold_cv.yaml`

```yaml
temporal_split:
  mode: fraction
  fraction:
    train: 0.8    # ← Più train (no validation hold-out)
    valid: 0.0    # ← NESSUNA VALIDATION!
    # test: 0.2

training:
  cv_when_no_val:
    enabled: true
    kind: kfold    # ← K-Fold CV
    n_splits: 5
    shuffle: true

target:
  transform: boxcox

database:
  use_poi: true
  use_ztl: true

profiles:
  tree: {enabled: true}
  catboost: {enabled: true}

training:
  models: {rf: {enabled: true}, xgboost: {enabled: true}, lightgbm: {enabled: true}, catboost: {enabled: true}}
```

**Ipotesi**:
- Più dati per training → modelli potenzialmente migliori
- CV con K=5 → tuning più robusto (5 fold invece di 1 validation set)
- Tempo: ~5x più lungo (5 fold)

**Metriche da Confrontare**:
- R² test: atteso ~0.90-0.91 (potenziale miglioramento)
- Std deviation dei CV scores: indicatore di stabilità

---

### 🧪 **Esperimento E2: Temporal Split con Data Fissa (Non Fraction)**

**Obiettivo**: Test set = dati 2023+ (simulazione real-world)

**Config**: `config/config_temporal_date.yaml`

```yaml
temporal_split:
  mode: date  # ← CAMBIO: usa data invece di fraction
  date:
    test_start_year: 2023
    test_start_month: 1

target:
  transform: boxcox

database:
  use_poi: true
  use_ztl: true

profiles:
  tree: {enabled: true}
  catboost: {enabled: true}

training:
  models: {rf: {enabled: true}, xgboost: {enabled: true}, lightgbm: {enabled: true}, catboost: {enabled: true}}
```

**Ipotesi**:
- Test set = dati recenti → valutazione più realistica
- Potenziale drift temporale (prezzi crescenti, inflazione)
- R² test atteso: ~0.87-0.89 (potenziale calo)

**Metriche da Confrontare**:
- R² test: confrontare con fraction mode
- Drift detection: atteso più alerts
- Price band metrics: verificare se prezzi sono shifted

---

---

### 📊 ESPERIMENTI SUMMARY TABLE

| Esperimento | Config File | Obiettivo | Tempo Atteso | R² Test Atteso | Priorità |
|------------|-------------|-----------|--------------|----------------|----------|
| **A1** | `config_no_transform.yaml` | Ablation target transform | 30min | 0.85-0.87 | 🔥 ALTA |
| **A2** | `config_no_poi.yaml` | Ablation POI features | 25min | 0.87-0.88 | 🔥 ALTA |
| **A3** | `config_no_ztl.yaml` | Ablation ZTL feature | 25min | 0.89-0.90 | 🟡 MEDIA |
| **A4** | `config_no_cened.yaml` | Ablation CENED views | 25min | 0.88-0.89 | 🟡 MEDIA |
| **A5** | `config_minimal.yaml` | Solo dati base (no enrichment) | 20min | 0.85-0.87 | 🔥 ALTA |
| **B1** | `config_target_mq.yaml` | Target = Prezzo/m² | 30min | 0.91-0.93 | 🔥 ALTA |
| **B2** | `config_log_transform.yaml` | Log vs Box-Cox | 30min | 0.89-0.90 | 🟡 MEDIA |
| **B3** | `config_yeojohnson.yaml` | Yeo-Johnson transform | 30min | 0.89-0.91 | 🟢 BASSA |
| **C1** | `config_outlier_aggressive.yaml` | Outlier removal più aggressivo | 30min | 0.90-0.92 | 🟡 MEDIA |
| **C2** | `config_no_winsor.yaml` | No winsorization | 30min | 0.89-0.90 | 🟢 BASSA |
| **C3** | `config_temporal_recent.yaml` | Test set più recente | 30min | 0.87-0.89 | 🔥 ALTA |
| **C4** | `config_low_corr_threshold.yaml` | Mantenere più feature | 35min | 0.88-0.89 | 🟢 BASSA |
| **D1** | `config_ensemble_only.yaml` | Focus su ensemble | 40min | 0.91-0.92 | 🟡 MEDIA |
| **D2** | `config_xgboost_only.yaml` | Tuning intensivo XGBoost | 2-3h | 0.90-0.92 | 🔥 ALTA |
| **D3** | `config_linear_models.yaml` | Lineari vs Tree-based | 25min | 0.80-0.85 | 🟢 BASSA |
| **D4** | `config_catboost_heavy.yaml` | CatBoost con molte iterazioni | 1-2h | 0.91-0.93 | 🟡 MEDIA |
| **E1** | `config_kfold_cv.yaml` | K-Fold CV (no validation hold-out) | 2-3h | 0.90-0.91 | 🟡 MEDIA |
| **E2** | `config_temporal_date.yaml` | Split temporale con data fissa | 30min | 0.87-0.89 | 🔥 ALTA |

---

### 🎯 ESPERIMENTI CONSIGLIATI - FASE 1 (Quick Wins)

Eseguire questi **5 esperimenti** per massimo impatto con minimo tempo:

1. **A1** - No Transform: capire valore trasformazione target
2. **A2** - No POI: capire valore features geografiche
3. **A5** - Minimal: baseline senza enrichment
4. **B1** - Target MQ: confrontare prezzo totale vs prezzo/m²
5. **C3** - Recent Test: valutare drift temporale

**Tempo totale Fase 1**: ~2.5 ore  
**Output**: Report comparativo con 6 configurazioni (baseline + 5 esperimenti)

---

### 🚀 ESPERIMENTI CONSIGLIATI - FASE 2 (Deep Dive)

Dopo Fase 1, eseguire questi per ottimizzazione:

1. **D2** - XGBoost Heavy: tuning intensivo best model
2. **D4** - CatBoost Heavy: sfruttare categoriche native
3. **E1** - K-Fold CV: validazione più robusta
4. **D1** - Ensemble Focus: ottimizzare combinazione modelli

**Tempo totale Fase 2**: ~8-10 ore  
**Output**: Best model production-ready

---

## 📊 COME CONFRONTARE I RISULTATI

### 1. **Creare Tabella Comparativa**

```python
import pandas as pd
import json
from pathlib import Path

results = []
for exp_name in ['baseline', 'no_poi', 'no_transform', 'target_mq', 'minimal']:
    model_dir = Path(f'models_{exp_name}')  # Usare models_dir diversi per esperimento
    summary = json.loads((model_dir / 'summary.json').read_text())
    
    # Best single model
    best_model = max(summary['models'].items(), key=lambda x: x[1]['metrics_test_original']['r2'])
    
    results.append({
        'Experiment': exp_name,
        'Best_Model': best_model[0],
        'R2_test_orig': best_model[1]['metrics_test_original']['r2'],
        'RMSE_test_orig': best_model[1]['metrics_test_original']['rmse'],
        'MAE_test_orig': best_model[1]['metrics_test_original']['mae'],
        'MAPE_floor': best_model[1]['metrics_test_original'].get('mape_floor', None),
        'Overfitting_gap_R2': best_model[1]['overfit']['gap_r2'],
        'Overfitting_ratio_RMSE': best_model[1]['overfit']['ratio_rmse']
    })

df_comparison = pd.DataFrame(results).sort_values('R2_test_orig', ascending=False)
df_comparison.to_csv('experiments_comparison.csv', index=False)
print(df_comparison)
```

**Output Esempio**:

```csv
Experiment,Best_Model,R2_test_orig,RMSE_test_orig,MAE_test_orig,MAPE_floor,Overfitting_gap_R2,Overfitting_ratio_RMSE
baseline,xgboost,0.9012,18567.89,13456.78,0.1259,0.0511,1.5034
target_mq,xgboost,0.9201,16234.56,11890.12,0.1089,0.0489,1.4523
no_poi,lightgbm,0.8789,21234.56,15678.90,0.1456,0.0612,1.6234
no_transform,xgboost,0.8567,23456.78,17890.12,0.1678,0.0734,1.7890
minimal,catboost,0.8456,25678.90,19123.45,0.1789,0.0678,1.6123
```

### 2. **Visualizzazione Comparativa**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# R² Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: R² Test
df_comparison.plot.bar(x='Experiment', y='R2_test_orig', ax=axes[0], legend=False)
axes[0].set_title('R² Test (Original Scale)', fontsize=14)
axes[0].set_ylabel('R²')
axes[0].axhline(y=0.90, color='r', linestyle='--', label='Target: 0.90')
axes[0].legend()

# Plot 2: RMSE Test
df_comparison.plot.bar(x='Experiment', y='RMSE_test_orig', ax=axes[1], legend=False, color='orange')
axes[1].set_title('RMSE Test (€)', fontsize=14)
axes[1].set_ylabel('RMSE (€)')
axes[1].axhline(y=20000, color='r', linestyle='--', label='Target: 20k€')
axes[1].legend()

# Plot 3: Overfitting Gap
df_comparison.plot.bar(x='Experiment', y='Overfitting_gap_R2', ax=axes[2], legend=False, color='green')
axes[2].set_title('Overfitting Gap (R²)', fontsize=14)
axes[2].set_ylabel('Gap R² (train - test)')
axes[2].axhline(y=0.05, color='r', linestyle='--', label='Threshold: 0.05')
axes[2].legend()

plt.tight_layout()
plt.savefig('experiments_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3. **Feature Importance Comparison (SHAP)**

```python
# Confrontare top 10 feature per ogni esperimento
import numpy as np

shap_comparison = {}
for exp_name in ['baseline', 'no_poi', 'target_mq']:
    model_dir = Path(f'models_{exp_name}/xgboost/shap')
    shap_values = np.load(model_dir / 'shap_values.npy', allow_pickle=True)
    feature_names = pd.read_parquet(model_dir.parent / 'shap_sample.parquet').columns
    
    # Mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_10_idx = np.argsort(mean_abs_shap)[-10:][::-1]
    
    shap_comparison[exp_name] = {
        'features': [feature_names[i] for i in top_10_idx],
        'importance': mean_abs_shap[top_10_idx]
    }

# Visualize
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(10)
width = 0.25

for i, (exp_name, data) in enumerate(shap_comparison.items()):
    ax.bar(x + i*width, data['importance'], width, label=exp_name)

ax.set_xlabel('Feature Rank')
ax.set_ylabel('Mean |SHAP value|')
ax.set_title('Top 10 Features Comparison Across Experiments')
ax.legend()
plt.tight_layout()
plt.savefig('shap_comparison.png', dpi=150)
plt.show()
```

---

## 🛠️ SCRIPT DI AUTOMAZIONE ESPERIMENTI

### `run_experiments.sh`

```bash
#!/bin/bash

# Script per eseguire automaticamente tutti gli esperimenti
# Usage: ./run_experiments.sh

set -e  # Exit on error

BASE_CONFIG="config/config.yaml"
EXPERIMENTS_DIR="experiments_results"
mkdir -p "$EXPERIMENTS_DIR"

# Array di esperimenti: (nome, config_file, models_dir)
declare -a EXPERIMENTS=(
    "baseline:config/config.yaml:models_baseline"
    "no_poi:config/config_no_poi.yaml:models_no_poi"
    "no_transform:config/config_no_transform.yaml:models_no_transform"
    "target_mq:config/config_target_mq.yaml:models_target_mq"
    "minimal:config/config_minimal.yaml:models_minimal"
)

echo "========================================="
echo "STIMATRIX EXPERIMENTS BATCH RUN"
echo "========================================="
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r name config_file models_dir <<< "$exp"
    
    echo ">>> Running experiment: $name"
    echo "    Config: $config_file"
    echo "    Output: $models_dir"
    echo ""
    
    # Modificare temporaneamente la config per usare models_dir custom
    export MODELS_DIR="$models_dir"
    
    # Eseguire la pipeline
    python main.py --config "$config_file" --steps preprocessing training evaluation --force-reload
    
    # Copiare risultati in experiments_results
    cp "$models_dir/summary.json" "$EXPERIMENTS_DIR/summary_${name}.json"
    cp "$models_dir/validation_results.csv" "$EXPERIMENTS_DIR/validation_${name}.csv"
    
    echo "✅ Completed: $name"
    echo ""
done

echo "========================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "========================================="
echo ""
echo "Results saved in: $EXPERIMENTS_DIR/"
echo ""
echo "To generate comparison report, run:"
echo "  python scripts/compare_experiments.py $EXPERIMENTS_DIR/"
```

### `scripts/compare_experiments.py`

```python
#!/usr/bin/env python3
"""
Compare results from multiple experiments and generate report.

Usage:
    python scripts/compare_experiments.py experiments_results/
"""

import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_experiment_summary(summary_path: Path) -> dict:
    """Load summary.json from experiment."""
    with open(summary_path) as f:
        return json.load(f)

def extract_best_model(summary: dict) -> dict:
    """Extract best model metrics from summary."""
    models = summary.get('models', {})
    if not models:
        return None
    
    # Find best by R² test original
    best_key = max(
        models.keys(),
        key=lambda k: models[k].get('metrics_test_original', {}).get('r2', -999)
    )
    
    best = models[best_key]
    return {
        'model_key': best_key,
        'r2_test_orig': best['metrics_test_original']['r2'],
        'rmse_test_orig': best['metrics_test_original']['rmse'],
        'mae_test_orig': best['metrics_test_original']['mae'],
        'mape_floor': best['metrics_test_original'].get('mape_floor', None),
        'overfit_gap_r2': best['overfit']['gap_r2'],
        'overfit_ratio_rmse': best['overfit']['ratio_rmse']
    }

def main(experiments_dir: str):
    """Generate comparison report."""
    exp_dir = Path(experiments_dir)
    
    # Find all summary files
    summaries = list(exp_dir.glob('summary_*.json'))
    print(f"Found {len(summaries)} experiments")
    
    # Extract results
    results = []
    for summary_path in summaries:
        exp_name = summary_path.stem.replace('summary_', '')
        summary = load_experiment_summary(summary_path)
        best = extract_best_model(summary)
        
        if best:
            results.append({
                'Experiment': exp_name,
                **best
            })
    
    # Create DataFrame
    df = pd.DataFrame(results).sort_values('r2_test_orig', ascending=False)
    
    # Save CSV
    output_csv = exp_dir / 'comparison.csv'
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved comparison table: {output_csv}")
    
    # Print table
    print("\n" + "="*80)
    print("EXPERIMENTS COMPARISON")
    print("="*80 + "\n")
    print(df.to_string(index=False))
    print("\n")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: R² Test
    df.plot.bar(x='Experiment', y='r2_test_orig', ax=axes[0, 0], legend=False, color='steelblue')
    axes[0, 0].set_title('R² Test (Original Scale)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('R²')
    axes[0, 0].axhline(y=0.90, color='red', linestyle='--', linewidth=2, label='Target: 0.90')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: RMSE Test
    df.plot.bar(x='Experiment', y='rmse_test_orig', ax=axes[0, 1], legend=False, color='darkorange')
    axes[0, 1].set_title('RMSE Test (€)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('RMSE (€)')
    axes[0, 1].axhline(y=20000, color='red', linestyle='--', linewidth=2, label='Target: 20k€')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: MAE Test
    df.plot.bar(x='Experiment', y='mae_test_orig', ax=axes[1, 0], legend=False, color='green')
    axes[1, 0].set_title('MAE Test (€)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('MAE (€)')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Overfitting Gap
    df.plot.bar(x='Experiment', y='overfit_gap_r2', ax=axes[1, 1], legend=False, color='purple')
    axes[1, 1].set_title('Overfitting Gap (R²)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Gap R² (train - test)')
    axes[1, 1].axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Threshold: 0.05')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Experiments Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_plot = exp_dir / 'comparison.png'
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comparison plot: {output_plot}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")
    print(f"Best R² Test:    {df['r2_test_orig'].max():.4f} ({df.loc[df['r2_test_orig'].idxmax(), 'Experiment']})")
    print(f"Worst R² Test:   {df['r2_test_orig'].min():.4f} ({df.loc[df['r2_test_orig'].idxmin(), 'Experiment']})")
    print(f"Best RMSE Test:  {df['rmse_test_orig'].min():.2f}€ ({df.loc[df['rmse_test_orig'].idxmin(), 'Experiment']})")
    print(f"Worst RMSE Test: {df['rmse_test_orig'].max():.2f}€ ({df.loc[df['rmse_test_orig'].idxmax(), 'Experiment']})")
    print()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python compare_experiments.py <experiments_dir>")
        sys.exit(1)
    
    main(sys.argv[1])
```

**Come Usare**:

```bash
# 1. Rendere eseguibile
chmod +x run_experiments.sh
chmod +x scripts/compare_experiments.py

# 2. Eseguire batch di esperimenti
./run_experiments.sh

# 3. Generare report comparativo
python scripts/compare_experiments.py experiments_results/

# Output:
# - experiments_results/comparison.csv
# - experiments_results/comparison.png
```

---

## 🎓 CONCLUSIONI E RACCOMANDAZIONI

### ✅ Cosa Funziona Bene

1. **Architettura Modulare**: Eccellente separazione dei concern
2. **Testing Coverage**: Suite test completa e ben strutturata
3. **Configurabilità**: YAML con env vars è ottimale per MLOps
4. **Experiment Tracking**: W&B integration nativa
5. **Target Transformation**: Box-Cox con Duan smearing è state-of-the-art
6. **Diagnostics**: Residual analysis, drift, PI sono production-ready
7. **Profili Multipli**: Supporto tree/catboost/scaled è flessibile
8. **Security**: Credential management e input validation sono robuste

### 🔧 Suggerimenti di Miglioramento (Non Urgenti)

#### 1. **Refactoring Codice**

```python
# File: src/preprocessing/pipeline.py
# Attuale: run_preprocessing() è lunga ~850 righe

# Suggerito: Spezzare in funzioni
def run_preprocessing(config: Dict[str, Any]) -> Path:
    # Load and clean
    df = load_and_clean_data(config)
    
    # Feature engineering
    df = extract_features(df, config)
    
    # Temporal split
    train_df, val_df, test_df = split_temporal(df, config)
    
    # Outliers and target transform
    train_df = remove_outliers(train_df, config)
    y_train, y_val, y_test, transform_meta = transform_targets(train_df, val_df, test_df, config)
    
    # Imputation
    X_train, X_val, X_test, imputers = fit_and_apply_imputation(train_df, val_df, test_df, config)
    
    # Process profiles
    for profile_name, profile_cfg in get_enabled_profiles(config):
        process_profile(profile_name, profile_cfg, X_train, y_train, X_val, y_val, X_test, y_test, config)
    
    return save_results(config)
```

#### 2. **Caching Intermedio**

```python
# Aggiungere cache per step preprocessing
@lru_cache(maxsize=1)
def load_and_clean_data(config_hash: str):
    # Cache del raw data cleaning
    cache_file = Path(f'cache/cleaned_{config_hash}.parquet')
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    # ... processing
    df.to_parquet(cache_file)
    return df
```

#### 3. **Monitoring Produzione**

```python
# File: src/utils/monitoring.py
def setup_monitoring(config):
    """Setup Prometheus/Grafana metrics."""
    from prometheus_client import Counter, Histogram
    
    PREDICTIONS = Counter('model_predictions_total', 'Total predictions')
    LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
    ERRORS = Counter('model_errors_total', 'Total errors')
    
    return PREDICTIONS, LATENCY, ERRORS
```

#### 4. **Deployment Ready**

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    unixodbc \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)"

# Run
CMD ["python", "main.py", "--config", "config/config.yaml", "--steps", "training"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  stimatrix-training:
    build: .
    environment:
      - SERVER=${DB_SERVER}
      - DATABASE=${DB_NAME}
      - DB_USER=${DB_USER}
      - DB_PASSWORD=${DB_PASSWORD}
      - WANDB_API_KEY=${WANDB_API_KEY}
      - WANDB_ENABLED=1
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    command: python main.py --steps training evaluation
```

### 🚀 Prossimi Passi Consigliati

**Short-term (1-2 settimane)**:
1. Eseguire esperimenti Fase 1 (A1, A2, A5, B1, C3)
2. Analizzare risultati e identificare best configuration
3. Documentare insights e pubblicare report interno

**Mid-term (1 mese)**:
1. Eseguire esperimenti Fase 2 (D2, D4, E1, D1)
2. Ottimizzare best model con tuning intensivo
3. Testare su holdout set finale (se disponibile)

**Long-term (2-3 mesi)**:
1. Deploy modello in produzione (API REST + monitoring)
2. Implementare CI/CD pipeline per retraining automatico
3. Setup A/B testing per confronto modelli in produzione

---

## 📚 RIFERIMENTI E RISORSE

### Documentazione Interna

- **README.md**: Guida completa della pipeline
- **notebooks/README.md**: Guida EDA e notebooks
- **sql/README.md**: Documentazione template SQL
- **tests/**: Suite test con esempi di uso

### Paper e Metodologie

- **Box-Cox Transformation**: Box, G. E. P., & Cox, D. R. (1964). "An analysis of transformations"
- **Duan Smearing**: Duan, N. (1983). "Smearing Estimate: A Nonparametric Retransformation Method"
- **PSI (Population Stability Index)**: Siddiqi, N. (2006). "Credit Risk Scorecards"
- **SHAP Values**: Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"

### Tool Documentation

- **Optuna**: https://optuna.org/
- **SHAP**: https://shap.readthedocs.io/
- **Weights & Biases**: https://docs.wandb.ai/
- **Scikit-learn**: https://scikit-learn.org/
- **XGBoost**: https://xgboost.readthedocs.io/
- **LightGBM**: https://lightgbm.readthedocs.io/
- **CatBoost**: https://catboost.ai/

---

**Fine della Review Completa**  
**Autore**: AI Assistant  
**Data**: 2025-11-11

Per domande o chiarimenti, fare riferimento ai file di configurazione e documentazione interna.
