# 📊 Notebooks - Stimatrix Analysis Suite

**Suite completa di notebook per analisi esplorativa e validazione pipeline**

---

## 📚 Indice Notebook

1. **[eda_project_analysis.ipynb](#1-eda_project_analysisinb-)** - EDA allineato al progetto
2. **[target_transformations_comparison.ipynb](#2-target_transformations_comparisoninb-)** - Confronto trasformazioni target
3. **[outlier_detection_analysis.ipynb](#3-outlier_detection_analysisinb-)** - Analisi metodi outlier detection
4. **[encoding_strategies_comparison.ipynb](#4-encoding_strategies_comparisoninb-)** - Validazione strategie encoding
5. **[model_results_deep_analysis.ipynb](#5-model_results_deep_analysisinb-)** - Analisi approfondita risultati modello

---

## 1. `eda_project_analysis.ipynb` ⭐

**Analisi esplorativa completa del progetto Stimatrix**

#### Contenuto

1. **Dataset Raw Overview**
   - Dimensioni, tipi di dato, missing values
   - Statistiche descrittive complete

2. **Target Analysis**
   - Distribuzione AI_Prezzo_Ridistribuito
   - Statistiche (mean, median, skewness, kurtosis)
   - Visualizzazioni (histogram, boxplot, Q-Q plot)

3. **Analisi Temporale**
   - Distribuzione transazioni per anno
   - Trend prezzi nel tempo
   - Identificazione pattern stagionali

4. **Analisi Zone OMI**
   - Distribuzione transazioni per zona
   - Statistiche prezzo per zona
   - Highlight zone da escludere (config)

5. **Analisi Tipologie**
   - Distribuzione per tipologia edilizia
   - Identificazione tipologie da escludere

6. **⭐ EFFETTO FILTRI - Confronto Pre/Post**
   - Applicazione filtri da config (anno>=2022, zone, tipologie)
   - Confronto statistiche raw vs filtered
   - Visualizzazioni comparative
   - Analisi impatto su distribuzioni

7. **Correlazioni**
   - Top 30 feature correlate con target
   - Visualizzazione barplot
   - Export CSV completo

8. **Feature Pruning**
   - Verifica colonne droppate dalla config
   - Categorie di feature rimosse

9. **Summary Report**
   - Report JSON completo
   - Metriche chiave
   - Top correlazioni

#### Output

Tutti i file vengono salvati in `eda_project_outputs/`:

```
eda_project_outputs/
├── 00_summary_report.json
├── 01_missing_values_raw.csv
├── 02_target_statistics_raw.csv
├── 03_target_distribution_raw.png
├── 04_temporal_distribution.csv
├── 05_temporal_analysis.png
├── 06_zone_statistics_raw.csv
├── 07_zone_analysis.png
├── 08_target_comparison_raw_vs_filtered.csv
├── 09_distribution_comparison_raw_vs_filtered.png
├── 10_zone_comparison_raw_vs_filtered.csv
├── 11_correlations_with_target.csv
└── 12_top_correlations.png
```

---

## 🚀 Quick Start

### Esecuzione

```bash
# In Jupyter
cd notebooks
jupyter notebook eda_project_analysis.ipynb

# O con JupyterLab
jupyter lab eda_project_analysis.ipynb
```

### Esecuzione Completa

```bash
# Esegui tutte le celle e genera HTML
jupyter nbconvert --to html --execute eda_project_analysis.ipynb

# Esegui e aggiorna notebook con output
jupyter nbconvert --to notebook --execute --inplace eda_project_analysis.ipynb
```

---

## 🎯 Caratteristiche Chiave

### ✅ Allineato al Progetto

- Usa **stesse funzioni** della pipeline (`apply_data_filters`)
- Legge **stesso config** del training (`config.yaml`)
- Analizza **esattamente i dati** che vede il modello
- Mostra **effetto reale dei filtri** con confronti

### ✅ Interattivo

- Celle separate per ogni analisi
- Output chiari e leggibili
- Grafici informativi
- Export automatico CSV/PNG

### ✅ Production-Ready

- Gestione errori robusta
- Helper functions riutilizzabili
- Output directory organizzata
- Report JSON strutturato

---

## 📖 Esempio Output

### Confronto Raw vs Filtered

```
CONFRONTO TARGET: RAW vs FILTERED
================================================================================

             Raw    Filtered    Delta  Delta_Pct
Count      5733        3421    -2312      -40.3%
Mean      62592       58234    -4358       -7.0%
Median    42000       41500     -500       -1.2%
Std       79533       71245    -8288      -10.4%
Skewness   5.16        4.82    -0.34       -6.6%
Kurtosis  54.18       48.21    -5.97      -11.0%
```

### Zone Comparison

```
CONFRONTO ZONE: RAW vs FILTERED
================================================================================

      Raw  Filtered  Removed  Removed_Pct
B1   1797      1797        0         0.0%
C4   1105      1105        0         0.0%
D2    718       718        0         0.0%
E3     59         0       59       100.0%  ← ESCLUSA
E2     49         0       49       100.0%  ← ESCLUSA
E1     37         0       37       100.0%  ← ESCLUSA
R1      8         0        8       100.0%  ← ESCLUSA
```

---

## 🔄 Workflow Tipico

1. **Prima del Training**
   ```bash
   # Esegui EDA per capire dati
   jupyter notebook eda_project_analysis.ipynb
   ```

2. **Dopo Modifica Config**
   ```bash
   # Ri-esegui per vedere effetto nuovi filtri
   jupyter nbconvert --to notebook --execute --inplace eda_project_analysis.ipynb
   ```

3. **Dopo Training**
   ```bash
   # Confronta EDA con metriche modello
   # Itera su filtri se necessario
   ```

---

## 📝 Note

### Dipendenze

```bash
pip install pandas numpy matplotlib seaborn scipy jupyter
```

### Config Required

Il notebook richiede:
- `config/config.yaml` con sezione `data_filters`
- `data/raw/raw.parquet` con il dataset

### Memory Usage

- Dataset ~5,000 righe: ~50 MB RAM
- Con tutti i plot: ~200 MB RAM

---

## 🎓 Spiegazione Tecnica

### Perché un Notebook Unico?

1. **Semplicità**: Un solo file da eseguire
2. **Coerenza**: Stesso flow per tutte le analisi
3. **Confronti**: Facile confrontare raw vs filtered
4. **Manutenibilità**: Un solo file da aggiornare

### Perché Celle Separate?

- Debugging più facile (esegui solo celle specifiche)
- Output progressivo (vedi risultati step-by-step)
- Flessibilità (salta celle non necessarie)

---

## 2. `target_transformations_comparison.ipynb` 🔄

**Confronto completo di tutte le trasformazioni possibili del target**

### Obiettivo

Valutare quale trasformazione matematica del target produce la distribuzione più normale e più adatta per il machine learning.

### Trasformazioni Analizzate

1. **None** (distribuzione originale)
2. **Log** (log naturale)
3. **Log10** (logaritmo base 10)
4. **Sqrt** (radice quadrata)
5. **Box-Cox** (parametrica, solo valori positivi)
6. **Yeo-Johnson** (parametrica, anche valori negativi) ← **ATTUALMENTE USATO**
7. **Quantile Uniform** (distribuzione uniforme)
8. **Quantile Normal** (distribuzione normale forzata)
9. **PowerTransformer** (sklearn implementation)

### Metriche di Valutazione

- **Skewness** (asimmetria, target: ~0)
- **Kurtosis** ("tailedness", target: ~0)
- **Shapiro-Wilk test** (normalità, p-value più alto = meglio)
- **Anderson-Darling test** (normalità robusta)
- **Jarque-Bera test** (normalità basata su skew+kurt)
- **Q-Q plots** (visual normality check)

### Output

```
transformations_outputs/
├── 00_summary_report.json
├── 01_transformations_comparison.csv
├── 02_distributions_grid.png
├── 03_qq_plots_grid.png
└── 04_metrics_comparison.png
```

### Quando Usarlo

- Prima di configurare `target_transform` in config
- Se metriche modello non soddisfacenti
- Per capire se trasformazione attuale è ottimale
- Dopo cambio filtri dataset (distribuzione cambiata)

---

## 3. `outlier_detection_analysis.ipynb` 🔍

**Confronto metodi di outlier detection e validazione ensemble attuale**

### Obiettivo

Analizzare l'efficacia di diversi metodi di outlier detection e validare la configurazione ensemble attualmente usata nella pipeline.

### Metodi Analizzati

1. **IQR (1.5)** - Standard interquartile range
2. **IQR (1.0)** ← **CONFIG PROJECT** (più aggressivo)
3. **Z-Score (3.0)** - Standard deviation based
4. **Z-Score (2.5)** ← **CONFIG PROJECT**
5. **Modified Z-Score** - Median-based (robusto)
6. **Isolation Forest** ← **CONFIG PROJECT**
7. **LOF** (Local Outlier Factor)
8. **Elliptic Envelope** (covariance-based)
9. **ENSEMBLE** (IQR + Z-Score + Isolation) ← **CONFIG PROJECT**

### Analisi Fornite

- Numero outlier rilevati per metodo
- Overlap tra metodi (Venn diagram)
- Scatter plots con outlier evidenziati
- Impatto su statistiche (mean, median, std, skew, kurt)
- Box plots e histograms before/after
- Comparison table completa

### Output

```
outliers_outputs/
├── 00_summary_report.json
├── 01_methods_comparison.csv
├── 02_methods_comparison_bar.png
├── 03_scatter_plots.png
├── 04_boxplots_comparison.png
├── 05_histograms_comparison.png
└── 06_venn_diagram_ensemble.png
```

### Quando Usarlo

- Prima di configurare `outlier_detection` in config
- Se sospetti troppi/troppo pochi outlier rimossi
- Per bilanciare aggressività detection
- Dopo cambio filtri (nuova distribuzione)

---

## 4. `encoding_strategies_comparison.ipynb` 🏷️

**Validazione strategie di encoding categorico e analisi dimensionalità**

### Obiettivo

Verificare che le soglie di cardinalità per le strategie di encoding siano ottimali e analizzare l'impatto sulla dimensionalità.

### Strategie Analizzate

1. **OneHot Encoding** (cardinality ≤ 10) - Interpretabile ma espande dims
2. **Target Encoding** (cardinality 11-50) - Compatto, richiede leak protection
3. **Frequency Encoding** (cardinality > 50) - Scalabile, perde info
4. **Ordinal Encoding** (fallback/custom)

### Analisi Fornite

- **Cardinalità per feature**: Quante categorie uniche
- **Dimensionalità impact**: Quante colonne risultanti
- **Strategia assegnata**: Quale encoding per quale feature
- **Correlation con target**: Predictive power (eta-squared)
- **Unseen categories**: Categorie in test non viste in train
- **Leak-free validation**: Test su split temporale

### Output

```
encoding_outputs/
├── 00_summary_report.json
├── 01_cardinality_analysis.csv
├── 02_cardinality_bar_chart.png
├── 03_strategies_pie_chart.png
├── 04_dimensionality_impact.png
├── 05_correlation_with_target.csv
├── 06_correlation_with_target.png
├── 07_unseen_categories_analysis.csv
└── 08_unseen_categories.png
```

### Quando Usarlo

- Prima di configurare `encoding` in config
- Se dimensionalità troppo alta (OOM errors)
- Se unseen categories causano problemi
- Per ottimizzare soglie cardinality (10, 50)

---

## 5. `model_results_deep_analysis.ipynb` 📊

**Analisi approfondita dei risultati del modello trainato**

### Prerequisiti

⚠️ **IMPORTANTE**: Questo notebook richiede che il training sia stato eseguito!

```bash
python main.py --config config/config.yaml --steps train
```

### Obiettivo

Analizzare in profondità le performance del modello, identificare pattern negli errori, validare residui, e comprendere feature importance.

### Analisi Fornite

1. **Performance Metrics**
   - MAE, RMSE, MAPE, R² per train/val/test
   - Bar charts comparativi

2. **Residual Analysis**
   - Distribuzione residui (normalità?)
   - Q-Q plots
   - Heteroskedasticity check (residui vs predictions)

3. **Prediction vs Actual**
   - Scatter plots con linea perfetta
   - R² visualizzato

4. **Error by Price Range**
   - MAE e MAPE per fascia di prezzo
   - Identifica dove modello fallisce

5. **Worst Predictions**
   - Top 20 predizioni peggiori
   - Analisi errori assoluti e percentuali

6. **Feature Importance** (se disponibile)
   - Tree-based: feature_importances_
   - Linear: coefficients
   - Top 20 features

### Output

```
model_analysis_outputs/
├── 00_summary_report.json
├── 01_performance_metrics.csv
├── 02_performance_metrics.png
├── 03_residual_analysis.png
├── 04_heteroskedasticity.png
├── 05_prediction_vs_actual.png
├── 06_error_by_price_range.csv
├── 07_error_by_price_range.png
├── 08_worst_predictions.csv
├── 09_feature_importance.csv
└── 10_feature_importance.png
```

### Quando Usarlo

- **DOPO ogni training** per validare risultati
- Se R² troppo basso
- Per identificare overfitting/underfitting
- Per feature selection (rimuovi low importance)
- Per debugging predizioni anomale

---

## 🚀 Quick Start - Workflow Completo

### 1. Prima del Training: EDA

```bash
cd notebooks
jupyter notebook eda_project_analysis.ipynb
```

**Obiettivo**: Capire dataset, verificare filtri, identificare problemi

### 2. Tuning Configurazione: Comparisons

```bash
# Confronta trasformazioni target
jupyter notebook target_transformations_comparison.ipynb

# Valida outlier detection
jupyter notebook outlier_detection_analysis.ipynb

# Ottimizza encoding strategies
jupyter notebook encoding_strategies_comparison.ipynb
```

**Obiettivo**: Ottimizzare config.yaml prima del training

### 3. Training

```bash
cd ..
python main.py --config config/config.yaml --steps preprocess train
```

### 4. Dopo Training: Analisi Risultati

```bash
cd notebooks
jupyter notebook model_results_deep_analysis.ipynb
```

**Obiettivo**: Validare modello, identificare miglioramenti, iterare

---

## 📊 Esecuzione Batch (tutti i notebook)

### Execute All (genera output)

```bash
# EDA
jupyter nbconvert --to notebook --execute --inplace eda_project_analysis.ipynb

# Comparisons (PRE-training)
jupyter nbconvert --to notebook --execute --inplace target_transformations_comparison.ipynb
jupyter nbconvert --to notebook --execute --inplace outlier_detection_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace encoding_strategies_comparison.ipynb

# Results analysis (POST-training, richiede model!)
jupyter nbconvert --to notebook --execute --inplace model_results_deep_analysis.ipynb
```

### Export HTML (per documentazione)

```bash
jupyter nbconvert --to html --execute eda_project_analysis.ipynb
jupyter nbconvert --to html --execute target_transformations_comparison.ipynb
jupyter nbconvert --to html --execute outlier_detection_analysis.ipynb
jupyter nbconvert --to html --execute encoding_strategies_comparison.ipynb
jupyter nbconvert --to html --execute model_results_deep_analysis.ipynb
```

---

## 🎯 Decision Tree - Quale Notebook Usare?

```
Vuoi capire i dati raw?
├─ YES → eda_project_analysis.ipynb
└─ NO
   │
   Vuoi ottimizzare config?
   ├─ Target transformation? → target_transformations_comparison.ipynb
   ├─ Outlier detection? → outlier_detection_analysis.ipynb
   ├─ Encoding strategies? → encoding_strategies_comparison.ipynb
   └─ NO
      │
      Hai già trainato il modello?
      ├─ YES → model_results_deep_analysis.ipynb
      └─ NO → Esegui prima il training!
```

---

## 📋 Checklist Pre-Esecuzione

### Tutti i Notebook (eccetto model_results)

- [ ] Config aggiornato (`config/config.yaml`)
- [ ] Dataset presente (`data/raw/raw.parquet`)
- [ ] Jupyter installato (`pip install jupyter`)
- [ ] Dipendenze installate (`pip install -r requirements.txt`)

### Solo model_results_deep_analysis.ipynb

- [ ] Training completato (`python main.py --steps train`)
- [ ] Model salvato (`models/best_model.pkl` o simili)
- [ ] Preprocessed data disponibile (`data/preprocessed/X_*.parquet`)

---

## 🎓 Dipendenze Extra (opzionali)

```bash
# Per Venn diagrams (outlier_detection_analysis.ipynb)
pip install matplotlib-venn

# Per SHAP (se usi model_results con SHAP)
pip install shap

# Per notebook interattivi avanzati
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

---

## ✅ Checklist Esecuzione

Prima di eseguire il notebook:

- [ ] Config aggiornato (`config/config.yaml`)
- [ ] Dataset presente (`data/raw/raw.parquet`)
- [ ] Jupyter installato (`pip install jupyter`)
- [ ] Spazio su disco per output (~10-20 MB)

Dopo esecuzione:

- [ ] Verifica output in `eda_project_outputs/`
- [ ] Controlla confronto raw vs filtered
- [ ] Valida che filtri siano applicati correttamente
- [ ] Documenta osservazioni per iterazioni future

---

## 📞 Supporto

Per domande o problemi:
1. Controlla log delle celle (errori Python)
2. Verifica path dei file (config, data)
3. Controlla dipendenze installate
4. Leggi documentazione inline nelle celle

---

## 🎨 Output Directory Structure

Dopo esecuzione completa:

```
notebooks/
├── eda_project_analysis.ipynb
├── target_transformations_comparison.ipynb
├── outlier_detection_analysis.ipynb
├── encoding_strategies_comparison.ipynb
├── model_results_deep_analysis.ipynb
├── README.md
│
├── eda_project_outputs/           # EDA
│   ├── 00_summary_report.json
│   ├── 01_missing_values_raw.csv
│   └── ...
│
├── transformations_outputs/        # Target transformations
│   ├── 00_summary_report.json
│   ├── 01_transformations_comparison.csv
│   └── ...
│
├── outliers_outputs/               # Outlier detection
│   ├── 00_summary_report.json
│   ├── 01_methods_comparison.csv
│   └── ...
│
├── encoding_outputs/               # Encoding strategies
│   ├── 00_summary_report.json
│   ├── 01_cardinality_analysis.csv
│   └── ...
│
└── model_analysis_outputs/         # Model results
    ├── 00_summary_report.json
    ├── 01_performance_metrics.csv
    └── ...
```

---

## 📞 Supporto e Troubleshooting

### Errori Comuni

**1. FileNotFoundError: config.yaml**
```bash
# Assicurati di essere nella directory corretta
cd /workspace/notebooks
# O usa path assoluto nel notebook
```

**2. FileNotFoundError: raw.parquet**
```bash
# Verifica che dataset esista
ls ../data/raw/raw.parquet
```

**3. ModuleNotFoundError**
```bash
# Installa dipendenze
pip install -r ../requirements.txt
```

**4. Model non trovato (model_results)**
```bash
# Esegui prima il training
cd ..
python main.py --config config/config.yaml --steps preprocess train
```

### Memory Issues

Se OOM (Out of Memory):
- Riduci `n_bins` nelle analisi
- Sample il dataset (es. `df.sample(frac=0.5)`)
- Chiudi altri notebook/applicazioni

---

## 🔄 Versioning

| Version | Date       | Changes                                      |
|---------|------------|----------------------------------------------|
| 1.0     | 2025-11-14 | Initial EDA notebook                         |
| 2.0     | 2025-11-14 | Added 4 comparison/analysis notebooks        |

---

**Ultimo aggiornamento**: 2025-11-14  
**Versione suite**: 2.0  
**Compatibile con**: Python 3.10+, Stimatrix pipeline v1.0
