# 📊 Notebooks - Stimatrix EDA

**Notebook unico allineato alle scelte del progetto**

---

## 📓 Notebook Disponibile

### `eda_project_analysis.ipynb` ⭐

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

## 🚧 Possibili Estensioni Future

### Notebook Aggiuntivi

Se necessario, si possono aggiungere:

1. **`eda_preprocessed_features.ipynb`**
   - Analisi feature dopo preprocessing
   - Target transformation (Yeo-Johnson)
   - Feature contestuali aggiunte
   - Encoding analysis

2. **`eda_model_results.ipynb`**
   - Analisi predizioni modelli
   - Error analysis per gruppo
   - SHAP importance
   - Residual plots

### Estensioni Notebook Corrente

- Analisi correlazioni tra feature (matrice completa)
- Feature importance preliminare (RandomForest)
- Outlier detection visualization
- PCA analysis per esplorare varianza

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

**Ultimo aggiornamento**: 2025-11-14  
**Versione notebook**: 1.0  
**Compatibile con**: Python 3.10+, Stimatrix pipeline v1.0
