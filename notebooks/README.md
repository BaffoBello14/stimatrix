# 📊 Notebooks EDA - StiMatrix

Questa directory contiene i notebook Jupyter per l'Exploratory Data Analysis (EDA) del progetto StiMatrix.

## 📁 Struttura

```
notebooks/
├── eda_utils.py                   # Modulo utilities condiviso
├── eda_basic.ipynb                # ✨ Analisi esplorativa di base
├── eda_advanced.ipynb             # 🚀 Analisi esplorativa avanzata
├── eda_analysis.py                # Script Python standalone (legacy)
├── eda_raw.ipynb                  # 📦 Notebook originale (da deprecare)
├── eda_comprehensive.ipynb        # 📦 Notebook originale avanzato (da deprecare)
├── eda_outputs/                   # Output analisi base
└── eda_comprehensive_outputs/     # Output analisi avanzata
```

## 🎯 Quali Notebook Usare?

### eda_basic.ipynb ✨ (RACCOMANDATO)
**Usa questo per**: Analisi esplorativa veloce e completa

**Contenuto**:
- Overview dataset (dimensioni, tipi, memoria)
- Analisi missingness
- Distribuzione target principale
- Top correlazioni con target
- Visualizzazioni base
- Summary per gruppi chiave
- Check geospaziale

**Tempo esecuzione**: ~2-5 minuti

**Output**: `eda_outputs/`

---

### eda_advanced.ipynb 🚀 (PER ANALISI APPROFONDITE)
**Usa questo per**: Analisi multi-target e correlazioni sofisticate

**Contenuto**:
- Analisi comparativa multi-target
- Correlazioni multiple (Pearson, Spearman, Kendall)
- Associazioni categoriche (Cramér's V)
- Correlation Ratio per relazioni miste
- Matrici di correlazione complete
- Feature importance comparativa
- Visualizzazioni avanzate

**Tempo esecuzione**: ~5-15 minuti

**Output**: `eda_comprehensive_outputs/`

---

### eda_analysis.py (LEGACY)
Script Python standalone - mantiene le stesse funzionalità di `eda_basic.ipynb` ma eseguibile da command line.

**Esecuzione**:
```bash
cd notebooks/
python eda_analysis.py
```

---

## 🚀 Quick Start

### 1. Installazione Dipendenze

```bash
# Assicurati di essere nell'environment corretto
pip install -r ../requirements.txt
```

### 2. Esecuzione Notebook Base

```bash
cd notebooks/
jupyter notebook eda_basic.ipynb
```

Oppure con JupyterLab:
```bash
jupyter lab eda_basic.ipynb
```

### 3. Esecuzione da Command Line

```bash
# Esegui tutto il notebook e genera HTML
jupyter nbconvert --to html --execute eda_basic.ipynb

# Esegui e aggiorna il notebook con gli output
jupyter nbconvert --to notebook --execute --inplace eda_basic.ipynb
```

---

## 📦 Modulo `eda_utils.py`

I nuovi notebook (`eda_basic.ipynb` e `eda_advanced.ipynb`) utilizzano il modulo `eda_utils.py` per:
- ✅ Evitare duplicazione di codice
- ✅ Migliorare manutenibilità
- ✅ Centralizzare best practices
- ✅ Facilitare testing

**Funzioni principali**:
- `load_config_and_data()`: Carica config e dataset con error handling
- `analyze_missingness()`: Analisi completa valori mancanti
- `analyze_target_distribution()`: Statistiche descrittive target
- `analyze_correlations()`: Calcolo correlazioni con target
- `plot_target_distribution()`: Visualizzazione distribuzione
- `create_correlation_heatmap()`: Heatmap correlazioni
- `save_plot()`: Salvataggio ottimizzato dei grafici
- `cramers_v()`: Associazione variabili categoriche
- `correlation_ratio()`: Relazione categorica-numerica

**Test del modulo**:
```bash
python eda_utils.py
```

---

## 🎨 Best Practices

### ✅ Prima del Commit
**IMPORTANTE**: Pulire sempre gli output dai notebook prima di committare per ridurre le dimensioni del repository.

```bash
# Pulisci output da tutti i notebook
jupyter nbconvert --clear-output --inplace *.ipynb

# Oppure solo da uno specifico
jupyter nbconvert --clear-output --inplace eda_basic.ipynb
```

### ✅ Configurazione Git Hook (Opzionale)

Puoi creare un pre-commit hook per pulire automaticamente i notebook:

```bash
# Crea il file .git/hooks/pre-commit
cat > ../.git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pulisci automaticamente output dai notebook
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
git add notebooks/*.ipynb
EOF

# Rendi eseguibile
chmod +x ../.git/hooks/pre-commit
```

### ✅ Grafici Ottimizzati

I nuovi notebook salvano i grafici con:
- **DPI = 100** (invece di 300) per dimensioni ragionevoli
- **Optimize = True** per compressione PNG
- **Nessun `plt.show()`** per evitare output nelle celle

**Dimensioni target per immagini**:
- Grafici semplici: < 300 KB
- Heatmap: < 1 MB
- Grafici complessi: < 500 KB

---

## 📊 Output Generati

### eda_outputs/ (da eda_basic.ipynb)
```
eda_outputs/
├── missingness_analysis.csv           # Analisi valori mancanti
├── target_statistics.csv              # Statistiche target
├── correlations_with_target.csv       # Correlazioni complete
├── group_summary_AI_ZonaOmi.csv      # Summary per zona
├── group_summary_AI_IdCategoriaCatastale.csv  # Summary per categoria
├── geospatial_columns_check.csv       # Check colonne geospaziali
├── target_distribution.png            # Distribuzione target
└── correlation_heatmap_top20.png      # Heatmap top 20 correlazioni
```

### eda_comprehensive_outputs/ (da eda_advanced.ipynb)
```
eda_comprehensive_outputs/
├── multi_target_comparison.csv                    # Confronto target
├── advanced_correlations_AI_Prezzo_Ridistribuito.csv  # Correlazioni multiple
├── advanced_correlations_AI_Prezzo_MQ.csv         # Correlazioni multiple
├── correlation_matrix_pearson.csv                 # Matrice Pearson
├── correlation_matrix_spearman.csv                # Matrice Spearman
├── correlation_matrix_mixed.csv                   # Matrice mista
├── target_distributions_comparison.png            # Confronto distribuzioni
├── targets_scatter_plot.png                       # Scatter tra target
├── correlation_heatmap_complete.png               # Heatmap completa (ottimizzata)
├── correlation_methods_comparison.png             # Confronto metodi
└── feature_importance_comparison.png              # Feature importance
```

---

## 🐛 Troubleshooting

### Problema: ModuleNotFoundError per eda_utils
```python
# Soluzione: Assicurati di essere nella directory notebooks/
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from eda_utils import *
```

### Problema: FileNotFoundError per config/data
```python
# Soluzione: Verifica i path relativi
config, df = load_config_and_data(
    config_path='../config/config.yaml',  # Un livello sopra
    data_path='../data/raw/raw.parquet'
)
```

### Problema: Notebook troppo lento
- Usa `eda_basic.ipynb` invece di `eda_advanced.ipynb`
- Riduci il numero di feature nelle analisi avanzate
- Filtra il dataset se troppo grande

### Problema: Immagini troppo grandi
```python
# Usa DPI ridotto quando salvi
save_plot('myplot', output_dir, dpi=100)  # invece di 300

# Oppure usa JPEG per grafici complessi
plt.savefig('plot.jpg', dpi=100, quality=85)
```

---

## 📚 Riferimenti

- **Pandas**: https://pandas.pydata.org/
- **Matplotlib**: https://matplotlib.org/
- **Seaborn**: https://seaborn.pydata.org/
- **SciPy Stats**: https://docs.scipy.org/doc/scipy/reference/stats.html
- **Jupyter**: https://jupyter.org/

---

## 🔄 Migration Guide

### Da eda_raw.ipynb → eda_basic.ipynb
I vecchi notebook `eda_raw.ipynb` e `eda_comprehensive.ipynb` sono deprecati ma mantenuti per compatibilità.

**Cosa è cambiato**:
1. ✅ Codice refactorizzato in `eda_utils.py`
2. ✅ Output puliti dai notebook
3. ✅ Grafici ottimizzati (DPI ridotto)
4. ✅ Logging migliorato
5. ✅ Error handling robusto
6. ✅ Nessun `plt.show()` nelle celle

**Migrazione consigliata**:
- Usa `eda_basic.ipynb` per analisi quotidiane
- Usa `eda_advanced.ipynb` per analisi approfondite
- Considera di deprecare i vecchi notebook dopo verifica

---

## 📝 Note

- **Versione Python**: 3.8+
- **Memoria raccomandata**: 8 GB+ per dataset completi
- **Tempo esecuzione**: 
  - Basic: 2-5 minuti
  - Advanced: 5-15 minuti

**Ultimo aggiornamento**: 2025-10-06

---

## 🤝 Contribuire

Per migliorare i notebook:
1. Segui le best practices sopra elencate
2. Testa le modifiche prima del commit
3. Pulisci gli output con `nbconvert`
4. Documenta nuove analisi nel README

**Domande?** Contatta il team Data Science.
