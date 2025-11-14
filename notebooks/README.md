# 📓 Notebooks - Analisi e Sperimentazione

Questa cartella contiene i notebook Jupyter per l'analisi esplorativa dei dati (EDA), la sperimentazione di strategie di preprocessing e l'analisi dei risultati dei modelli.

## 📋 Indice dei Notebook

### 🔍 Analisi Esplorativa

#### 1. [`eda_project_analysis.ipynb`](eda_project_analysis.ipynb)
**Analisi esplorativa completa del dataset**

**Obiettivo**: Comprendere la struttura, distribuzione e qualità dei dati

**Analisi incluse**:
- 📊 Statistiche descrittive complete
- 📈 Distribuzione delle variabili numeriche e categoriche
- 🗺️ Analisi geografica (zone OMI, comuni)
- 🏠 Analisi tipologie immobiliari
- 💰 Distribuzione prezzi e target variable
- 🔗 Correlazioni tra features
- ❌ Missing values e data quality
- 📊 Outliers detection e analisi

**Prerequisiti**: 
```bash
# Nessun prerequisito - usa i dati raw
```

**Output**: Grafici e insights salvati in `eda_outputs/`

---

#### 2. [`outlier_detection_analysis.ipynb`](outlier_detection_analysis.ipynb)
**Analisi approfondita degli outliers**

**Obiettivo**: Identificare e analizzare outliers nei dati per decidere strategie di gestione

**Analisi incluse**:
- 🎯 Detection con metodi multipli (IQR, Z-score, Isolation Forest, LOF)
- 📊 Distribuzione outliers per feature
- 🗺️ Distribuzione geografica degli outliers
- 💡 Impact analysis: effetto sulle metriche del modello
- 🔧 Strategie di gestione (rimozione, capping, winsorization)

**Prerequisiti**: 
```bash
python main.py --config config/config.yaml --steps retrieve
```

**Output**: Report outliers salvati in `outlier_outputs/`

---

### 🧪 Sperimentazione Preprocessing

#### 3. [`encoding_strategies_comparison.ipynb`](encoding_strategies_comparison.ipynb)
**Confronto strategie di encoding per variabili categoriche**

**Obiettivo**: Confrontare diverse tecniche di encoding per scegliere la migliore

**Strategie testate**:
- 🔢 **One-Hot Encoding**: Creazione colonne binarie
- 📊 **Target Encoding**: Encoding basato su media target
- 🎯 **Frequency Encoding**: Encoding basato su frequenza
- 🔄 **Leave-One-Out Encoding**: Target encoding con LOO per evitare leakage
- 📈 **Weight of Evidence (WoE)**: Encoding per regressione logistica
- 🏆 **CatBoost Encoding**: Encoding ottimizzato per CatBoost

**Metriche confronto**:
- Performance modello (R², MAE, RMSE)
- Training time
- Dimensionalità risultante
- Robustezza a overfitting

**Prerequisiti**: 
```bash
python main.py --config config/config.yaml --steps retrieve
```

**Output**: Report comparativo salvato in `encoding_comparison_outputs/`

---

#### 4. [`target_transformations_comparison.ipynb`](target_transformations_comparison.ipynb)
**Confronto trasformazioni del target variable**

**Obiettivo**: Testare trasformazioni del target per migliorare performance e normalità residui

**Trasformazioni testate**:
- 🔄 **Log Transform**: `log(y)` - per distribuzioni right-skewed
- 📦 **Box-Cox**: Trasformazione parametrica ottimale
- 🎯 **Yeo-Johnson**: Come Box-Cox ma gestisce valori negativi
- √ **Square Root**: `sqrt(y)` - trasformazione moderata
- 📐 **Quantile Transform**: Mapping a distribuzione uniforme/normale

**Analisi**:
- Impatto su normalità residui
- Performance metriche (prima e dopo trasformazione)
- Stabilità train/test
- Interpretabilità risultati

**Prerequisiti**: 
```bash
python main.py --config config/config.yaml --steps preprocess
```

**Output**: Report trasformazioni salvato in `target_transform_outputs/`

---

### 📊 Analisi Risultati

#### 5. [`model_results_deep_analysis.ipynb`](model_results_deep_analysis.ipynb)
**Analisi approfondita dei risultati dei modelli trainati**

**Obiettivo**: Valutare performance, identificare problemi e suggerire miglioramenti

**Analisi incluse**:
- 📊 **Model Comparison**: Confronto tra tutti i modelli trainati
- 🏆 **Best Model Selection**: Identificazione modello ottimale
- 📉 **Overfitting Analysis**: Gap train-test e generalizzazione
- 🎯 **Group Performance**: Errori per categoria catastale, zona OMI, tipologia
- ❌ **Worst Predictions**: Analisi predizioni peggiori
- 📈 **Residual Analysis**: Distribuzione e pattern nei residui
- 📊 **Prediction Intervals**: Coverage e calibrazione intervalli di confidenza

**Metriche analizzate**:
- R² (coefficient of determination)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

**Prerequisiti**: 
```bash
python main.py --config config/config.yaml --steps train
```

**Output**: Report completo salvato in `model_analysis_outputs/`

**File generati**:
- `00_analysis_summary.json` - Summary completo
- `01_model_comparison.csv` - Confronto modelli
- `02-08_*.png` - Grafici analisi
- `07_prediction_intervals.csv` - Analisi intervalli

---

## 🛠️ Utility

### [`eda_utils.py`](eda_utils.py)
**Funzioni di supporto per EDA**

Contiene funzioni helper per:
- Plot standardizzati
- Statistiche comuni
- Formattazione output
- Color schemes

Importato automaticamente nei notebook EDA.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Attiva virtual environment
source venv/bin/activate  # Linux/Mac
# oppure
venv\Scripts\activate  # Windows

# Installa dipendenze
pip install -r requirements.txt
```

### 2. Run Pipeline (per avere dati per i notebook)
```bash
# Retrieve data
python main.py --config config/config.yaml --steps retrieve

# Preprocess
python main.py --config config/config.yaml --steps preprocess

# Train models
python main.py --config config/config.yaml --steps train
```

### 3. Open Notebooks
```bash
# Avvia Jupyter
jupyter notebook

# Oppure con JupyterLab
jupyter lab
```

### 4. Esegui i notebook nell'ordine suggerito:
1. **EDA** → `eda_project_analysis.ipynb`
2. **Outliers** → `outlier_detection_analysis.ipynb`
3. **Encoding** → `encoding_strategies_comparison.ipynb`
4. **Target Transform** → `target_transformations_comparison.ipynb`
5. **Results** → `model_results_deep_analysis.ipynb`

---

## 📊 Spiegazione Prediction Intervals

I file `*_prediction_intervals.json` contengono informazioni sugli **intervalli di confidenza** delle predizioni:

```json
{
  "80%": {
    "coverage": 0.78,              // % valori reali nell'intervallo
    "average_width": 125277.86,    // Larghezza media intervallo (€)
    "average_width_pct": 209701.52, // Larghezza % rispetto al prezzo
    "target_coverage": 0.8         // Coverage target (80%)
  }
}
```

### Interpretazione:

- **`coverage`**: Percentuale di osservazioni reali che cadono nell'intervallo
  - Idealmente dovrebbe essere ~80% per intervallo 80%
  - Se < target: intervallo troppo stretto (under-coverage)
  - Se > target: intervallo troppo largo (over-coverage)

- **`average_width`**: Larghezza media dell'intervallo in euro
  - Indica l'incertezza del modello
  - Intervalli larghi = alta incertezza

- **`average_width_pct`**: Larghezza in percentuale rispetto al prezzo
  - Normalizza la larghezza per confronti
  - >100% indica intervalli molto ampi

### Diagnostics:

| Coverage Gap | Status | Azione |
|-------------|--------|--------|
| \|gap\| < 0.02 | 🟢 Well calibrated | OK |
| gap < -0.05 | 🔴 Under-coverage | Allarga intervalli |
| gap > 0.05 | 🟠 Over-coverage | Restringi intervalli |
| -0.05 < gap < 0.05 | 🟡 Acceptable | Minor tuning |

---

## 📁 Output Directories

Ogni notebook crea una cartella di output:

```
notebooks/
├── eda_outputs/                    # EDA analysis
├── outlier_outputs/                # Outlier detection
├── encoding_comparison_outputs/    # Encoding strategies
├── target_transform_outputs/       # Target transformations
└── model_analysis_outputs/         # Model results analysis
```

---

## 🔧 Troubleshooting

### Problema: Notebook non trova i moduli
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
```
Questo è già incluso nei notebook, assicurati di eseguire dall'interno della cartella `notebooks/`.

### Problema: File non trovati
Assicurati di aver eseguito gli step del pipeline prima di aprire i notebook:
- `retrieve` → per EDA e outlier analysis
- `preprocess` → per encoding e target transform
- `train` → per model results analysis

### Problema: Kernel non trovato
```bash
# Crea kernel per il progetto
python -m ipykernel install --user --name=stimatrix --display-name="Stimatrix"
```

### Problema: Memoria insufficiente
Se i notebook crashano per memoria, considera:
1. Ridurre il dataset in `config.yaml`
2. Usare `chunksize` per lettura dati
3. Liberare memoria con `del variable` dopo uso

---

## 📚 Risorse

### Documentazione
- [Pandas](https://pandas.pydata.org/docs/)
- [Matplotlib](https://matplotlib.org/stable/contents.html)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)

### Best Practices
- ✅ Esegui le celle in ordine
- ✅ Riavvia kernel se modifichi moduli esterni
- ✅ Salva output importanti in file
- ✅ Commenta insights direttamente nel notebook
- ✅ Usa `%matplotlib inline` per plot inline

---

## 🤝 Contribuire

Per aggiungere nuovi notebook:

1. Segui la struttura esistente
2. Includi sezione "Obiettivo" e "Prerequisiti"
3. Salva output in cartella dedicata
4. Aggiungi documentazione in questo README
5. Testa il notebook da fresh kernel

---

## 📝 Note

- I notebook sono **self-contained**: includono tutto il codice necessario
- Gli output sono **salvati automaticamente** nelle rispettive cartelle
- I plot usano **style consistente** per uniformità
- Le metriche sono **calcolate su scala originale** del target per interpretabilità

---

**Last Updated**: 2025-11-14  
**Maintainer**: Stimatrix Team  
**Python Version**: 3.12+  
**Jupyter Version**: Latest
