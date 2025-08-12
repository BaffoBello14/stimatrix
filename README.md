# Stimatrix ML Pipeline

Un pipeline modulare di machine learning per la stima automatica dei prezzi immobiliari, con funzionalità avanzate di preprocessing, feature engineering e training di modelli.

## 🎯 Obiettivo

Questo progetto implementa un sistema completo per l'analisi e la predizione di prezzi immobiliari attraverso:
- **Feature extraction** automatica da geometrie WKT e dati GeoJSON
- **Preprocessing** modulare con imputazione, encoding e scaling
- **Training** di modelli ML (alberi decisionali, linear models, boosting, CatBoost)
- **Ottimizzazione** degli iperparametri con Optuna
- **Valutazione** con metriche avanzate e SHAP analysis

## 🚀 Quick Start

### Prerequisiti
- **Python 3.13+**
- Virtual environment (raccomandato)

### Installazione
```bash
# Clona il repository
git clone <repository-url>
cd stimatrix-ml-pipeline

# Crea virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# oppure
.venv\Scripts\activate     # Windows

# Installa dipendenze
pip install -r requirements.txt
```

### Esecuzione
```bash
# Esegui l'intera pipeline
python main.py --config config/config.yaml --steps all

# Oppure step specifici
python main.py --config config/config.yaml --steps preprocessing training
```

## 📂 Struttura del Progetto

```
stimatrix-ml-pipeline/
├── main.py                 # 🎮 Orchestratore CLI principale
├── config/
│   └── config.yaml         # ⚙️ Configurazione pipeline
├── src/                    # 📦 Codice sorgente
│   ├── dataset_builder/    # 🗄️ Retrieval dati dal database
│   ├── db/                 # 🔌 Utilità database e schema extraction
│   ├── preprocessing/      # 🔧 Pipeline preprocessing e feature engineering
│   ├── training/           # 🤖 Training, tuning e valutazione modelli
│   └── utils/              # 🛠️ Logging, I/O e utilità generali
├── tests/                  # 🧪 Test suite completa
├── data/                   # 📊 Dati (ignorati da git)
│   ├── raw/                # Dati grezzi
│   └── preprocessed/       # Dati processati
└── models/                 # 🎯 Modelli salvati (ignorati da git)
```

## 🧪 Testing

### Test Rapidi (Raccomandato)
```bash
# Linux/macOS
./run_tests.sh basic

# Windows
run_tests.bat basic
```

I test di base verificano:
- ✅ Import di tutti i moduli
- ✅ Feature extraction da geometrie WKT
- ✅ Validazione e preprocessing dati
- ✅ Training dei modelli
- ✅ Sistema di logging

### Test Completi
```bash
# Linux/macOS
./run_tests.sh all

# Windows
run_tests.bat all
```

### Modalità Test Disponibili

| Comando | Descrizione |
|---------|-------------|
| `basic` | Test essenziali senza dipendenze pesanti |
| `all` | Suite completa con pytest |
| `features` | Test feature extractors |
| `preprocessing` | Test pipeline preprocessing |
| `training` | Test training modelli |
| `coverage` | Test con report di copertura |
| `verbose` | Output dettagliato |

## ⚙️ Configurazione

Il file `config/config.yaml` controlla tutti gli aspetti della pipeline:

### Sezioni Principali

- **`paths`**: Percorsi input/output per dati e modelli
- **`target`**: Configurazione variabile target e trasformazioni
- **`feature_extraction`**: Estrazione da geometrie WKT e JSON
- **`temporal_split`**: Split temporale per evitare data leakage
- **`outliers`**: Rimozione outlier per gruppo o globale
- **`imputation`**: Strategie di imputazione per numeriche/categoriche
- **`encoding`**: Piani di encoding (OHE/ordinal) con gestione cardinalità
- **`profiles`**: Tre profili predefiniti (`scaled`, `tree`, `catboost`)
- **`training`**: Configurazione modelli, tuning e valutazione

### Profili di Preprocessing

1. **`scaled`**: Per modelli lineari e reti neurali
   - Encoding categorico completo
   - Scaling e PCA opzionale
   - Winsorization outlier

2. **`tree`**: Per modelli ad albero (Random Forest, XGBoost)
   - Encoding semplificato
   - Preservazione informazione ordinale
   - Pruning correlazioni

3. **`catboost`**: Per CatBoost nativo
   - Mantenimento variabili categoriche
   - Preprocessing minimo
   - Gestione automatica delle categorie

## 🔧 Pipeline di Preprocessing

### Flusso Principale

1. **Caricamento Dati**: Import da `data/raw/*.parquet`
2. **Feature Extraction**:
   - **WKT Geometries**: `POINT` → coordinate x,y; `POLYGON` → conteggio vertici; `MULTIPOLYGON` → statistiche avanzate
   - **GeoJSON**: Estrazione `areaMq`, `perimetroM`, codici catastali, bounding box
3. **Normalizzazioni**:
   - **Superfici**: Unificazione in `AI_Superficie` (m²)
   - **Piani**: Feature engineering robusto con parsing di token complessi
   - **Indirizzi**: Estrazione parte numerica da `AI_Civico`
4. **Split Temporale**: Divisione train/validation/test basata su timestamp
5. **Outlier Detection**: Rimozione outlier con metodi configurabili
6. **Imputazione**: Gestione valori mancanti per tipo di variabile
7. **Encoding**: Trasformazione variabili categoriche
8. **Scaling/PCA**: Normalizzazione e riduzione dimensionalità (profilo `scaled`)

### Feature Engineering Avanzato

- **Geometrie WKT**: Parsing nativo senza dipendenze GIS
- **Piano Building**: Riconoscimento pattern complessi (P1-P12, S/S1/S2, PT/T/ST, RIAL, AMMEZZATO)
- **Coercizione Numerica**: Conversione automatica con blacklist configurabile
- **Temporal Keys**: Creazione chiavi temporali da anno/mese stipula

## 🤖 Training e Modelli

### Modelli Supportati

- **Linear Models**: Ridge, Lasso, Elastic Net
- **Tree-based**: Random Forest, Extra Trees
- **Boosting**: XGBoost, LightGBM, CatBoost
- **Ensemble**: Stacking e averaging

### Ottimizzazione

- **Optuna**: Hyperparameter tuning automatico
- **Cross-validation**: Validazione robusta dei risultati
- **Early stopping**: Prevenzione overfitting
- **Parallelizzazione**: Supporto multi-core configurabile

### Valutazione

- **Metriche**: MAE, RMSE, R², MAPE
- **SHAP Analysis**: Interpretabilità dei modelli
- **Validation**: Split temporale per prevenire data leakage

## 📊 Output

### Dati Preprocessati
```
data/preprocessed/
├── scaled/          # Dati per modelli lineari
├── tree/            # Dati per modelli ad albero  
├── catboost/        # Dati per CatBoost
└── preprocessed.parquet  # Dataset combinato
```

### Modelli Addestrati
```
models/
├── model_*.pkl      # Modelli serializzati
├── scaler_*.pkl     # Oggetti di scaling
└── shap/            # Report di interpretabilità
```

## 🔍 Troubleshooting

### Problemi Comuni

| Problema | Soluzione |
|----------|-----------|
| Errori di import | Verifica `PYTHONPATH` o esegui via `main.py` |
| Pacchetti mancanti | Attiva virtual environment e reinstalla `requirements.txt` |
| File raw assente | Posiziona dati in `data/raw/raw.parquet` |
| Test falliscono | Prova `./run_tests.sh basic` per verificare setup |
| Memory error | Riduci batch size o abilita processing incrementale |

### Logging e Debug

Il sistema di logging è configurabile tramite `config.yaml`:
- **Livelli**: DEBUG, INFO, WARNING, ERROR
- **Output**: Console e file rotating
- **Filtri**: Per modulo e categoria

## 🔒 Sicurezza e Privacy

- **Dati sensibili**: Non committare mai dati reali
- **Credenziali**: Usa file `.env` per configurazioni sensibili
- **Modelli**: I modelli addestrati sono esclusi dal version control

## 📝 Sviluppo

### Contribuire al Progetto

1. **Fork** del repository
2. **Branch**: Crea feature branch (`git checkout -b feature/amazing-feature`)
3. **Test**: Esegui test suite (`./run_tests.sh all`)
4. **Commit**: Commit con messaggi descrittivi
5. **Push**: Push del branch (`git push origin feature/amazing-feature`)
6. **PR**: Apri Pull Request

### Coding Standards

- **Python**: PEP 8, type hints, docstrings
- **Testing**: Coverage minima 80%
- **Documentation**: Aggiorna README per nuove feature

## 📜 Licenza

Proprietario. Uso interno.

---

**Sviluppato con ❤️ per l'analisi immobiliare intelligente**