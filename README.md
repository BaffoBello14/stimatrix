# Stimatrix ML Pipeline

Un pipeline modulare e robusto di machine learning per la stima automatica dei prezzi immobiliari, con funzionalità enterprise-grade di preprocessing, quality checks, tracking e evaluation avanzata.

## 🎯 Obiettivo

Questo progetto implementa un sistema enterprise-grade per l'analisi e la predizione di prezzi immobiliari attraverso:
- **Feature extraction** automatica da geometrie WKT e dati GeoJSON con fallback robusti
- **Preprocessing** modulare con tracking evoluzione dataset e quality checks automatici
- **Training** di modelli ML multipli con profili ottimizzati per algoritmo
- **Ottimizzazione** degli iperparametri con Optuna e sampler avanzati
- **Evaluation** dual-scale con feature importance multi-metodo e SHAP analysis
- **Quality assurance** con validazione data leakage e monitoring pipeline
- **Tracking completo** con report automatici e alerting performance

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
# Pipeline completa con configurazione standard
python main.py --config config/config.yaml --steps all

# Pipeline con quality checks forzati
python main.py --config config/config.yaml --steps all --enable-quality-checks

# Pipeline con configurazione avanzata (tutte le funzionalità)
python main.py --config config/config_enhanced.yaml --steps all

# Validazione configurazione senza esecuzione
python main.py --config config/config.yaml --validate-config

# Debug mode per troubleshooting
python main.py --config config/config.yaml --steps preprocessing --debug

# Step specifici
python main.py --config config/config.yaml --steps preprocessing training
```

## 📦 Dipendenze per Modello

Per alcuni modelli sono richieste librerie opzionali. Installa quelle che ti servono:

```bash
# Gradient boosting librerie esterne (opzionali)
pip install xgboost lightgbm catboost
```

- **xgboost**: necessario per `training.models.xgboost`
- **lightgbm**: necessario per `training.models.lightgbm`
- **catboost**: necessario per `training.models.catboost` e richiede il profilo `profiles.catboost.enabled: true` nel preprocessing.

Abilitare il profilo CatBoost in `config_fast_test.yaml`:
```yaml
profiles:
  catboost:
    enabled: true
```

## 📂 Struttura del Progetto

```
stimatrix-ml-pipeline/
├── main.py                 # 🎮 Orchestratore CLI con quality checks e debug
├── Makefile                # 🔧 Automazione comandi sviluppo e testing
├── config/
│   ├── config.yaml         # ⚙️ Configurazione standard
│   └── config_enhanced.yaml # 🚀 Configurazione con tutte le funzionalità avanzate
├── src/                    # 📦 Codice sorgente
│   ├── dataset_builder/    # 🗄️ Retrieval dati dal database
│   ├── db/                 # 🔌 Utilità database e schema extraction
│   ├── preprocessing/      # 🔧 Pipeline preprocessing con tracking e robust ops
│   │   ├── pipeline.py     # Pipeline integrata con quality checks
│   │   └── pipeline_tracker.py # 📊 Tracking evoluzione dataset
│   ├── training/           # 🤖 Training con feature importance e evaluation avanzata
│   │   ├── train.py        # Training integrato con sistemi avanzati
│   │   ├── feature_importance_advanced.py # 🧠 Feature importance multi-metodo
│   │   └── evaluation_advanced.py # 📈 Evaluation dual-scale
│   ├── validation/         # 🔍 Quality checks e validazione robustezza
│   │   └── quality_checks.py # Sistema completo quality assurance
│   └── utils/              # 🛠️ Utilità avanzate
│       ├── detailed_logging.py # 📝 Logging con statistiche operative
│       ├── robust_operations.py # 🛡️ Operazioni fail-safe con fallback
│       ├── temporal_advanced.py # ⏰ Utilities temporali anti-leakage
│       └── smart_config.py # ⚙️ Configuration manager intelligente
├── tests/                  # 🧪 Test suite completa con integration tests
│   ├── test_quality_checks.py # Test quality assurance
│   ├── test_robust_operations.py # Test operazioni robuste
│   ├── test_temporal_advanced.py # Test utilities temporali
│   └── test_pipeline_integration.py # Test integrazione completa
├── data/                   # 📊 Dati (ignorati da git)
│   ├── raw/                # Dati grezzi
│   └── preprocessed/       # Dati processati + tracking reports
│       └── tracking_reports/ # 📋 Report evoluzione pipeline
└── models/                 # 🎯 Modelli salvati con artifacts avanzati
    ├── feature_importance/ # 🧠 Plot e analisi feature importance
    └── evaluation/         # 📈 Report comparativi e visualizzazioni
```

## 🆕 Funzionalità Avanzate Integrate

### 🔍 **Quality Checks Automatici**
- **Data Leakage Detection**: Previene sovrapposizioni temporali e target leakage
- **Category Drift Monitoring**: Rileva cambiamenti distribuzione tra splits
- **Feature Stability Validation**: Verifica consistenza durante preprocessing

### 📊 **Pipeline Tracking Completo**
- **Evolution Monitoring**: Traccia shape, memoria, timing per ogni step
- **Performance Alerting**: Notifiche automatiche per anomalie performance
- **Report Multi-formato**: Export JSON/CSV/Excel per analisi post-execution

### 🛡️ **Operazioni Robuste**
- **Fallback Automatici**: Gestione colonne mancanti senza interruzioni
- **Error Recovery**: Continuazione pipeline anche con errori parziali
- **Validation Avanzata**: Controlli strutturali DataFrame automatici

### 🧠 **Feature Importance Multi-Metodo**
- **Consensus Analysis**: Combina Built-in + Permutation + SHAP
- **Model-Specific Optimization**: Explainer ottimizzati per tipo modello
- **Stability Metrics**: Consistenza importance tra modelli

### 📈 **Evaluation Dual-Scale**
- **Transform-Aware Metrics**: Metriche su scala trasformata E originale
- **Residual Analysis**: Test statistici per validazione assunzioni
- **Comparative Visualization**: Plot automatici performance modelli

## 🧪 Testing

### Test Rapidi (Raccomandato)
```bash
# Con Makefile (raccomandato)
make test-fast              # Test veloci
make test-quality           # Test quality checks e robust operations
make diagnose              # Diagnostica completa sistema

# Tradizionali
# Linux/macOS
./run_tests.sh basic

# Windows  
run_tests.bat basic
```

I test verificano:
- ✅ Import di tutti i moduli (inclusi sistemi avanzati)
- ✅ Feature extraction da geometrie WKT con fallback robusti
- ✅ Quality checks per data leakage prevention
- ✅ Operazioni robuste con gestione colonne mancanti
- ✅ Temporal utilities per split anti-leakage
- ✅ Pipeline tracking con evolution monitoring
- ✅ Smart configuration con risoluzione automatica
- ✅ Feature importance multi-metodo
- ✅ Evaluation dual-scale per target trasformati
- ✅ Training dei modelli con profili ottimizzati
- ✅ Sistema di logging dettagliato

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
- **`drop_non_descriptive`**: Soglia NA per il drop di colonne poco informative
- **`training`**: Configurazione modelli, tuning e valutazione (opzione `timeout` supportata)

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

### Problemi Avanzati e Soluzioni

#### 🚨 Quality Checks Failures

**Problema:** `QualityCheckError: Temporal leakage detected`
```yaml
# Soluzione: Verifica configurazione split temporale
temporal_split:
  mode: 'fraction'
  train_fraction: 0.6  # Riduci se necessario
  valid_fraction: 0.2
```

**Problema:** `Target leakage detected in features`
```yaml
# Soluzione: Rimuovi features sospette
surface:
  drop_columns:
    - 'AI_Prezzo_Originale'  # Aggiungi colonne problematiche
    - 'suspicious_feature'
```

#### 🔧 Feature Engineering Issues

**Problema:** `GeometryError: WKT parsing failed`
```yaml
# Soluzione: Disabilita feature extraction problematica
feature_extraction:
  geometry: false  # Temporaneamente
  json: true
```

**Problema:** `ProfileGenerationError: CatBoost profile failed`
```yaml
# Soluzione: Disabilita profilo problematico
profiles:
  catboost:
    enabled: false  # Temporaneamente
  tree:
    enabled: true   # Usa profilo alternativo
```

#### 📊 Performance e Memoria

**Problema:** `MemoryError during SHAP calculation`
```yaml
# Soluzione: Riduci sample size
training:
  shap:
    enabled: true
    sample_size: 100  # Era 500
    max_display: 10   # Era 30
```

**Problema:** `Pipeline troppo lenta`
```yaml
# Soluzione: Ottimizzazioni performance
training:
  models:
    catboost:
      enabled: false  # Disabilita modelli lenti
      trials: 10      # Riduci trials
```

#### 🏷️ Categorical Encoding Issues

**Problema:** `High cardinality encoding failed`
```yaml
# Soluzione: Ajusta soglie encoding
encoding:
  max_ohe_cardinality: 5  # Era 12
profiles:
  scaled:
    encoding:
      max_ohe_cardinality: 8
```

#### ⏰ Temporal Split Problems

**Problema:** `Insufficient temporal data for split`
```yaml
# Soluzione: Fallback a split random
temporal_split:
  mode: 'random'  # Invece di 'fraction'
  train_fraction: 0.8
```

### Diagnostica Avanzata

#### Verifica Stato Pipeline
```bash
# Controlla quality checks
python -c "
from src.validation.quality_checks import QualityChecker
checker = QualityChecker({'quality_checks': {}})
print('Quality Checker: OK')
"

# Controlla temporal utilities
python -c "
from src.utils.temporal_advanced import AdvancedTemporalUtils
print('Temporal Utils: OK')
"

# Controlla robust operations
python -c "
from src.utils.robust_operations import RobustDataOperations
print('Robust Operations: OK')
"
```

#### Debug Mode Avanzato
```bash
# Esegui con tracking completo
python main.py --config config/config.yaml --debug

# Controlla report tracking
ls data/preprocessed/tracking_reports/

# Analizza log dettagliati
tail -f logs/pipeline.log | grep -E "(ERROR|WARNING|CRITICAL)"
```

#### Memory Profiling
```bash
# Profiling memoria
python -m memory_profiler main.py --config config/config.yaml --steps preprocessing

# Monitoring risorse
python -c "
import psutil
print(f'RAM disponibile: {psutil.virtual_memory().available/1e9:.1f}GB')
print(f'CPU cores: {psutil.cpu_count()}')
"
```

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