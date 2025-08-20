## Stimatrix Pipeline – Guida Utente e Tecnica

### Panoramica
Stimatrix è una pipeline end-to-end per la preparazione dati, l’addestramento e la valutazione di modelli di regressione su dati immobiliari. Comprende:
- Estrazione schema DB e dataset (con arricchimenti opzionali: POI, ZTL)
- Preprocessing completo con split temporale, imputazione, encoding, scaling/PCA, pruning correlazioni e profili multipli
- Addestramento/tuning modelli con Optuna/OptunaHub e generazione di ensemble
- Valutazione e salvataggio risultati/artefatti

### Requisiti
- Python 3.10+
- OS Linux/Mac/Windows
- Accesso DB SQL Server (se si usano step `schema`/`dataset`)

### Installazione rapida
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Struttura del progetto
- `main.py`: orchestratore della pipeline (invoca i passi in base alla configurazione)
- `config/config.yaml`: configurazione standard completa
- `config/config_fast_test.yaml`: configurazione per run veloci (riduce tempi di tuning/SHAP)
- `src/`:
  - `db/`: connessione DB e estrazione schema (`connect.py`, `schema_extract.py`)
  - `dataset_builder/`: generazione query e retrieval dataset (`retrieval.py`)
  - `preprocessing/`: pipeline di preparazione dati e trasformazioni
  - `training/`: modelli, tuning, metriche, ensemble, SHAP, evaluation
  - `utils/`: logging, I/O, config, sicurezza
- `tests/`: suite PyTest con unit/integration test

### Configurazione (YAML con variabili d’ambiente)
Il loader espande `${VAR:-default}`. Esempio:
```yaml
logging:
  level: ${LOG_LEVEL:-INFO}
paths:
  raw_data: ${RAW_DATA_DIR:-'data/raw'}
```

Sezioni principali:
- `logging`: livello, formato, file, console e rotazione
- `paths`: directory/file per raw, preprocessed, schema, modelli
- `database`: opzioni retrieval e formati salvataggio
- `target`: colonna target e log-transform opzionale
- `temporal_split`: split per data o frazione (train/valid/test time-aware)
- `outliers`: IQR/Z-score/IsolationForest (anche per-gruppo) con fallback
- `imputation`: strategie numeriche/categoriche e grouping
- `encoding`: soglia per One-Hot Encoding
- `numeric_coercion`: coercizione stringhe numeriche -> numeric (con blacklist)
- `pca`, `correlation`, `drop_non_descriptive`, `feature_extraction`, `surface`, `scaling`, `winsorization`: controlli dettagli preprocessing
- `profiles`: profili multipli (es. `scaled`, `tree`, `catboost`) con override locale
- `training`: metriche, seme, parallelismo, spazi di ricerca per modello, SHAP, ensemble
- `execution`: passi da eseguire e flag di rielaborazione

Esempio completo di `execution`:
```yaml
execution:
  steps: ["all"]            # oppure [schema, dataset, preprocessing, training, evaluation]
  force_reload: false        # i passi possono usarlo per ignorare cache/output esistenti
```

### Credenziali DB e sicurezza
Le credenziali sono lette da variabili d’ambiente (caricate anche via `.env`, vedi `python-dotenv`):
- `SERVER`, `DATABASE`, `DB_USER`, `DB_PASSWORD`
Il modulo `src/db/connect.py` usa `SecureCredentialManager` e `InputValidator` (`utils/security.py`) per audit e sanificazione input, e costruisce una connessione `SQLAlchemy` MSSQL con ODBC 18 (TLS attivo), retry e test di connessione.

### Orchestrazione e CLI
Puoi definire i passi in `config.yaml` oppure passarli da CLI (la CLI ha priorità):
```bash
# Esegui i passi da config
python main.py --config config/config.yaml

# Override passi da CLI
python main.py --config config/config.yaml --steps preprocessing training evaluation

# Forza rielaborazione (se supportata dai passi)
python main.py --config config/config.yaml --force-reload

# Config fast test
python main.py --config config/config_fast_test.yaml
```
Note:
- Se `steps` non è impostato né in CLI né in config, il programma chiede input interattivo.
- `all` espande: `schema`, `dataset`, `preprocessing`, `training`, `evaluation`.

### Passi della pipeline in dettaglio
#### 1) Schema (`src/db/schema_extract.py`)
- Funzione: `run_schema(config)`
- Legge lo schema DB (con normalizzazione tipi e cattura tipi non riconosciuti) e salva JSON in `paths.schema`.
- Config rilevante: `database.schema_name`, `paths.schema`.

#### 2) Dataset (`src/dataset_builder/retrieval.py`)
- Funzione: `run_dataset(config)` → `DatasetBuilder.retrieve_data(...)`
- Costruisce SELECT da schema JSON (con alias tabelle e gestione colonne geometry/geography -> WKT), opzionale arricchimento POI e ZTL con CTE/LEFT JOIN.
- Filtro coerenza atti e stima prezzi ridistribuiti per immobile, con statistiche.
- Rimozione colonne duplicate e salvataggio in Parquet/CSV (configurabile: formato, compressione).
- Config rilevante: `paths.schema`, `paths.raw_data`, `paths.raw_filename`, `database.selected_aliases`, `database.use_poi`, `database.use_ztl`, `database.output_format`, `database.compression`.

#### 3) Preprocessing (`src/preprocessing/pipeline.py`)
- Funzione: `run_preprocessing(config)`
- Carica primo `*.parquet` da `paths.raw_data` e applica:
  - Pulizia iniziale (drop colonne completamente vuote)
  - Estrazione feature: geometry (WKT → coordinate, lunghezze/aree) e JSON; estrazione da GeoJSON se disponibile; parsing AI_Piano; normalizzazione civico
  - Costruzione `TemporalKey` da anno/mese
  - Selezione target dalla lista candidati (`target.column_candidates`), optional log1p transform (`target.log_transform`)
  - Split temporale 3-way (train/valid/test) configurabile (modalità `date` o `fraction`)
  - Rilevazione outlier su train con configurazione per-gruppo e seed globale (da `training.seed`)
  - Imputazione valori mancanti (numeriche/categoriche), con fit solo su train e trasformazione coerente su val/test; artefatti salvati in `preprocessed/artifacts`
  - Profili multipli:
    - `scaled`: OHE (max cardinalità), coercizione numerica intelligente, drop colonne non descrittive (soglia NA), winsorization, scaling, PCA, pruning correlazioni e salvataggio trasformazioni
    - `tree`: OHE senza scaling; fill NaN numeriche/categoriche; pruning correlazioni sulle sole numeriche; re-assemblaggio dataset
    - `catboost`: preserva categoriche, coercizione numerica, drop non-descrittive, pruning numeriche e lista colonne categoriche
  - Salvataggio dataset per profilo: `X_train_*`, `y_train_*`, `X_val_*`, `y_val_*`, `X_test_*`, `y_test_*` (+ target original-scale per valutazione)
  - Copie “backward-compatible” senza suffisso basate sul primo profilo abilitato, e file combinato `preprocessed.parquet`
  - Report Markdown con profili DataFrame in `preprocessed/reports/preprocessing.md`

Config chiave per preprocessing:
- `feature_extraction`, `surface.drop_columns`, `temporal_split`, `outliers`, `imputation`, `encoding.max_ohe_cardinality`, `winsorization`, `scaling`, `pca`, `correlation.numeric_threshold`, `drop_non_descriptive.na_threshold`, `profiles.*`

Artefatti salvati:
- `preprocessed/` con Parquet di train/val/test per ciascun profilo, `preprocessed/artifacts` (imputers, encoders, scaler/pca, winsorizer), `preprocessing_info.json`, report markdown.

#### 4) Training e tuning (`src/training/train.py`)
- Funzione: `run_training(config)`
- Carica dataset per ciascun modello abilitato (profilo scelto via `training.models.<key>.profile`), gestisce requirement “numeric-only” e NaN.
- Tuning con `optuna`/`optunahub`:
  - `sampler: auto` usa `optunahub` AutoSampler, altrimenti `tpe`
  - split di tuning coerente al frazionamento temporale (`temporal_split.fraction.train`)
  - `trials`/`timeout` per modello
- Calcolo baseline senza tuning, poi fit finale con best params (merge `base_params` + `best_params`).
- SHAP opzionale: salvataggio grafici/valori e sample.
- Ensemble: `voting` (con pesi euristici o tuning esterno) e `stacking` (final estimator configurabile, CV folds).
- Output:
  - `models/<model_key>/model.pkl`
  - `models/<model_key>/metrics.json` (train/test metrics, overfit, best value/params)
  - `models/<model_key>/optuna_trials.csv` (se disponibile)
  - `models/summary.json` aggregato e `models/validation_results.csv` con ranking

Config chiave per training:
- `training.primary_metric`, `training.report_metrics`, `training.seed`, `training.n_jobs_default`, `training.timeout`, `training.models.*` (abilitazione, `profile`, `trials`, `base_params`, `fit_params`, `search_space`), `training.shap`, `training.ensembles`

#### 5) Evaluation (`src/training/evaluation.py`)
- Funzione: `run_evaluation(config)`
- Carica dataset test e metriche da `models/summary.json`, opzionalmente ranking da `validation_results.csv`, crea `models/evaluation_summary.json` con top models e metriche.

### Moduli di utilità
- `utils/logger.py`: configurazione logging (file/console/rotate) e integrazione Optuna logging
- `utils/config.py`: caricamento YAML con espansione variabili d’ambiente
- `utils/io.py`: salvataggio JSON/Parquet/CSV e helper path
- `utils/security.py`: gestione sicura credenziali e validazioni input

### Esempi di configurazione
Esempio profilo `scaled` con winsorization e PCA:
```yaml
profiles:
  scaled:
    enabled: true
    output_prefix: "scaled"
    encoding:
      max_ohe_cardinality: 12
    winsorization:
      enabled: true
      lower_quantile: 0.01
      upper_quantile: 0.99
    scaling:
      scaler_type: "standard"
      with_mean: true
      with_std: true
    pca:
      enabled: true
      n_components: 0.95
```

Esempio modello `ridge` con ricerca iperparametri:
```yaml
training:
  primary_metric: "neg_root_mean_squared_error"
  sampler: "tpe"
  seed: 42
  models:
    ridge:
      enabled: true
      profile: scaled
      trials: 50
      search_space:
        alpha: {type: float, low: 1e-6, high: 1000.0, log: true}
```

### Esecuzione tipica end-to-end
```bash
# 1) Estrai schema dal DB
python main.py --config config/config.yaml --steps schema

# 2) Recupera dataset
python main.py --config config/config.yaml --steps dataset

# 3) Preprocessing (profili multipli)
python main.py --config config/config.yaml --steps preprocessing

# 4) Training (+ tuning, SHAP, ensemble)
python main.py --config config/config.yaml --steps training

# 5) Evaluation
python main.py --config config/config.yaml --steps evaluation
```

### Riproducibilità e performance
- Fissa `training.seed` e usa versioning dei file `config/*.yaml`
- Usa `config_fast_test.yaml` per cicli rapidi; passa a `config.yaml` per run finali
- Imposta `training.n_jobs_default` per parallelismo (RF/KNN/XGBoost/LightGBM, CatBoost `thread_count`)
- Aumenta `logging.level` a `DEBUG` per tracing approfondito

### Test
```bash
pytest -q
```
I test coprono: import e logging di base, estrazione feature WKT, pipeline di preprocessing (split, outliers, imputazione), caricamento dati per training, model zoo, metriche/diagnostiche, tuning con Optuna (mock), training end-to-end e gestione errori comuni.

### Troubleshooting
- DB: verifica variabili d’ambiente (`SERVER`, `DATABASE`, `DB_USER`, `DB_PASSWORD`) e connettività ODBC 18
- File mancanti: controlla `paths.*` e l’ordine dei passi
- Nan/Tipi: usa `numeric_coercion` e soglia appropriata; verifica blacklist per evitare coercioni indesiderate
- Tuning lento: riduci `trials` o usa `config_fast_test.yaml`; imposta `timeout`
- SHAP pesante: riduci `sample_size`/`max_display` o disabilita in `training.shap`

### Note su `execution.force_reload`
Il flag è propagato nella config come `config.execution.force_reload`. I moduli possono leggerlo per ignorare risultati esistenti o cache locali. Se il tuo workflow introduce caching, rispetta questo flag nella logica di skip.