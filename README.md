### Raw dataset builder per DB MSSQL

Questo progetto si connette a un database SQL Server, esegue query complesse con valori OMI, ZTL e conteggi dei punti di interesse (POI) e salva un dataset "raw" in formato Parquet.

### Requisiti
- Python 3.10+
- Driver ODBC 18 per SQL Server installato sul sistema
- Accesso a un'istanza SQL Server con le tabelle richieste

### Setup
1. Creare ed attivare un virtualenv, quindi installare le dipendenze:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Configurare le variabili d'ambiente copiando `.env.example` in `.env` e impostando `SERVER`, `DATABASE`, `DB_USER`, `DB_PASSWORD`.
   - Se hai già la variabile di sistema `USER`, evitare di usarla per l'utente DB: usare `DB_USER`.
3. Assicurarsi che il driver Microsoft ODBC 18 sia installato. Su Ubuntu:
   - seguire la guida Microsoft per installare `msodbcsql18` e `msodbcsql`: vedere la documentazione Microsoft.

### Struttura
- `src/dataset_builder/db/connect.py`: connessione al DB
- `src/dataset_builder/data/retrieval.py`: generazione query e post-processing
- `src/dataset_builder/utils/*`: logging e IO
- `schema/schema.example.json`: esempio di schema per la SELECT

### Preprocessing

La pipeline predispone i dati per ML e (in futuro) DL evitando data leakage:
- Estrazione feature (opzionale e configurabile):
  - Geometrie WKT (POINT-> x,y; POLYGON -> vertex_count), con drop delle colonne raw.
  - JSON superficiale (chiavi di primo livello), con drop della colonna raw.
- Split temporale time-ordered:
  - Modalità frazione (consigliata): `train_fraction`, `valid_fraction`, resto test.
  - Modalità data: soglia anno/mese.
- Outlier detection (solo sul train):
  - Metodi: IQR, Z-score, IsolationForest, oppure `ensemble` (maggioranza su 3).
  - Raggruppata per `AI_IdCategoriaCatastale` con `min_group_size` e `fallback_strategy`.
- Imputazione: fit sul train, transform su val/test; per numeriche (mean/median), per categoriche (most_frequent), opzionale per gruppi.
- Encoding: piano auto tra OHE (cardinalità bassa) e ordinal (alta); fit sul train, transform su val/test.
- Coercizione numerica per stringhe numeriche (decisa sul train, applicata a val/test).
- Rimozioni: colonne non descrittive (costanti/troppi NaN) e colonne altamente correlate (soglia).
- Winsorization (opzionale): clipping ai quantili per features skewed.
- Scaling (configurabile): Standard/Robust/none. PCA opzionale.
- Output:
  - `data/preprocessed/preprocessed.parquet` (compat.)
  - `X_train.parquet`, `y_train.parquet`, `X_val.parquet`, `y_val.parquet` (se `valid_fraction>0`), `X_test.parquet`, `y_test.parquet`
  - Report: `data/preprocessed/reports/preprocessing.md`

Configurazione chiave (estratto `config/config.yaml`):
- target: `column_candidates`, `log_transform`
- outliers: `method`, `z_thresh`, `iqr_factor`, `iso_forest_contamination`, `group_by_col`, `min_group_size`, `fallback_strategy`
- imputation: `numeric_strategy`, `categorical_strategy`, `group_by_col`
- encoding: `max_ohe_cardinality`
- feature_extraction: `geometry`, `json`
- temporal_split: `mode`, `train_fraction`, `valid_fraction`, `test_start_year`, `test_start_month`
- winsorization: `enabled`, `lower_quantile`, `upper_quantile`
- scaling: `scaler_type`, `with_mean`, `with_std`
- pca: `enabled`, `n_components`, `random_state`
- correlation: `numeric_threshold`

### Scelta del dataset per i modelli
- Modelli lineari/SVR/KNN: profilo “scaled” (OHE + scaling + opzionale PCA; log-target spesso utile).
- Tree/Boosting (DT, RF, GBDT, HistGBDT, XGBoost, LightGBM): profilo “tree” (no scaling/PCA, OHE moderata/ordinal; log-target utile).
- CatBoost: categoriche raw (nessuna OHE), passing indices di colonne categoriche.

Attualmente la pipeline genera un profilo numerico unico; è possibile estendere per produrre profili multipli (scaled/tree/catboost) se richiesto.

### Optuna
- Usa `X_train/y_train` per ottimizzare su `X_val/y_val` (se `valid_fraction>0`).
- Dopo tuning, riaddestra su train+val e valuta su test.

### Esecuzione
Esempio di comando:
```bash
python -m dataset_builder.cli \
  --schema schema/schema.example.json \
  --aliases A AI PC OV \
  --out data/raw.parquet
```
Opzioni:
- `--no-poi`: disabilita conteggi POI
- `--no-ztl`: disabilita flag ZTL

### Esporta schema DB
Per esportare lo schema del database in JSON:
```bash
python -m dataset_builder.db.schema_extract \
  --output schema/db_schema.json \
  --schema dbo
```

### Test
Eseguire i test offline (non richiedono il DB):
```bash
pytest
```

### Note
- La connessione usa `mssql+pyodbc` con Driver 18. Verificare che il driver sia presente sul sistema.
- Il salvataggio di Parquet richiede `pyarrow`.