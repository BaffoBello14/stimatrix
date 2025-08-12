# Stimatrix ML Pipeline

## Panoramica
Pipeline modulare per la preparazione dati (feature extraction, imputazione, encoding, split temporale, scaling/PCA, pruning) e addestramento modelli ML (alberi, linear, boosting, catboost) per la stima di prezzi immobiliari.

## Requisiti
- Python 3.13+
- Dipendenze: vedi `requirements.txt` (la pipeline si appoggia a pandas/pyarrow, scikit-learn, category_encoders, xgboost/lightgbm/catboost, optuna, shap).

Suggerito l'uso di un virtualenv:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Struttura progetto
- `main.py`: orchestratore CLI (schema, dataset, preprocessing, training)
- `config/config.yaml`: configurazione della pipeline
- `src/`:
  - `dataset_builder/`: retrieval dal DB secondo lo schema
  - `db/`: utilità DB e schema extraction
  - `preprocessing/`: pipeline, feature extractors, encoding, imputazione, trasformazioni, outlier detection, report
  - `training/`: training, tuning, metriche, ensemble, SHAP
  - `utils/`: logging, IO
- `tests/`: test suite completa
- Dati:
  - raw: `data/raw/raw.parquet`
  - preprocessed: `data/preprocessed/*`

## Testing

### 🚀 Esecuzione Rapida

#### Test di Base (Raccomandato)
```bash
# Linux/macOS
./run_tests.sh basic

# Windows
run_tests.bat basic
```

Test rapidi che verificano le funzionalità principali senza dipendenze pesanti:
- ✅ Import di tutti i moduli
- ✅ Feature extraction WKT/geometrie  
- ✅ Validazione dati
- ✅ Model zoo e training
- ✅ Sistema di logging

#### Test Completi
```bash
# Linux/macOS
./run_tests.sh all

# Windows  
run_tests.bat all
```

### 📁 Struttura Test
```
tests/
├── conftest.py                    # Configurazione pytest e fixtures
├── test_basic.py                  # Test di base standalone
├── test_feature_extractors.py     # Test estrazione features
├── test_preprocessing_pipeline.py # Test pipeline preprocessing  
├── test_training.py               # Test training modelli
└── __init__.py
```

### 🎯 Modalità Disponibili

| Comando | Descrizione |
|---------|-------------|
| `basic` | Test di base senza dipendenze pesanti (raccomandato) |
| `all` | Tutti i test con pytest |
| `features` | Solo test feature extractors |
| `preprocessing` | Solo test pipeline preprocessing |
| `training` | Solo test training modelli |
| `coverage` | Test con report di copertura |
| `verbose` | Test con output dettagliato |

### 🪟 Note per Windows

I test sono completamente compatibili con Windows. Utilizzare:
- **Command Prompt**: `run_tests.bat basic`
- **PowerShell**: `run_tests.bat basic`  
- **Git Bash**: `./run_tests.sh basic`

Per installazione dipendenze su Windows:
```cmd
pip install -r requirements.txt
```

## Esecuzione
Esegui uno o più step:
```bash
python main.py --config config/config.yaml --steps preprocessing training
# oppure
python main.py --config config/config.yaml --steps all
```

## Configurazione (principali chiavi)
- `paths`: percorsi input/output
- `target`:
  - `column_candidates`: ordine di preferenza per la target (default `AI_Prezzo_Ridistribuito`)
  - `log_transform`: applica log1p al target
- `feature_extraction`:
  - `geometry`: abilita l'estrazione da colonne WKT (`POINT`, `POLYGON`, `MULTIPOLYGON`)
  - `json`: abilita l'estrazione da JSON generici (per GeoJSON usa l'estrattore dedicato)
- `temporal_split`: split time-ordered (fraction o by-date)
- `outliers`: rimozione outlier sul train (anche per gruppo)
- `imputation`: imputazione numeriche/categoriche (anche per gruppo)
- `encoding`: piani encoding (OHE/ordinal) con `max_ohe_cardinality`
- `profiles`: tre profili (`scaled`, `tree`, `catboost`) con pipeline dedicate e salvataggio separato degli output
- `numeric_coercion`: soglia e lista di pattern per escludere colonne da coercizione numerica
- `training`: modelli, iperparametri, tuning, SHAP e ensemble. Include `n_jobs_default` per il parallelismo e `cv_when_no_val` per abilitare la cross-validation quando non si crea un validation set esterno.

## Preprocessing: logica chiave
1) Caricamento raw (`data/raw/*.parquet`), drop colonne completamente vuote.
2) Estrazione feature da geometrie WKT e GeoJSON:
   - `POINT (lon lat)`: crea `*_x`, `*_y` e droppa la colonna raw.
   - `POLYGON`: `*_vertex_count` (outer ring).
   - `MULTIPOLYGON`: `__wkt_mpoly_count`, `__wkt_mpoly_vertices`, `__wkt_mpoly_outer_vertices_avg` e drop raw.
   - GeoJSON (es. `PC_PoligonoGeoJson`): estrae `areaMq`, `perimetroM`, `codiceCatastale`, `foglio`, `sezione`, `particella` e bbox (`minx,miny,maxx,maxy`), poi drop raw.
3) Normalizzazioni mirate:
   - Superfici: si mantiene solo `AI_Superficie` (m²). Le colonne da droppare sono configurabili in `surface.drop_columns` (default include `AI_SuperficieCalcolata`, `AI_SuperficieVisura*`, ecc.).
   - `AI_Piano`: feature engineering robusto (min/max/n_floors/span, pesata, flag basement/ground/upper, conteggi) e drop della colonna raw `AI_Piano`.
   - `AI_Civico`: estrazione parte numerica in `AI_Civico_num` e drop della colonna raw.
   - Geo SRID costante: drop `PC_PoligonoMetricoSrid`.
4) Creazione chiave temporale `TemporalKey` se disponibili `A_AnnoStipula`, `A_MeseStipula`.
5) Scelta target (preferenza da `config.target.column_candidates`).
6) Split temporale train/val/test (evita leakage).
7) Outlier detection sul train (globale o per gruppo).
8) Imputation (numeriche/categoriche) con fitting sul solo train.
9) Profili:
   - `scaled`: encoding (OHE/ordinal), coercizione numerica più sicura (soglia da `numeric_coercion.threshold`, blacklist da `numeric_coercion.blacklist_patterns`), drop non descrittive, winsor (opzionale), scaling e PCA (opzionale), pruning correlazioni, salvataggi.
   - `tree`: encoding, coercizione numerica, drop non descrittive, pruning solo su numeriche e reallineamento colonne.
   - `catboost`: preserva categoriche, coercizione numerica, drop non descrittive, pruning numeriche.
10) Salvataggi: `X_train_*`, `y_train_*`, `X_val_*`, `X_test_*`, con copia "back-compat" e `preprocessed.parquet` combinato.

## Dettagli implementativi
- Geometrie: parsing WKT/GeoJSON senza dipendenze GIS. Per esigenze più avanzate, valutare `shapely/geopandas`.
- `AI_Piano`: parsing robusto di token (P1..P12, S/S1/S2, PT/T/ST, RIAL/AMMEZZATO, numeri, range 1-3), aggregazioni e flag mirati.
- Coercizione numerica: attiva su colonne object con ratio conversione ≥ soglia (default 0.95). La blacklist di pattern è configurabile in `config.yaml` alla chiave `numeric_coercion.blacklist_patterns` (esempi inclusi: `II_*`, `AI_Id*`, `Foglio`, `Particella*`, `Subalterno`, `SezioneAmministrativa`, `ZonaOmi`, `*COD*`).
- Parallelismo: per i modelli che lo supportano si usa `n_jobs = -1` (o `thread_count` per CatBoost) configurabile da `training.n_jobs_default`.
- **LightGBM**: utilizza sempre numpy arrays (`.values`) per evitare warning sui feature names.

## Output
- `data/preprocessed/` contiene file per ogni profilo.
- In modalità catboost viene anche salvata la lista delle categoriche rilevate.

## Note e convenzioni
- Tutte le superfici sono in m² via `AI_Superficie`. Qualsiasi altra fonte di area viene scartata.
- Colonne costanti e non descrittive vengono droppate automaticamente.
- Evitiamo leakage tramite split temporale e fitting solo su train di imputers/encoders/scalers/PCA/winsorizer.

## Troubleshooting
- Manca un pacchetto Python: assicurati di aver attivato il virtualenv e installato `requirements.txt`.
- Errori di import: verifica `PYTHONPATH` includa `src` o avvia tramite `main.py`.
- File raw assente: posiziona `data/raw/raw.parquet` o esegui lo step `dataset`.
- Test falliscono: prova prima `./run_tests.sh basic` per verificare le funzionalità core.

## Licenza
Proprietario. Uso interno.