## Repository: Data Retrieval and Raw Dataset Builder

Prerequisiti:
- Python 3.11+
- ODBC Driver 18 for SQL Server installato

Setup:
1. Creare e compilare `.env` partendo da `.env.example` (DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD)
2. Installare dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

Estrazione schema DB:
```bash
python scripts/extract_schema.py --output data/db_schema.json --schema dbo --include-spatial
```

Recupero dati raw (parquet):
```bash
python scripts/retrieve_data.py --schema data/db_schema.json --aliases A AI PC OV OZ ISC II --output data/raw/dataset.parquet
```

Costruzione dataset base:
```bash
python scripts/build_dataset.py --config config/config.yaml --input data/raw/dataset.parquet --output data/processed/dataset_base.parquet
```

Note miglioramenti rispetto al codice originale:
- Gestione `TOP N` corretta su query con CTE per i test
- Supporto robusto a categorie POI numeriche e alias colonne stabili
- Alias di tabella univoci in estrazione schema
- Opzione per includere colonne spaziali in WKT
- Connessione DB con variabili `DB_*` e test via SQLAlchemy 2.0 `text()`