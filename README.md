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