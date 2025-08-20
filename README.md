## Stimatrix Pipeline – Guida Completa

### Requisiti
- Python 3.10+
- Sistema operativo Linux/Mac/Windows
- Consigliato: virtualenv o conda

### Installazione
```bash
python -m venv .venv
source .venv/bin/activate  # su Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Struttura del progetto
- `main.py`: orchestratore della pipeline
- `config/config.yaml`: configurazione standard
- `config/config_fast_test.yaml`: configurazione per run veloci
- `src/`: implementazione (preprocessing, training, evaluation, utils, ...)
- `tests/`: test automatici

### Configurazione
La configurazione è in YAML ed espande variabili d'ambiente nel formato `${VAR:-default}`.

- `logging`: livello, formato, destinazioni
- `paths`: directory e file principali
- `database`: opzioni di estrazione dati
- `target`: colonna target e trasformazioni
- `outliers`, `imputation`, `encoding`, `temporal_split`, `numeric_coercion`, `pca`, `correlation`, `drop_non_descriptive`, `feature_extraction`, `surface`, `scaling`, `winsorization`, `profiles`: controlli del preprocessing
- `training`: metriche, seme, parallelismo, modelli, spazi di ricerca, SHAP, ensemble
- `execution`:
  - `steps`: lista dei passi da eseguire tra `schema`, `dataset`, `preprocessing`, `training`, `evaluation`. Puoi usare anche `all`.
  - `force_reload`: se true, forza la rielaborazione anche se esistono output intermedi.

Esempio minimale di `execution`:
```yaml
execution:
  steps: ["all"]
  force_reload: false
```

### Esecuzione
Puoi controllare i passi da `config.yaml` oppure da CLI (la CLI ha priorità e può fare override).

- Esecuzione completa leggendo i passi da config:
```bash
python main.py --config config/config.yaml
```

- Override dei passi da CLI (esempio: solo preprocessing, training, evaluation):
```bash
python main.py --config config/config.yaml --steps preprocessing training evaluation
```

- Esecuzione completa con config FAST TEST:
```bash
python main.py --config config/config_fast_test.yaml
```

- Forzare la rielaborazione indipendentemente dal caching intermedio:
```bash
python main.py --config config/config.yaml --force-reload
```

Note:
- Se né CLI né config specificano `steps`, all'avvio verrà chiesto interattivamente.
- Il valore `all` è espanso in: `schema`, `dataset`, `preprocessing`, `training`, `evaluation`.

### Passi della pipeline
- `schema`: estrazione e salvataggio dello schema dal database
- `dataset`: retrieval e salvataggio del dataset grezzo
- `preprocessing`: pulizia, imputazione, encoding, split temporale, generazione profili
- `training`: tuning/allenamento dei modelli e salvataggio dei risultati
- `evaluation`: valutazioni finali e report

### Best practice
- Usa `config_fast_test.yaml` per iterazioni rapide; passa a `config.yaml` per run completi
- Versiona i file di config e mantieni coerenza tra ambienti
- Fissa `SEED` via env o config per esperimenti riproducibili

### Test
```bash
pytest -q
```

### Troubleshooting
- Verifica i percorsi in `paths`
- Aumenta `logging.level` a `DEBUG` per maggiori dettagli
- Usa `--force-reload` se sospetti cache corrotte o risultati obsoleti