# Notebooks

Raccolta di notebook utilizzati per l’esplorazione dei dati, i confronti tra encoding/trasformazioni e l’analisi dei risultati di training.

## Primi passi

- Crea l’ambiente: `pip install -r requirements.txt`
- Avvia Jupyter da root repo per mantenere il `PYTHONPATH` corretto (`src/` serve per gli import interni)
- I notebook assumono la presenza degli artefatti nelle cartelle `data/` e `models/` generate dalla pipeline (`python main.py --config config/config.yaml --steps preprocess train evaluate`)

## Contenuto

- `eda_project_analysis.ipynb`: panoramica generale del dataset, distribuzioni e prime correlazioni
- `encoding_strategies_comparison.ipynb`: confronto di diverse strategie di encoding per le feature categoriche
- `outlier_detection_analysis.ipynb`: analizza metodologie di individuazione e trattamento degli outlier
- `target_transformations_comparison.ipynb`: valuta trasformazioni del target (log, box-cox, smearing, ecc.)
- `model_results_deep_analysis.ipynb`: analisi approfondita delle metriche salvate in `models/` (leaderboard, segmentazioni, prediction intervals, worst predictions)

Le utility comuni (funzioni di plotting, helper per EDA) sono raccolte in `eda_utils.py`.

## Suggerimenti

- Imposta `PYTHONPATH` con `export PYTHONPATH="$PYTHONPATH:$(pwd)/src"` se lanci i notebook da un IDE diverso da Jupyter Lab
- I notebook sono pensati per essere modulari: puoi eseguire solo le sezioni di interesse e puntare direttamente ai file generati
