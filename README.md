### Esecuzione
```bash
# Esegui l'intera pipeline (incluso evaluation)
python main.py --config config/config.yaml --steps all

# Oppure step specifici
python main.py --config config/config.yaml --steps preprocessing training evaluation
```