# 🧪 Guida ai Test - Stimatrix ML Pipeline

## 📋 Panoramica

Questo progetto include un sistema di test completo con diverse modalità di esecuzione per verificare la qualità e il funzionamento della pipeline ML.

## 🚀 Esecuzione Rapida

### Test di Base (Raccomandato per iniziare)
```bash
# Test che verificano le funzionalità principali senza dipendenze pesanti
./run_tests.sh basic
```

### Test Completi con Pytest
```bash
# Tutti i test (richiede tutte le dipendenze)
./run_tests.sh all
```

## 📁 Struttura dei Test

```
tests/
├── conftest.py                    # Configurazione pytest e fixtures
├── test_feature_extractors.py     # Test estrazione features
├── test_preprocessing_pipeline.py # Test pipeline preprocessing  
├── test_training.py               # Test training modelli
└── __init__.py

test_basic.py                      # Test di base standalone
run_tests.sh                       # Script launcher per test
pytest.ini                         # Configurazione pytest
```

## 🎯 Modalità di Esecuzione

### 1. Test di Base (`./run_tests.sh basic`)
**✅ RACCOMANDATO per sviluppo quotidiano**

Test rapidi che verificano:
- ✅ Import di tutti i moduli
- ✅ Feature extraction WKT/geometrie  
- ✅ Validazione dati
- ✅ Funzionalità di sicurezza
- ✅ Performance utilities
- ✅ Dependency injection
- ✅ Error handling

**Vantaggi:**
- ⚡ Veloce (< 5 secondi)
- 🔧 Non richiede dipendenze pesanti (XGBoost, CatBoost, etc.)
- 🎯 Testa le funzionalità core implementate

### 2. Test Specifici

```bash
# Solo feature extractors
./run_tests.sh features

# Solo preprocessing pipeline
./run_tests.sh preprocessing  

# Solo training (richiede XGBoost, LightGBM, etc.)
./run_tests.sh training
```

### 3. Test con Coverage

```bash
# Genera report di coverage HTML
./run_tests.sh coverage
```

Il report viene salvato in `htmlcov/index.html`

### 4. Test Verbosi

```bash
# Output dettagliato per debugging
./run_tests.sh verbose
```

### 5. Test Veloce

```bash
# Test di fumo rapido
./run_tests.sh quick
```

## 🔧 Setup Dipendenze

### Dipendenze Base (per test basic)
```bash
pip install pandas numpy scikit-learn pytest psutil cryptography pyyaml category_encoders
```

### Dipendenze Complete (per test completi)
```bash
pip install -r requirements.txt
```

**Nota:** Su sistemi managed, usa `--break-system-packages`:
```bash
pip install --break-system-packages <pacchetto>
```

## 📊 Output di Esempio

### Test di Base Riusciti
```
🧪 Test Base Stimatrix ML Pipeline
==================================================
🔍 Test import moduli...
✅ Import utils: OK
✅ Import preprocessing: OK
✅ Import training: OK
✅ Import core: OK

🔧 Test feature extraction...
✅ Feature extraction WKT: OK

📊 Test data validation...  
✅ Data validation: OK

🔒 Test security...
✅ Security validation: OK

⚡ Test performance...
   Memoria originale: 283.3 KB
   Memoria ottimizzata: 10.1 KB
✅ Performance utilities: OK

🏗️ Test dependency injection...
✅ Dependency injection: OK

🛠️ Test error handling...
✅ Error handling: OK

==================================================
📊 Risultati: 7/7 test passati
🎉 Tutti i test sono passati!
```

### Test Pytest
```
============================= test session starts ==============================
platform linux -- Python 3.13.3, pytest-8.4.1, pluggy-1.6.0
collected 25 items

tests/test_feature_extractors.py::TestWKTExtraction::test_extract_point_xy_valid PASSED [ 4%]
tests/test_feature_extractors.py::TestWKTExtraction::test_extract_point_xy_invalid PASSED [ 8%]
...
```

## 🐛 Risoluzione Problemi

### Errore: "ModuleNotFoundError"
```bash
# Installa la dipendenza mancante
pip install --break-system-packages <modulo_mancante>
```

### Errore: "pytest: command not found"
```bash
# Installa pytest
pip install --break-system-packages pytest pytest-cov

# Aggiungi al PATH
export PATH=$PATH:/home/ubuntu/.local/bin
```

### Errore: "Permission denied"
```bash
# Rendi eseguibile lo script
chmod +x run_tests.sh
```

### Test Falliscono
1. **Verifica dipendenze:** controlla che tutti i moduli siano installati
2. **Usa test basic:** partire sempre dai test di base
3. **Controlla logs:** usa `./run_tests.sh verbose` per dettagli

## 📈 Coverage Report

Per generare un report di coverage dettagliato:

```bash
./run_tests.sh coverage
```

Apri `htmlcov/index.html` nel browser per vedere:
- 📊 Percentuale di coverage per file
- 🔍 Linee coperte/non coperte
- 📝 Report dettagliati per modulo

## 🎯 Test Raccomandati per Sviluppo

### Workflow Giornaliero
1. **Sviluppo rapido:** `./run_tests.sh basic` 
2. **Prima di commit:** `./run_tests.sh coverage`
3. **Prima di deploy:** `./run_tests.sh all`

### Test su CI/CD
```bash
# In pipeline CI/CD
./run_tests.sh all --cov=src --cov-fail-under=80
```

## 🔄 Aggiornare i Test

### Aggiungere Nuovi Test
1. Crea file in `tests/test_<modulo>.py`
2. Usa fixtures da `conftest.py`
3. Segui naming convention: `test_<funzione>`

### Test Personalizzati
```python
# tests/test_custom.py
import pytest

def test_my_function():
    # Il tuo test qui
    assert True
```

## 📚 Test Coverage Obiettivi

- 🎯 **Core utilities:** > 90%
- 🎯 **Feature extraction:** > 85%  
- 🎯 **Security functions:** > 95%
- 🎯 **Error handling:** > 80%
- 🎯 **Performance utilities:** > 75%

## ✅ Checklist Pre-Release

- [ ] `./run_tests.sh basic` passa al 100%
- [ ] `./run_tests.sh coverage` mostra coverage > 80%
- [ ] Nessun test fallisce in `./run_tests.sh all`
- [ ] Documentazione aggiornata
- [ ] Nuove features hanno test dedicati

---

## 🆘 Supporto

Per problemi con i test:

1. **Consulta logs:** `./run_tests.sh verbose`
2. **Test minimali:** `python3 test_basic.py`  
3. **Verifica setup:** controlla dipendenze installate
4. **Controlla documentazione:** questo file contiene le soluzioni più comuni