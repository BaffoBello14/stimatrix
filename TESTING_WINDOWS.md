# 🧪 Guida ai Test per Windows - Stimatrix ML Pipeline

## 🪟 Setup per Windows

### 1. Prerequisiti
- Python 3.8+ installato e nel PATH
- Git Bash, PowerShell, o Command Prompt
- Connessione internet per installare dipendenze

### 2. Aprire il Prompt dei Comandi
**Opzione A: Command Prompt**
```cmd
cd C:\path\to\stimatrix-project
```

**Opzione B: PowerShell**
```powershell
cd C:\path\to\stimatrix-project
```

**Opzione C: Git Bash** (raccomandato se disponibile)
```bash
cd /c/path/to/stimatrix-project
```

## 🚀 Esecuzione Test su Windows

### Metodo 1: Script Batch (Raccomandato)

```cmd
# Test di base (veloce, senza dipendenze pesanti)
run_tests.bat basic

# Tutti i test
run_tests.bat all

# Test specifici
run_tests.bat features
run_tests.bat preprocessing
run_tests.bat training

# Test con coverage
run_tests.bat coverage

# Test verbosi
run_tests.bat verbose

# Aiuto
run_tests.bat help
```

### Metodo 2: Comandi Diretti Python

```cmd
# Test di base
python test_basic.py

# Test con pytest (dopo aver installato le dipendenze)
python -m pytest tests/ -v

# Test specifici
python -m pytest tests/test_feature_extractors.py -v
python -m pytest tests/test_preprocessing_pipeline.py -v
python -m pytest tests/test_training.py -v

# Test con coverage
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
```

## 🔧 Installazione Dipendenze su Windows

### Dipendenze Base (per test basic)
```cmd
pip install pandas numpy scikit-learn pytest psutil cryptography pyyaml category_encoders
```

### Dipendenze Complete
```cmd
pip install -r requirements.txt
```

### Se hai problemi di permessi
```cmd
# Installa per l'utente corrente
pip install --user pandas numpy scikit-learn pytest

# O con privilegi amministratore (apri cmd come amministratore)
pip install pandas numpy scikit-learn pytest
```

## 🎯 Workflow Raccomandato per Windows

### 1. Setup Iniziale
```cmd
# Naviga alla directory del progetto
cd C:\path\to\stimatrix-project

# Installa dipendenze base
pip install pandas numpy scikit-learn pytest psutil cryptography pyyaml

# Testa che funzioni
python test_basic.py
```

### 2. Test Quotidiani
```cmd
# Test veloce durante sviluppo
run_tests.bat basic

# O direttamente:
python test_basic.py
```

### 3. Test Completi (prima di commit)
```cmd
# Installa tutte le dipendenze se non fatto
pip install -r requirements.txt

# Esegui tutti i test
run_tests.bat all

# O con coverage
run_tests.bat coverage
```

## 📊 Output di Esempio su Windows

### Test di Base Riusciti
```
C:\stimatrix> python test_basic.py

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

### Test Pytest su Windows
```
C:\stimatrix> run_tests.bat all

🧪 Stimatrix ML Pipeline - Test Runner (Windows)
======================================
🚀 Esecuzione di tutti i test...
============================= test session starts ==============================
platform win32 -- Python 3.11.5, pytest-7.4.2, pluggy-1.3.0
collected 25 items

tests\test_feature_extractors.py::TestWKTExtraction::test_extract_point_xy_valid PASSED [ 4%]
tests\test_feature_extractors.py::TestWKTExtraction::test_extract_point_xy_invalid PASSED [ 8%]
...
```

## 🐛 Risoluzione Problemi Windows

### Errore: "python: command not found"
```cmd
# Verifica che Python sia installato e nel PATH
python --version

# Se non funziona, prova:
py --version

# O usa il launcher Python:
py -m pytest tests/ -v
```

### Errore: "ModuleNotFoundError"
```cmd
# Installa il modulo mancante
pip install nome_modulo

# Se hai più versioni di Python:
py -m pip install nome_modulo
```

### Errore: "Permission denied" o "Access denied"
```cmd
# Apri Command Prompt come Amministratore
# Tasto destro su "Command Prompt" -> "Esegui come amministratore"

# O installa per l'utente corrente:
pip install --user nome_modulo
```

### Errore: "Script non riconosciuto"
```cmd
# Assicurati di essere nella directory corretta
dir
# Dovresti vedere run_tests.bat e test_basic.py

# Se non funziona, esegui direttamente:
python test_basic.py
```

### Errore: Encoding/caratteri strani
```cmd
# Imposta codifica UTF-8
chcp 65001

# Poi riesegui i test
python test_basic.py
```

## 💡 Suggerimenti per Windows

### 1. Usa Git Bash (se disponibile)
Git Bash fornisce un ambiente simile a Linux e tutti i comandi funzionano meglio:
```bash
# In Git Bash puoi usare anche lo script bash originale
./run_tests.sh basic
```

### 2. PowerShell avanzato
```powershell
# In PowerShell puoi usare alias più friendly
Set-Alias pytest 'python -m pytest'
pytest tests/ -v
```

### 3. Virtual Environment (raccomandato)
```cmd
# Crea virtual environment
python -m venv venv

# Attiva (Command Prompt)
venv\Scripts\activate

# Attiva (PowerShell)
venv\Scripts\Activate.ps1

# Installa dipendenze
pip install pandas numpy scikit-learn pytest

# Testa
python test_basic.py

# Disattiva quando finito
deactivate
```

## 📁 Struttura File Windows

```
C:\stimatrix-project\
├── run_tests.bat          # Script batch per Windows
├── run_tests.sh           # Script bash (per Git Bash)
├── test_basic.py          # Test di base standalone
├── pytest.ini            # Configurazione pytest
├── requirements.txt       # Dipendenze Python
├── tests\                 # Directory test
│   ├── conftest.py
│   ├── test_feature_extractors.py
│   ├── test_preprocessing_pipeline.py
│   └── test_training.py
└── src\                   # Codice sorgente
    ├── utils\
    ├── preprocessing\
    ├── training\
    └── core\
```

## ✅ Checklist Windows

- [ ] Python installato e funzionante (`python --version`)
- [ ] pip funzionante (`pip --version`)
- [ ] Dipendenze base installate (`pip list | findstr pandas`)
- [ ] Test base funzionanti (`python test_basic.py`)
- [ ] Script batch eseguibile (`run_tests.bat help`)

## 🆘 Supporto Windows

Se hai ancora problemi:

1. **Verifica versione Python:** `python --version` (dovrebbe essere 3.8+)
2. **Controlla PATH:** Assicurati che Python sia nel PATH di sistema
3. **Usa virtual environment:** Per evitare conflitti di dipendenze
4. **Prova Git Bash:** Spesso più compatibile con script bash originali
5. **Esegui come amministratore:** Se hai problemi di permessi

---

**Comando di test rapido per verificare tutto:**
```cmd
cd C:\path\to\stimatrix-project
python test_basic.py
```

Se questo funziona, sei pronto! 🚀