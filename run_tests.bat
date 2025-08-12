@echo off
REM Script per eseguire i test del progetto Stimatrix ML Pipeline su Windows

echo 🧪 Stimatrix ML Pipeline - Test Runner (Windows)
echo ======================================

REM Funzione per mostrare l'aiuto
if "%1"=="help" goto show_help
if "%1"=="" goto show_help

REM Controlla se pytest è disponibile
python -m pytest --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pytest non trovato. Installazione in corso...
    pip install pytest pytest-cov
)

REM Esegue il comando in base al parametro
if "%1"=="all" goto run_all
if "%1"=="features" goto run_features
if "%1"=="preprocessing" goto run_preprocessing
if "%1"=="training" goto run_training
if "%1"=="coverage" goto run_coverage
if "%1"=="quick" goto run_quick
if "%1"=="basic" goto run_basic
if "%1"=="verbose" goto run_verbose

echo ❌ Opzione non riconosciuta: %1
echo.
goto show_help

:show_help
echo Utilizzo: run_tests.bat [opzione]
echo.
echo Opzioni disponibili:
echo   all              - Esegue tutti i test
echo   features         - Esegue solo i test dei feature extractors
echo   preprocessing    - Esegue solo i test del preprocessing
echo   training         - Esegue solo i test del training
echo   coverage         - Esegue i test con report di coverage
echo   quick            - Esegue un test veloce di base
echo   basic            - Esegue test di base (senza dipendenze pesanti)
echo   verbose          - Esegue tutti i test con output verboso
echo   help             - Mostra questo aiuto
echo.
echo Esempi:
echo   run_tests.bat all
echo   run_tests.bat basic
echo   run_tests.bat coverage
goto end

:run_all
echo 🚀 Esecuzione di tutti i test...
python -m pytest tests/ -v
goto end

:run_features
echo 🔧 Test feature extractors...
python -m pytest tests/test_feature_extractors.py -v
goto end

:run_preprocessing
echo ⚙️ Test preprocessing pipeline...
python -m pytest tests/test_preprocessing_pipeline.py -v
goto end

:run_training
echo 🎯 Test training...
python -m pytest tests/test_training.py -v
goto end

:run_coverage
echo 📊 Test con coverage report...
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
echo 📈 Report HTML generato in htmlcov\index.html
goto end

:run_quick
echo ⚡ Test veloce...
python tests/test_basic.py
goto end

:run_basic
echo 🔧 Test di base (senza dipendenze pesanti)...
python tests/test_basic.py
goto end

:run_verbose
echo 📝 Test verbosi con dettagli...
python -m pytest tests/ -v -s --tb=short
goto end

:end