@echo off
REM Script veloce per test base su Windows

echo 🧪 Test Veloce Stimatrix ML Pipeline
echo =====================================

REM Controlla Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python non trovato
    echo Assicurati che Python sia installato e nel PATH
    pause
    exit /b 1
)

echo ⚡ Eseguendo test di base...
python test_basic.py

if errorlevel 1 (
    echo ❌ Test falliti
    pause
    exit /b 1
) else (
    echo ✅ Test completati con successo!
    echo.
    echo 💡 Per test completi usa: run_tests.bat all
    echo 📖 Vedi TESTING_WINDOWS.md per maggiori dettagli
)

pause