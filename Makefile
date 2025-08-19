# Stimatrix ML Pipeline - Makefile
# Automazione comandi per sviluppo e testing

.PHONY: help install install-dev test test-unit test-integration test-quality test-fast test-slow test-coverage lint format type-check clean setup-dev

# Default target
help:
	@echo "🏠 Stimatrix ML Pipeline - Comandi Disponibili"
	@echo ""
	@echo "📦 INSTALLAZIONE:"
	@echo "  make install          Installa dipendenze base"
	@echo "  make install-dev      Installa dipendenze sviluppo"
	@echo "  make setup-dev        Setup completo ambiente sviluppo"
	@echo ""
	@echo "🧪 TESTING:"
	@echo "  make test             Tutti i test"
	@echo "  make test-unit        Solo unit test"
	@echo "  make test-integration Solo integration test"
	@echo "  make test-quality     Test quality checks"
	@echo "  make test-fast        Solo test veloci"
	@echo "  make test-slow        Solo test lenti"
	@echo "  make test-coverage    Test con coverage report"
	@echo ""
	@echo "🔧 CODE QUALITY:"
	@echo "  make lint             Linting completo"
	@echo "  make format           Formattazione codice"
	@echo "  make type-check       Type checking"
	@echo ""
	@echo "🧹 PULIZIA:"
	@echo "  make clean            Pulizia file temporanei"
	@echo "  make clean-all        Pulizia completa"
	@echo ""
	@echo "🚀 PIPELINE:"
	@echo "  make run-basic        Esegui pipeline base"
	@echo "  make run-full         Esegui pipeline completa"

# Installazione
install:
	@echo "📦 Installazione dipendenze base..."
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	@echo "📦 Installazione dipendenze sviluppo..."
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy isort bandit safety

setup-dev: install-dev
	@echo "🔧 Setup ambiente sviluppo..."
	@if command -v pre-commit >/dev/null 2>&1; then \
		echo "Installazione pre-commit hooks..."; \
		pip install pre-commit; \
		pre-commit install; \
	fi
	@echo "✅ Ambiente sviluppo configurato"

# Testing
test:
	@echo "🧪 Esecuzione tutti i test..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	@echo "🧪 Esecuzione unit test..."
	pytest tests/ -v -m "not integration and not slow"

test-integration:
	@echo "🧪 Esecuzione integration test..."
	pytest tests/ -v -m "integration"

test-quality:
	@echo "🧪 Test quality checks specifici..."
	pytest tests/test_quality_checks.py tests/test_robust_operations.py -v

test-fast:
	@echo "⚡ Esecuzione test veloci..."
	pytest tests/ -v -m "not slow"

test-slow:
	@echo "🐌 Esecuzione test lenti..."
	pytest tests/ -v -m "slow"

test-coverage:
	@echo "📊 Test con coverage dettagliato..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term --cov-fail-under=70
	@echo "📋 Report coverage: htmlcov/index.html"

# Code Quality
lint:
	@echo "🔍 Linting completo..."
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	@echo "✅ Linting completato"

format:
	@echo "🎨 Formattazione codice..."
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile black
	@echo "✅ Codice formattato"

type-check:
	@echo "📝 Type checking..."
	mypy src/ --ignore-missing-imports --no-strict-optional
	@echo "✅ Type checking completato"

security-check:
	@echo "🔒 Security scan..."
	bandit -r src/ -f json -o security_report.json || true
	safety check --json --output safety_report.json || true
	@echo "📋 Report sicurezza: security_report.json, safety_report.json"

# Pipeline Execution
run-basic:
	@echo "🚀 Esecuzione pipeline base..."
	python main.py --config config/config.yaml --steps preprocessing training

run-full:
	@echo "🚀 Esecuzione pipeline completa..."
	python main.py --config config/config.yaml --steps all

run-with-quality-checks:
	@echo "🚀 Esecuzione con quality checks..."
	python main.py --config config/config.yaml --steps all --enable-quality-checks

# Pulizia
clean:
	@echo "🧹 Pulizia file temporanei..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ .mypy_cache/
	@echo "✅ Pulizia completata"

clean-data:
	@echo "🧹 Pulizia dati temporanei..."
	rm -rf data/preprocessed/tracking_reports/
	rm -rf data/preprocessed/*.parquet
	rm -rf models/*.pkl models/*.joblib
	@echo "✅ Dati temporanei rimossi"

clean-all: clean clean-data
	@echo "🧹 Pulizia completa..."
	rm -rf logs/*.log
	rm -rf *.json  # Report vari
	@echo "✅ Pulizia completa terminata"

# Development helpers
check-imports:
	@echo "🔍 Verifica import moduli..."
	@python -c "
import sys
from pathlib import Path
sys.path.append(str(Path('.') / 'src'))

modules = [
    'utils.detailed_logging',
    'utils.robust_operations', 
    'utils.temporal_advanced',
    'utils.smart_config',
    'validation.quality_checks',
    'training.feature_importance_advanced',
    'training.evaluation_advanced',
    'preprocessing.pipeline_tracker'
]

for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except Exception as e:
        print(f'❌ {module}: {e}')
"

validate-config:
	@echo "⚙️ Validazione configurazione..."
	@python -c "
import sys
from pathlib import Path
sys.path.append(str(Path('.') / 'src'))
from utils.smart_config import SmartConfigurationManager
try:
    manager = SmartConfigurationManager('config/config.yaml')
    print('✅ Configurazione valida')
except Exception as e:
    print(f'❌ Configurazione invalida: {e}')
"

# Info sistema
system-info:
	@echo "💻 Informazioni sistema:"
	@python -c "
import sys, platform, psutil
print(f'Python: {sys.version.split()[0]}')
print(f'Platform: {platform.platform()}')
print(f'RAM: {psutil.virtual_memory().total/1e9:.1f}GB')
print(f'CPU cores: {psutil.cpu_count()}')
print(f'Disk space: {psutil.disk_usage(\".\").free/1e9:.1f}GB free')
"

# Quick diagnostics
diagnose:
	@echo "🔍 Diagnostica rapida pipeline..."
	@make check-imports
	@make validate-config
	@make system-info
	@echo ""
	@echo "🧪 Test rapidi..."
	@pytest tests/test_robust_operations.py::TestRobustDataOperations::test_remove_columns_safe -v