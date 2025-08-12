#!/bin/bash

# Script per eseguire i test del progetto Stimatrix ML Pipeline
export PATH=$PATH:/home/ubuntu/.local/bin

echo "🧪 Stimatrix ML Pipeline - Test Runner"
echo "======================================"

# Funzione per mostrare l'aiuto
show_help() {
    echo "Utilizzo: ./run_tests.sh [opzione]"
    echo ""
    echo "Opzioni disponibili:"
    echo "  all              - Esegue tutti i test"
    echo "  features         - Esegue solo i test dei feature extractors"
    echo "  preprocessing    - Esegue solo i test del preprocessing"
    echo "  training         - Esegue solo i test del training"
    echo "  coverage         - Esegue i test con report di coverage"
    echo "  quick            - Esegue un test veloce di base"
    echo "  basic            - Esegue test di base (senza dipendenze pesanti)"
    echo "  verbose          - Esegue tutti i test con output verboso"
    echo "  help             - Mostra questo aiuto"
    echo ""
    echo "Esempi:"
    echo "  ./run_tests.sh all"
    echo "  ./run_tests.sh features"
    echo "  ./run_tests.sh coverage"
}

# Controlla se pytest è disponibile
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest non trovato. Installazione in corso..."
    pip install --break-system-packages pytest pytest-cov
fi

case "${1:-all}" in
    "help")
        show_help
        ;;
    "all")
        echo "🚀 Esecuzione di tutti i test..."
        python3 -m pytest tests/ -v
        ;;
    "features")
        echo "🔧 Test feature extractors..."
        python3 -m pytest tests/test_feature_extractors.py -v
        ;;
    "preprocessing")
        echo "⚙️ Test preprocessing pipeline..."
        python3 -m pytest tests/test_preprocessing_pipeline.py -v
        ;;
    "training")
        echo "🎯 Test training..."
        python3 -m pytest tests/test_training.py -v
        ;;
    "coverage")
        echo "📊 Test con coverage report..."
        python3 -m pytest tests/ --cov=src --cov-report=html --cov-report=term
        echo "📈 Report HTML generato in htmlcov/index.html"
        ;;
    "quick")
        echo "⚡ Test veloce..."
        python3 tests/test_basic.py
        ;;
    "basic")
        echo "🔧 Test di base (senza dipendenze pesanti)..."
        python3 tests/test_basic.py
        ;;
    "verbose")
        echo "📝 Test verbosi con dettagli..."
        python3 -m pytest tests/ -v -s --tb=short
        ;;
    *)
        echo "❌ Opzione non riconosciuta: $1"
        echo ""
        show_help
        exit 1
        ;;
esac