#!/bin/bash
# Script per eseguire la pipeline ottimizzata

echo "======================================"
echo "STIMATRIX - Pipeline Ottimizzata"
echo "======================================"
echo ""

# Verifica che Python sia disponibile
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 non trovato!"
    exit 1
fi

# Verifica che il file di configurazione ottimizzato esista
if [ ! -f "config/config_optimized.yaml" ]; then
    echo "ERROR: File config/config_optimized.yaml non trovato!"
    echo "Assicurati di aver creato il file di configurazione ottimizzato."
    exit 1
fi

# Opzioni di esecuzione
STEPS="all"
FORCE=false

# Parse argomenti
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --steps)
            STEPS="$2"
            shift
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help)
            echo "Uso: $0 [opzioni]"
            echo ""
            echo "Opzioni:"
            echo "  --steps STEPS    Steps da eseguire (default: all)"
            echo "                   Opzioni: schema, dataset, preprocessing, training, evaluation, all"
            echo "  --force          Forza la riesecuzione di tutti gli steps"
            echo "  --help           Mostra questo messaggio"
            exit 0
            ;;
        *)
            echo "Opzione sconosciuta: $1"
            echo "Usa --help per vedere le opzioni disponibili"
            exit 1
            ;;
    esac
done

# Costruisci comando
CMD="python3 main_optimized.py --config config/config_optimized.yaml"

if [ "$STEPS" != "all" ]; then
    CMD="$CMD --steps $STEPS"
fi

if [ "$FORCE" = true ]; then
    CMD="$CMD --force"
fi

# Esegui
echo "Eseguendo: $CMD"
echo ""

$CMD

# Controlla risultato
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Pipeline completata con successo!"
    echo "======================================"
    
    # Mostra risultati se disponibili
    if [ -f "models/model_comparison_optimized.csv" ]; then
        echo ""
        echo "Top 5 modelli per RMSE:"
        python3 -c "
import pandas as pd
df = pd.read_csv('models/model_comparison_optimized.csv')
print(df.head()[['Model', 'Category', 'Test_R2', 'Test_RMSE']].to_string(index=False))
"
    fi
else
    echo ""
    echo "======================================"
    echo "ERRORE: Pipeline fallita!"
    echo "======================================"
    exit 1
fi