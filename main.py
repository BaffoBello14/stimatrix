from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

# Ensure 'src' is on sys.path when running directly
import sys as _sys
_src_path = str(Path(__file__).resolve().parent / "src")
if _src_path not in _sys.path:
    _sys.path.append(_src_path)

from utils.config import load_config
from utils.logger import setup_logger
from utils.smart_config import SmartConfigurationManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stimatrix ML Pipeline - Sistema avanzato predizione prezzi immobiliari")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                       help="Path al file configurazione YAML")
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        choices=["schema", "dataset", "preprocessing", "training", "all"],
        required=False,
        help="Steps da eseguire (uno o più). 'all' esegue pipeline completa",
    )
    parser.add_argument("--enable-quality-checks", action="store_true",
                       help="Forza abilitazione quality checks anche se disabilitati in config")
    parser.add_argument("--debug", action="store_true",
                       help="Abilita logging debug dettagliato")
    parser.add_argument("--validate-config", action="store_true",
                       help="Valida solo configurazione senza eseguire pipeline")
    return parser.parse_args()


def main() -> None:
    """
    Entry point principale per Stimatrix ML Pipeline.
    
    Gestisce:
    - Parsing argomenti CLI con opzioni avanzate
    - Validazione e ottimizzazione configurazione
    - Orchestrazione step pipeline con error handling
    - Quality checks automatici se abilitati
    - Logging configurabile (INFO/DEBUG)
    """
    args = parse_args()
    
    # Setup logging iniziale
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        print("🔍 Debug mode abilitato")
    
    # Carica e valida configurazione con smart manager
    try:
        config = load_config(args.config)
        
        # Inizializza smart configuration manager
        smart_config = SmartConfigurationManager(args.config)
        
        # Validazione configurazione se richiesta
        if args.validate_config:
            print("⚙️ Validazione configurazione...")
            if smart_config.validation_errors:
                print(f"❌ Errori: {smart_config.validation_errors}")
                return
            elif smart_config.validation_warnings:
                print(f"⚠️ Warnings: {smart_config.validation_warnings}")
            else:
                print("✅ Configurazione valida")
            
            if smart_config.applied_fallbacks:
                print(f"🔄 Fallback applicati: {smart_config.applied_fallbacks}")
            
            return
        
        # Override quality checks se forzati da CLI
        if args.enable_quality_checks:
            config.setdefault('quality_checks', {})['check_temporal_leakage'] = True
            config['quality_checks']['check_target_leakage'] = True
            print("🔍 Quality checks forzati da CLI")
        
    except Exception as e:
        print(f"❌ Errore caricamento configurazione: {e}")
        return

    # Initialize logging according to config
    setup_logger(args.config)

    # Determina steps da eseguire
    steps: List[str] = args.steps or []
    if not steps:
        print("Seleziona i passi da eseguire separati da spazio (schema, dataset, preprocessing, training, all):")
        user_input = input().strip()
        steps = user_input.split()
    if "all" in steps:
        steps = ["schema", "dataset", "preprocessing", "training"]

    print(f"🚀 Esecuzione pipeline Stimatrix: {steps}")

    # Esecuzione step con error handling robusto
    for step in steps:
        try:
            print(f"\n{'='*50}")
            print(f"📋 Esecuzione step: {step.upper()}")
            print(f"{'='*50}")
            
            if step == "schema":
                from db.schema_extract import run_schema  # lazy import
                run_schema(config)
                print(f"✅ Step {step} completato")
                
            elif step == "dataset":
                from dataset_builder.retrieval import run_dataset  # lazy import
                run_dataset(config)
                print(f"✅ Step {step} completato")
                
            elif step == "preprocessing":
                from preprocessing.pipeline import run_preprocessing  # lazy import
                output_path = run_preprocessing(config)
                print(f"✅ Step {step} completato - Output: {output_path}")
                
            elif step == "training":
                from training.train import run_training  # lazy import
                training_results = run_training(config)
                models_trained = len(training_results.get("models", {}))
                ensembles_created = len(training_results.get("ensembles", {}))
                print(f"✅ Step {step} completato - Modelli: {models_trained}, Ensemble: {ensembles_created}")
                
        except Exception as e:
            print(f"❌ Errore step {step}: {e}")
            print("🔄 Continuazione con step successivi...")
            continue

    print(f"\n{'='*60}")
    print("🎉 PIPELINE STIMATRIX COMPLETATA")
    print(f"{'='*60}")
    print("📁 Controlla le directory:")
    print("  • data/preprocessed/ - Dati processati e report")
    print("  • models/ - Modelli addestrati e artifacts")
    print("  • logs/ - Log dettagliati esecuzione")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()