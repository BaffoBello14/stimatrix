#!/usr/bin/env python3
"""
Script per testare le configurazioni corrette.
"""

import sys
from pathlib import Path

# Ensure 'src' is on sys.path when running directly
_src_path = str(Path(__file__).resolve().parent / "src")
if _src_path not in sys.path:
    sys.path.append(_src_path)

from utils.config import load_config
from utils.logger import setup_logger
from training.early_stopping_utils import prepare_early_stopping_params
import pandas as pd
import numpy as np


def test_config_loading():
    """Test caricamento configurazioni."""
    
    configs_to_test = [
        "config/config.yaml",
        "config/config_optimized.yaml", 
        "config/config_safe.yaml"
    ]
    
    print("🔍 TEST CARICAMENTO CONFIGURAZIONI")
    print("=" * 50)
    
    for config_path in configs_to_test:
        try:
            config = load_config(config_path)
            print(f"✅ {config_path}: OK")
            
            # Test sezioni critiche
            assert "training" in config, f"Sezione 'training' mancante in {config_path}"
            assert "models" in config["training"], f"Sezione 'models' mancante in {config_path}"
            
            # Test modelli abilitati
            enabled_models = [k for k, v in config["training"]["models"].items() if v.get("enabled", False)]
            print(f"   Modelli abilitati: {enabled_models}")
            
            # Test trials
            trials_info = {
                "base": config["training"].get("trials_base", "N/A"),
                "advanced": config["training"].get("trials_advanced", "N/A"), 
                "conservative": config["training"].get("trials_conservative", "N/A")
            }
            print(f"   Trials: {trials_info}")
            
        except Exception as e:
            print(f"❌ {config_path}: ERRORE - {e}")
    
    print()


def test_early_stopping_utils():
    """Test utilità early stopping."""
    
    print("🔍 TEST EARLY STOPPING UTILS")
    print("=" * 50)
    
    # Dati di test
    X_val = pd.DataFrame(np.random.randn(100, 5))
    y_val = pd.Series(np.random.randn(100))
    
    # Test parametri diversi modelli
    test_cases = [
        {
            "model_key": "lightgbm",
            "base_params": {"n_estimators": 1000, "early_stopping_rounds": 50, "learning_rate": 0.1},
            "fit_params": {"eval_metric": "rmse"}
        },
        {
            "model_key": "xgboost", 
            "base_params": {"n_estimators": 1000, "early_stopping_rounds": 50, "learning_rate": 0.1},
            "fit_params": {"eval_metric": "rmse"}
        },
        {
            "model_key": "catboost",
            "base_params": {"iterations": 1000, "early_stopping_rounds": 50, "use_best_model": True},
            "fit_params": {"cat_features": []}
        },
        {
            "model_key": "ridge",
            "base_params": {"alpha": 1.0},
            "fit_params": {}
        }
    ]
    
    for case in test_cases:
        try:
            # Test con validation set
            clean_params, enhanced_fit = prepare_early_stopping_params(
                case["model_key"], case["base_params"], case["fit_params"], X_val, y_val
            )
            print(f"✅ {case['model_key']} (con val): OK")
            print(f"   Clean params keys: {list(clean_params.keys())}")
            print(f"   Enhanced fit keys: {list(enhanced_fit.keys())}")
            
            # Test senza validation set
            clean_params_no_val, enhanced_fit_no_val = prepare_early_stopping_params(
                case["model_key"], case["base_params"], case["fit_params"], None, None
            )
            print(f"✅ {case['model_key']} (senza val): OK")
            print(f"   N_estimators ridotto: {'n_estimators' in clean_params_no_val or 'iterations' in clean_params_no_val}")
            
        except Exception as e:
            print(f"❌ {case['model_key']}: ERRORE - {e}")
    
    print()


def test_feature_extraction_config():
    """Test configurazione feature extraction."""
    
    print("🔍 TEST FEATURE EXTRACTION CONFIG")
    print("=" * 50)
    
    config = load_config("config/config_safe.yaml")
    fe_config = config.get("feature_extraction", {}).get("derived_features", {})
    
    if fe_config.get("enabled", False):
        print("✅ Feature derivate abilitate")
        
        # Test price ratios
        price_ratios = fe_config.get("price_ratios", {}).get("features", [])
        print(f"   Price ratios: {price_ratios}")
        
        # Verifica nessun data leakage
        leakage_features = ["prezzo_per_mq", "prezzo_vs_rendita", "prezzo_medio_gruppo"]
        found_leakage = [f for f in price_ratios if any(leak in f for leak in leakage_features)]
        
        if found_leakage:
            print(f"❌ DATA LEAKAGE TROVATO: {found_leakage}")
        else:
            print("✅ Nessun data leakage trovato")
        
        # Test altre feature
        spatial_features = fe_config.get("spatial_features", {}).get("features", [])
        temporal_features = fe_config.get("temporal_features", {}).get("features", [])
        categorical_features = fe_config.get("categorical_aggregates", {}).get("features", [])
        
        print(f"   Spatial features: {len(spatial_features)}")
        print(f"   Temporal features: {len(temporal_features)}")
        print(f"   Categorical features: {len(categorical_features)}")
        
    else:
        print("❌ Feature derivate disabilitate")
    
    print()


def main():
    """Esegue tutti i test."""
    
    print("🧪 TEST CONFIGURAZIONI STIMATRIX")
    print("=" * 60)
    print()
    
    try:
        test_config_loading()
        test_early_stopping_utils()
        test_feature_extraction_config()
        
        print("🎉 TUTTI I TEST COMPLETATI")
        print("=" * 60)
        print()
        print("📋 CONFIGURAZIONI DISPONIBILI:")
        print("  • config/config.yaml - Configurazione generale aggiornata")
        print("  • config/config_optimized.yaml - Configurazione ottimizzata")  
        print("  • config/config_safe.yaml - Configurazione ultra-sicura")
        print()
        print("🚀 RACCOMANDAZIONE:")
        print("  python main.py --config config/config_safe.yaml")
        
    except Exception as e:
        print(f"❌ ERRORE NEI TEST: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()