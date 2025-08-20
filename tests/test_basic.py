"""Test di base per verificare il funzionamento delle componenti principali."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test che tutti i moduli principali si importino correttamente."""
    try:
        from preprocessing import pipeline
        from training import train, model_zoo, tuner
        from utils import logger
        from db import schema_extract
        from preprocessing import feature_extractors
        print("✓ Tutti i moduli si importano correttamente")
    except ImportError as e:
        print(f"✗ Errore import: {e}")
        assert False, f"Errore import: {e}"


def test_feature_extraction_basic():
    """Test basilare di estrazione feature da geometrie."""
    try:
        from preprocessing.feature_extractors import extract_geometry_features
        
        # Test POINT
        point_data = {"wkt_col": ["POINT (12.4924 41.8902)", "POINT (9.1895 45.4642)"]}
        
        import pandas as pd
        df = pd.DataFrame(point_data)
        out_df, dropped = extract_geometry_features(df)
        
        # Verifica che abbia estratto coordinate
        assert "wkt_col_x" in out_df.columns
        assert "wkt_col_y" in out_df.columns
        assert len(out_df) == 2
        
        print("✓ Estrazione feature WKT funziona")
    except Exception as e:
        print(f"✗ Errore feature extraction: {e}")
        assert False, f"Errore feature extraction: {e}"


def test_model_zoo():
    """Test che il model zoo crei correttamente i modelli."""
    try:
        from training.model_zoo import build_estimator
        
        # Test creazione modelli base
        models_to_test = [
            ("linear", {}),
            ("ridge", {"alpha": 1.0}),
        ]
        # Add LightGBM only if importable
        try:
            import lightgbm  # type: ignore
            models_to_test.append(("lightgbm", {"n_estimators": 10, "verbose": -1}))
        except Exception:
            pass
        
        for model_key, params in models_to_test:
            try:
                model = build_estimator(model_key, params)
                print(f"✓ Modello {model_key} creato correttamente")
            except Exception as e:
                print(f"✗ Errore creazione modello {model_key}: {e}")
                assert False, f"Errore creazione modello {model_key}: {e}"
    except Exception as e:
        print(f"✗ Errore model zoo: {e}")
        assert False, f"Errore model zoo: {e}"


def test_data_validation():
    """Test validazione dati di base."""
    try:
        import pandas as pd
        import numpy as np
        
        # Test dati con problemi comuni
        data = pd.DataFrame({
            "numeric_col": [1, 2, np.nan, 4, 5],
            "text_col": ["a", "b", None, "d", "e"],
            "constant_col": [1, 1, 1, 1, 1],
            "empty_col": [None, None, None, None, None]
        })
        
        # Test identificazione colonne problematiche
        numeric_nulls = data["numeric_col"].isnull().sum()
        text_nulls = data["text_col"].isnull().sum()
        constant_check = data["constant_col"].nunique() == 1
        empty_check = data["empty_col"].isnull().all()
        
        assert numeric_nulls == 1
        assert text_nulls == 1
        assert constant_check
        assert empty_check
        
        print("✓ Validazione dati funziona")
    except Exception as e:
        print(f"✗ Errore validazione dati: {e}")
        assert False, f"Errore validazione dati: {e}"


def test_logger():
    """Test sistema di logging."""
    try:
        from utils.logger import get_logger
        
        logger = get_logger("test_logger")
        logger.info("Test logging message")
        
        print("✓ Sistema di logging funziona")
    except Exception as e:
        print(f"✗ Errore logging: {e}")
        assert False, f"Errore logging: {e}"


def run_basic_tests():
    """Esegue tutti i test di base."""
    print("🧪 Esecuzione test di base...\n")
    
    tests = [
        ("Import moduli", test_imports),
        ("Feature extraction", test_feature_extraction_basic),
        ("Model zoo", test_model_zoo),
        ("Validazione dati", test_data_validation),
        ("Sistema logging", test_logger),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"🔍 Test: {test_name}")
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Errore inaspettato in {test_name}: {e}\n")
    
    print("="*50)
    print(f"📊 Risultati: {passed}/{total} test passati")
    
    if passed == total:
        print("🎉 Tutti i test di base sono passati!")
        return True
    else:
        print("❌ Alcuni test hanno fallito")
        return False


if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)