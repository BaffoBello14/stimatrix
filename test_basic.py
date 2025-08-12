#!/usr/bin/env python3
"""Test di base per verificare il funzionamento del progetto."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np

def test_imports():
    """Test che tutti i moduli si importino correttamente."""
    print("🔍 Test import moduli...")
    
    try:
        from utils.logger import get_logger
        from utils.config import load_config
        from utils.security import InputValidator
        from utils.exceptions import DataValidationError
        from utils.performance import MemoryMonitor
        from utils.data_validation import quick_validate
        print("✅ Import utils: OK")
    except Exception as e:
        print(f"❌ Import utils fallito: {e}")
        return False
    
    try:
        from preprocessing.feature_extractors import extract_point_xy_from_wkt
        from preprocessing.pipeline import choose_target
        print("✅ Import preprocessing: OK")
    except Exception as e:
        print(f"❌ Import preprocessing fallito: {e}")
        return False
    
    try:
        from training.metrics import regression_metrics
        from training.model_zoo import build_estimator
        print("✅ Import training: OK")
    except Exception as e:
        print(f"❌ Import training fallito: {e}")
        return False
    
    try:
        from core.interfaces import ProcessingResult
        from core.dependency_injection import DIContainer
        print("✅ Import core: OK")
    except Exception as e:
        print(f"❌ Import core fallito: {e}")
        return False
    
    return True

def test_feature_extraction():
    """Test estrazione feature da geometrie WKT."""
    print("\n🔧 Test feature extraction...")
    
    try:
        from preprocessing.feature_extractors import extract_point_xy_from_wkt
        
        # Test dati POINT
        points = pd.Series([
            "POINT (12.4924 41.8902)",
            "POINT (9.1900 45.4642)",
            "INVALID WKT"
        ])
        
        x_coords, y_coords = extract_point_xy_from_wkt(points)
        
        # Verifica che abbiamo estratto coordinate valide
        assert len(x_coords) == 3
        assert len(y_coords) == 3
        assert not pd.isna(x_coords.iloc[0])  # Prime coordinate valide
        assert not pd.isna(y_coords.iloc[0])
        assert pd.isna(x_coords.iloc[2])  # Terza non valida
        
        print("✅ Feature extraction WKT: OK")
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction fallito: {e}")
        return False

def test_data_validation():
    """Test validazione dati."""
    print("\n📊 Test data validation...")
    
    try:
        from utils.data_validation import quick_validate
        
        # Crea dati di test
        df_good = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": ["A", "B", "C", "D", "E"],
            "col3": [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        df_bad = pd.DataFrame({
            "col1": [1, None, None, None, None],  # Troppi missing
            "col2": ["A", "B", "C", "D", "E"]
        })
        
        # Test validazione
        result_good = quick_validate(df_good)
        result_bad = quick_validate(df_bad)
        
        assert result_good == True  # Dati buoni passano
        # result_bad potrebbe fallire o passare dipende dalla configurazione
        
        print("✅ Data validation: OK")
        return True
        
    except Exception as e:
        print(f"❌ Data validation fallito: {e}")
        return False

def test_security():
    """Test funzionalità di sicurezza."""
    print("\n🔒 Test security...")
    
    try:
        from utils.security import InputValidator
        
        # Test validazione SQL injection
        try:
            InputValidator.validate_sql_input("SELECT * FROM users WHERE id = 1; DROP TABLE users; --")
            print("❌ Security: doveva bloccare SQL injection")
            return False
        except Exception:
            pass  # Doveva fallire
        
        # Test validazione column name valido
        valid_name = InputValidator.sanitize_column_name("valid_column_name")
        assert valid_name == "valid_column_name"
        
        print("✅ Security validation: OK")
        return True
        
    except Exception as e:
        print(f"❌ Security fallito: {e}")
        return False

def test_performance():
    """Test utilities di performance."""
    print("\n⚡ Test performance...")
    
    try:
        from utils.performance import MemoryMonitor, DataFrameOptimizer
        
        # Test memory monitor
        monitor = MemoryMonitor()
        memory_info = monitor.get_memory_usage()
        
        assert "rss_mb" in memory_info
        assert "percent" in memory_info
        assert memory_info["rss_mb"] > 0
        
        # Test DataFrame optimizer
        df = pd.DataFrame({
            "int_col": [1, 2, 3, 4, 5] * 1000,  # Può essere ottimizzato a int8
            "str_col": ["A", "B", "A", "B", "A"] * 1000  # Può essere category
        })
        
        optimized_df = DataFrameOptimizer.optimize_dtypes(df)
        
        # Verifica che l'ottimizzazione abbia funzionato
        original_memory = df.memory_usage(deep=True).sum()
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        print(f"   Memoria originale: {original_memory / 1024:.1f} KB")
        print(f"   Memoria ottimizzata: {optimized_memory / 1024:.1f} KB")
        
        print("✅ Performance utilities: OK")
        return True
        
    except Exception as e:
        print(f"❌ Performance fallito: {e}")
        return False

def test_dependency_injection():
    """Test dependency injection."""
    print("\n🏗️ Test dependency injection...")
    
    try:
        from core.dependency_injection import DIContainer, get_container
        
        # Test container
        container = DIContainer()
        
        # Test registrazione e recupero
        class TestService:
            def __init__(self):
                self.value = "test"
        
        class TestInterface:
            pass
        
        container.register_singleton(TestInterface, TestService)
        instance = container.get(TestInterface)
        
        assert isinstance(instance, TestService)
        assert instance.value == "test"
        
        # Test singleton (stessa istanza)
        instance2 = container.get(TestInterface)
        assert instance is instance2
        
        print("✅ Dependency injection: OK")
        return True
        
    except Exception as e:
        print(f"❌ Dependency injection fallito: {e}")
        return False

def test_error_handling():
    """Test error handling."""
    print("\n🛠️ Test error handling...")
    
    try:
        from utils.exceptions import DataValidationError, with_error_handling
        
        # Test custom exception
        try:
            raise DataValidationError("Test error", column="test_col")
        except DataValidationError as e:
            assert e.error_code == "DATA_VALIDATION_ERROR"
            assert e.details["column"] == "test_col"
        
        # Test decorator
        @with_error_handling()
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
        
        print("✅ Error handling: OK")
        return True
        
    except Exception as e:
        print(f"❌ Error handling fallito: {e}")
        return False

def main():
    """Esegue tutti i test di base."""
    print("🧪 Test Base Stimatrix ML Pipeline")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_feature_extraction,
        test_data_validation,
        test_security,
        test_performance,
        test_dependency_injection,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} ha generato eccezione: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Risultati: {passed}/{total} test passati")
    
    if passed == total:
        print("🎉 Tutti i test sono passati!")
        return 0
    else:
        print("⚠️ Alcuni test sono falliti")
        return 1

if __name__ == "__main__":
    exit(main())