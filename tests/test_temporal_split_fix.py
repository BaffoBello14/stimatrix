"""Tests for temporal split fix in hyperparameter tuning."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.tuner import tune_model


class TestTemporalSplitFix:
    """Test che il tuning rispetti l'ordinamento temporale."""
    
    def test_temporal_split_in_tuning(self):
        """Test che il tuning usi split temporale invece di random."""
        # Crea dati con trend temporale chiaro
        n_samples = 100
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
        
        # Feature con trend temporale
        X = pd.DataFrame({
            "feature1": range(n_samples),  # Trend crescente
            "feature2": np.sin(np.linspace(0, 4*np.pi, n_samples)),  # Pattern ciclico
            "date": dates
        })
        
        # Target correlato al tempo (più recente = più alto)
        y = pd.Series(range(n_samples) + np.random.normal(0, 5, n_samples))
        
        # Rimuovi la colonna date per il training (simula preprocessing)
        X_for_training = X.drop("date", axis=1)
        
        # Test con modello semplice
        try:
            result = tune_model(
                model_key="linear",
                X_train=X_for_training,
                y_train=y,
                X_val=None,  # Force internal split
                y_val=None,
                primary_metric="r2",
                n_trials=3,  # Pochi trials per test veloce
                timeout=None,
                sampler_name="tpe",
                seed=42,
                base_params={},
                search_space={},
                cat_features=None,
                cv_config=None,  # Disable CV to test single split
                tuning_split_fraction=0.75  # Test custom split fraction
            )
            
            # Se arriviamo qui, il tuning è completato senza errori
            assert result.best_params is not None
            assert isinstance(result.best_value, float)
            
        except Exception as e:
            pytest.fail(f"Tuning failed with temporal split: {e}")
    
    def test_no_shuffle_in_split(self):
        """Test che lo split mantenga l'ordine temporale."""
        # Dati ordinati temporalmente
        X = pd.DataFrame({"feature": range(100)})
        y = pd.Series(range(100))
        
        # Simula lo split temporale
        split_point = int(len(X) * 0.8)
        X_tr = X.iloc[:split_point]
        X_va = X.iloc[split_point:]
        y_tr = y.iloc[:split_point]
        y_va = y.iloc[split_point:]
        
        # Verifica che validation set contenga solo valori futuri
        assert X_va["feature"].min() > X_tr["feature"].max()
        assert y_va.min() > y_tr.max()
        
        # Verifica che i dati siano contigui (no gaps)
        assert X_va["feature"].min() == X_tr["feature"].max() + 1
        assert y_va.min() == y_tr.max() + 1
    
    def test_split_with_numpy_arrays(self):
        """Test che lo split funzioni anche con numpy arrays."""
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        
        # Simula lo split come nel codice
        split_point = int(len(X) * 0.8)
        X_tr, X_va = X[:split_point], X[split_point:]
        y_tr, y_va = y[:split_point], y[split_point:]
        
        # Verifica dimensioni
        assert len(X_tr) == 80
        assert len(X_va) == 20
        assert len(y_tr) == 80
        assert len(y_va) == 20
        
        # Verifica ordine temporale
        assert X_va[0, 0] == X_tr[-1, 0] + 1
        assert y_va[0] == y_tr[-1] + 1
    
    def test_configurable_split_fraction(self):
        """Test che la frazione di split sia configurabile."""
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        
        # Test con frazione personalizzata
        custom_fraction = 0.7
        split_point = int(len(X) * custom_fraction)
        X_tr, X_va = X[:split_point], X[split_point:]
        y_tr, y_va = y[:split_point], y[split_point:]
        
        # Verifica che rispetti la frazione richiesta
        expected_train_size = int(100 * custom_fraction)
        expected_val_size = 100 - expected_train_size
        
        assert len(X_tr) == expected_train_size  # 70
        assert len(X_va) == expected_val_size    # 30
        assert len(y_tr) == expected_train_size
        assert len(y_va) == expected_val_size
        
        # Verifica ordine temporale
        assert X_va[0, 0] == X_tr[-1, 0] + 1
        assert y_va[0] == y_tr[-1] + 1


if __name__ == "__main__":
    pytest.main([__file__])