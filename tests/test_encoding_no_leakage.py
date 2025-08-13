"""Tests to verify encoding does not cause data leakage."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.encoders import plan_encodings, fit_apply_encoders, transform_with_encoders


class TestEncodingNoLeakage:
    """Test che l'encoding non causi data leakage."""
    
    def test_encoding_fit_only_on_train(self):
        """Test che gli encoder siano fit solo su training set."""
        # Crea train set con categorie A, B, C
        train_data = pd.DataFrame({
            "cat1": ["A", "B", "C", "A", "B"] * 10,
            "cat2": ["X", "Y", "X", "Y", "X"] * 10,
            "num": range(50)
        })
        
        # Crea validation set con categoria nuova "D"
        val_data = pd.DataFrame({
            "cat1": ["A", "B", "D", "D", "A"],  # D è categoria unseen
            "cat2": ["X", "Y", "Z", "Z", "X"],  # Z è categoria unseen  
            "num": [100, 101, 102, 103, 104]
        })
        
        # Test del flusso corretto: fit solo su train
        plan = plan_encodings(train_data, max_ohe_cardinality=10)
        train_encoded, fitted_encoders, dropped_cols = fit_apply_encoders(train_data, plan)
        
        # Verifica che il fit abbia imparato solo le categorie del train
        assert fitted_encoders.one_hot is not None
        expected_features = set(fitted_encoders.one_hot.get_feature_names_out())
        
        # Deve contenere solo categorie viste nel train (A, B, C, X, Y)
        assert any("cat1_A" in f for f in expected_features)
        assert any("cat1_B" in f for f in expected_features) 
        assert any("cat1_C" in f for f in expected_features)
        assert any("cat2_X" in f for f in expected_features)
        assert any("cat2_Y" in f for f in expected_features)
        
        # Non deve contenere categorie unseen (D, Z)
        assert not any("cat1_D" in f for f in expected_features)
        assert not any("cat2_Z" in f for f in expected_features)
        
        # Transform su validation (con categorie unseen)
        val_encoded = transform_with_encoders(val_data, fitted_encoders)
        
        # Verifica che il transform gestisca correttamente categorie unseen
        assert len(val_encoded) == len(val_data), "Non dovrebbe perdere righe"
        assert not val_encoded.isnull().all().any(), "Non dovrebbe avere colonne completamente NaN"
        
        # Verifica che le colonne siano allineate
        train_ohe_cols = [c for c in train_encoded.columns if any(cat in c for cat in ["cat1_", "cat2_"])]
        val_ohe_cols = [c for c in val_encoded.columns if any(cat in c for cat in ["cat1_", "cat2_"])]
        assert set(train_ohe_cols) == set(val_ohe_cols), "Colonne OHE dovrebbero essere identiche"
    
    def test_ordinal_encoding_handles_unseen(self):
        """Test che ordinal encoding gestisca categorie unseen."""
        # Train con cardinalità alta per forzare ordinal encoding
        categories = [f"cat_{i}" for i in range(20)]
        train_data = pd.DataFrame({
            "high_card": np.random.choice(categories, 100),
            "num": range(100)
        })
        
        # Validation con categoria nuova
        val_data = pd.DataFrame({
            "high_card": ["cat_0", "cat_1", "UNSEEN_CATEGORY", "cat_2", "ANOTHER_UNSEEN"],
            "num": [200, 201, 202, 203, 204]
        })
        
        # Encoding con low cardinality threshold per forzare ordinal
        plan = plan_encodings(train_data, max_ohe_cardinality=5)
        train_encoded, fitted_encoders, _ = fit_apply_encoders(train_data, plan)
        
        # Verifica che sia stato usato ordinal encoding
        assert len(fitted_encoders.ordinal) > 0, "Dovrebbe usare ordinal encoding"
        assert "high_card" in fitted_encoders.ordinal, "high_card dovrebbe essere ordinale"
        
        # Transform su validation
        val_encoded = transform_with_encoders(val_data, fitted_encoders)
        
        # Verifica gestione categorie unseen
        ordinal_col = "high_card__ord"
        assert ordinal_col in val_encoded.columns, "Colonna ordinale dovrebbe esistere"
        
        # Categorie unseen dovrebbero essere gestite (tipicamente con NaN o valore speciale)
        unseen_mask = val_data["high_card"].isin(["UNSEEN_CATEGORY", "ANOTHER_UNSEEN"])
        known_mask = ~unseen_mask
        
        # Categorie conosciute dovrebbero avere valori ordinali validi
        known_values = val_encoded.loc[known_mask, ordinal_col]
        assert not known_values.isnull().all(), "Categorie conosciute dovrebbero avere valori"
        
        # Test che il comportamento sia consistente
        assert len(val_encoded) == len(val_data), "Non dovrebbe perdere righe"
    
    def test_encoding_reproducibility(self):
        """Test che l'encoding sia riproducibile."""
        data = pd.DataFrame({
            "cat": ["A", "B", "C"] * 20,
            "num": range(60)
        })
        
        # Due esecuzioni identiche dovrebbero dare stesso risultato
        plan1 = plan_encodings(data, max_ohe_cardinality=10)
        data1, enc1, _ = fit_apply_encoders(data.copy(), plan1)
        
        plan2 = plan_encodings(data, max_ohe_cardinality=10)
        data2, enc2, _ = fit_apply_encoders(data.copy(), plan2)
        
        # Risultati dovrebbero essere identici
        assert data1.equals(data2), "Encoding dovrebbe essere riproducibile"
        assert set(data1.columns) == set(data2.columns), "Colonne dovrebbero essere identiche"
    
    def test_no_information_leakage_across_splits(self):
        """Test che non ci sia leakage di informazioni tra split."""
        # Crea dati dove train e test hanno distribuzioni diverse
        train_categories = ["A", "B", "C"] 
        test_categories = ["A", "B", "D", "E"]  # D, E sono nuove
        
        train_data = pd.DataFrame({
            "cat": np.random.choice(train_categories, 100),
            "target": np.random.random(100)
        })
        
        test_data = pd.DataFrame({
            "cat": np.random.choice(test_categories, 50), 
            "target": np.random.random(50)
        })
        
        # Encoding fit solo su train
        plan = plan_encodings(train_data.drop("target", axis=1), max_ohe_cardinality=10)
        train_encoded, encoders, _ = fit_apply_encoders(train_data.drop("target", axis=1), plan)
        test_encoded = transform_with_encoders(test_data.drop("target", axis=1), encoders)
        
        # Verifica che l'encoder non abbia "visto" le categorie del test
        if encoders.one_hot is not None:
            feature_names = encoders.one_hot.get_feature_names_out()
            # Non dovrebbe contenere categorie presenti solo nel test (D, E)
            assert not any("cat_D" in f for f in feature_names), "Encoder non dovrebbe conoscere categoria D"
            assert not any("cat_E" in f for f in feature_names), "Encoder non dovrebbe conoscere categoria E"
            
            # Dovrebbe contenere tutte le categorie del train
            assert any("cat_A" in f for f in feature_names), "Dovrebbe contenere categoria A"
            assert any("cat_B" in f for f in feature_names), "Dovrebbe contenere categoria B" 
            assert any("cat_C" in f for f in feature_names), "Dovrebbe contenere categoria C"


if __name__ == "__main__":
    pytest.main([__file__])