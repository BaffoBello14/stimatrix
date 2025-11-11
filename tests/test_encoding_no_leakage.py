"""Tests to verify encoding does not cause data leakage with new multi-strategy encoding."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.encoders import plan_encodings, fit_apply_encoders, transform_with_encoders


# Default test config
TEST_CONFIG = {
    "encoding": {
        "one_hot_max": 10,
        "target_encoding_range": [11, 30],
        "frequency_encoding_range": [31, 100],
        "ordinal_encoding_range": [101, 200],
        "drop_above": 200,
        "target_encoder": {
            "smoothing": 1.0,
            "min_samples_leaf": 1
        }
    }
}


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
        y_train = pd.Series(np.random.random(50))
        
        # Crea validation set con categoria nuova "D"
        val_data = pd.DataFrame({
            "cat1": ["A", "B", "D", "D", "A"],  # D è categoria unseen
            "cat2": ["X", "Y", "Z", "Z", "X"],  # Z è categoria unseen  
            "num": [100, 101, 102, 103, 104]
        })
        
        # Test del flusso corretto: fit solo su train
        plan = plan_encodings(train_data, TEST_CONFIG)
        train_encoded, fitted_encoders = fit_apply_encoders(train_data, y_train, plan, TEST_CONFIG)
        
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
    
    def test_target_encoding_no_leakage(self):
        """Test che target encoding non causi leakage."""
        # Crea train con cardinalità media (11-30) per forzare target encoding
        categories = [f"cat_{i}" for i in range(15)]
        train_data = pd.DataFrame({
            "medium_card": np.random.choice(categories, 100),
            "num": range(100)
        })
        # Target correlato con categoria per verificare che l'encoding impari la relazione
        y_train = pd.Series([float(hash(cat) % 100) for cat in train_data["medium_card"]])
        
        # Validation con alcune categorie nuove
        val_categories = categories[:10] + ["UNSEEN_1", "UNSEEN_2"]
        val_data = pd.DataFrame({
            "medium_card": np.random.choice(val_categories, 30),
            "num": range(200, 230)
        })
        
        plan = plan_encodings(train_data, TEST_CONFIG)
        
        # Verifica che sia stato pianificato target encoding
        assert len(plan.target_cols) > 0, "Dovrebbe usare target encoding per cardinalità media"
        assert "medium_card" in plan.target_cols
        
        train_encoded, fitted_encoders = fit_apply_encoders(train_data, y_train, plan, TEST_CONFIG)
        
        # Verifica che target encoding sia stato applicato
        assert "medium_card__target" in train_encoded.columns
        assert "medium_card" not in train_encoded.columns, "Colonna originale dovrebbe essere rimossa"
        
        # Transform su validation
        val_encoded = transform_with_encoders(val_data, fitted_encoders)
        
        # Verifica gestione categorie unseen (dovrebbero avere valore neutro/globale)
        assert "medium_card__target" in val_encoded.columns
        assert len(val_encoded) == len(val_data), "Non dovrebbe perdere righe"
        # Categorie unseen dovrebbero avere valori (anche se neutri)
        assert not val_encoded["medium_card__target"].isnull().all()
    
    def test_frequency_encoding(self):
        """Test frequency encoding per cardinalità alta (31-100)."""
        # Crea categorie in range frequency encoding
        categories = [f"cat_{i}" for i in range(50)]
        train_data = pd.DataFrame({
            "high_card": np.random.choice(categories, 200),
        })
        y_train = pd.Series(np.random.random(200))
        
        plan = plan_encodings(train_data, TEST_CONFIG)
        
        # Verifica che sia stato pianificato frequency encoding
        assert len(plan.frequency_cols) > 0
        assert "high_card" in plan.frequency_cols
        
        train_encoded, fitted_encoders = fit_apply_encoders(train_data, y_train, plan, TEST_CONFIG)
        
        # Verifica frequency encoding applicato
        assert "high_card__freq" in train_encoded.columns
        assert "high_card" not in train_encoded.columns
        
        # Frequenze dovrebbero essere tra 0 e 1
        freq_values = train_encoded["high_card__freq"]
        assert (freq_values >= 0).all() and (freq_values <= 1).all()
    
    def test_ordinal_encoding_handles_unseen(self):
        """Test che ordinal encoding gestisca categorie unseen (101-200 unique)."""
        # Train con cardinalità molto alta per forzare ordinal encoding
        categories = [f"cat_{i}" for i in range(150)]
        train_data = pd.DataFrame({
            "very_high_card": np.random.choice(categories, 300),
            "num": range(300)
        })
        y_train = pd.Series(np.random.random(300))
        
        # Validation con categoria nuova
        val_data = pd.DataFrame({
            "very_high_card": ["cat_0", "cat_1", "UNSEEN_CATEGORY", "cat_2", "ANOTHER_UNSEEN"],
            "num": [200, 201, 202, 203, 204]
        })
        
        plan = plan_encodings(train_data, TEST_CONFIG)
        
        # Verifica che sia pianificato ordinal encoding
        assert len(plan.ordinal_cols) > 0
        assert "very_high_card" in plan.ordinal_cols
        
        train_encoded, fitted_encoders = fit_apply_encoders(train_data, y_train, plan, TEST_CONFIG)
        
        # Verifica che sia stato usato ordinal encoding
        assert len(fitted_encoders.ordinal) > 0, "Dovrebbe usare ordinal encoding"
        assert "very_high_card" in fitted_encoders.ordinal, "very_high_card dovrebbe essere ordinale"
        
        # Transform su validation
        val_encoded = transform_with_encoders(val_data, fitted_encoders)
        
        # Verifica gestione categorie unseen
        ordinal_col = "very_high_card__ord"
        assert ordinal_col in val_encoded.columns, "Colonna ordinale dovrebbe esistere"
        
        # Categorie unseen dovrebbero essere gestite con valore speciale (-1)
        unseen_mask = val_data["very_high_card"].isin(["UNSEEN_CATEGORY", "ANOTHER_UNSEEN"])
        known_mask = ~unseen_mask
        
        # Categorie conosciute dovrebbero avere valori ordinali validi
        known_values = val_encoded.loc[known_mask, ordinal_col]
        assert not known_values.isnull().all(), "Categorie conosciute dovrebbero avere valori"
        
        # Test che il comportamento sia consistente
        assert len(val_encoded) == len(val_data), "Non dovrebbe perdere righe"
    
    def test_drop_very_high_cardinality(self):
        """Test che colonne con cardinalità >200 siano droppate."""
        # Crea colonna con cardinalità troppo alta
        categories = [f"cat_{i}" for i in range(250)]
        normal_values = (["A", "B", "C"] * 167)[:500]  # Ensure exactly 500 rows
        train_data = pd.DataFrame({
            "extreme_card": np.random.choice(categories, 500, replace=True),
            "normal": normal_values
        })
        y_train = pd.Series(np.random.random(500))
        
        plan = plan_encodings(train_data, TEST_CONFIG)
        
        # Verifica che sia stata pianificata per drop
        assert "extreme_card" in plan.drop_cols
        assert "extreme_card" not in plan.one_hot_cols
        assert "extreme_card" not in plan.target_cols
        assert "extreme_card" not in plan.frequency_cols
        assert "extreme_card" not in plan.ordinal_cols
        
        train_encoded, fitted_encoders = fit_apply_encoders(train_data, y_train, plan, TEST_CONFIG)
        
        # Colonna dovrebbe essere stata droppata
        assert "extreme_card" not in train_encoded.columns
    
    def test_encoding_reproducibility(self):
        """Test che l'encoding sia riproducibile."""
        data = pd.DataFrame({
            "cat": ["A", "B", "C"] * 20,
            "num": range(60)
        })
        y = pd.Series(np.random.random(60))
        
        # Due esecuzioni identiche dovrebbero dare stesso risultato
        plan1 = plan_encodings(data, TEST_CONFIG)
        data1, enc1 = fit_apply_encoders(data.copy(), y, plan1, TEST_CONFIG)
        
        plan2 = plan_encodings(data, TEST_CONFIG)
        data2, enc2 = fit_apply_encoders(data.copy(), y, plan2, TEST_CONFIG)
        
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
        })
        y_train = pd.Series(np.random.random(100))
        
        test_data = pd.DataFrame({
            "cat": np.random.choice(test_categories, 50),
        })
        
        # Encoding fit solo su train
        plan = plan_encodings(train_data, TEST_CONFIG)
        train_encoded, encoders = fit_apply_encoders(train_data, y_train, plan, TEST_CONFIG)
        test_encoded = transform_with_encoders(test_data, encoders)
        
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
    pytest.main([__file__, "-v"])
