"""Tests to verify contextual features do not cause data leakage."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.contextual_features import (
    fit_contextual_features,
    transform_contextual_features,
    fit_transform_contextual_features
)


class TestContextualFeaturesNoLeakage:
    """Test che le contextual features non causino data leakage."""
    
    def test_fit_only_on_train(self):
        """Test che stats siano calcolate SOLO sul training set."""
        # Train con zone A, B
        train = pd.DataFrame({
            'AI_ZonaOmi': ['A', 'A', 'B', 'B'] * 10,
            'AI_Prezzo_Ridistribuito': [100, 150, 200, 250] * 10,
            'AI_Superficie': [50, 75, 100, 125] * 10,
            'AI_IdTipologiaEdilizia': [1, 1, 2, 2] * 10,
            'A_AnnoStipula': [2020, 2020, 2021, 2021] * 10,
            'A_MeseStipula': [1, 2, 3, 4] * 10,
        })
        
        # Test con zona nuova C (non vista in train)
        test = pd.DataFrame({
            'AI_ZonaOmi': ['A', 'C', 'C'],
            'AI_Prezzo_Ridistribuito': [120, 500, 600],  # Prezzi molto diversi
            'AI_Superficie': [60, 200, 250],
            'AI_IdTipologiaEdilizia': [1, 3, 3],
            'A_AnnoStipula': [2022, 2022, 2022],
            'A_MeseStipula': [1, 2, 3],
        })
        
        # Fit solo su train
        stats = fit_contextual_features(train)
        
        # Verifica che stats contengano solo zone A e B (non C)
        assert 'zone_price' in stats
        zone_keys = set(stats['zone_price'].keys())
        assert 'A' in zone_keys
        assert 'B' in zone_keys
        assert 'C' not in zone_keys, "Test zone C should not be in training stats"
        
        # Verifica che le medie siano calcolate solo su train
        zone_A_mean = stats['zone_price']['A']['zone_price_mean']
        expected_A_mean = train[train['AI_ZonaOmi'] == 'A']['AI_Prezzo_Ridistribuito'].mean()
        assert abs(zone_A_mean - expected_A_mean) < 1e-6
    
    def test_no_target_instance_features(self):
        """Test che non vengano create feature che usano il target dell'istanza corrente."""
        train = pd.DataFrame({
            'AI_ZonaOmi': ['A', 'A', 'B', 'B'],
            'AI_Prezzo_Ridistribuito': [100, 200, 150, 250],
            'AI_Superficie': [50, 100, 75, 125],
            'AI_IdTipologiaEdilizia': [1, 1, 2, 2],
            'A_AnnoStipula': [2020, 2020, 2021, 2021],
            'A_MeseStipula': [1, 2, 3, 4],
        })
        
        stats = fit_contextual_features(train)
        transformed = transform_contextual_features(train, stats)
        
        # ❌ PROIBITE: Feature che richiedono il target dell'istanza corrente
        prohibited_features = [
            'price_vs_zone_mean',
            'price_vs_zone_mean_ratio',
            'price_vs_zone_median_ratio',
            'price_zone_zscore',
            'price_zone_iqr_position',
            'price_zone_range_position',
            'price_vs_type_zone_mean',
            'prezzo_mq',  # Richiede prezzo corrente!
            'prezzo_mq_vs_zone',
            'price_vs_temporal_mean',
        ]
        
        for feature in prohibited_features:
            assert feature not in transformed.columns, \
                f"❌ LEAKAGE: Feature '{feature}' requires current instance target and should NOT exist!"
        
        # ✅ PERMESSE: Feature aggregate che non usano target corrente
        allowed_features = [
            'zone_price_mean',  # Media della zona (da train)
            'zone_price_median',
            'zone_count',
            'type_zone_rarity',  # Basata su count, non su target
            'surface_vs_zone_mean',  # Ratio di superfici, non prezzi
        ]
        
        for feature in allowed_features:
            if feature in transformed.columns:
                # Ok, feature permessa presente
                pass
    
    def test_transform_with_unseen_categories(self):
        """Test che transform gestisca correttamente categorie non viste in train."""
        train = pd.DataFrame({
            'AI_ZonaOmi': ['A', 'A', 'B', 'B'],
            'AI_Prezzo_Ridistribuito': [100, 200, 150, 250],
            'AI_Superficie': [50, 100, 75, 125],
            'AI_IdTipologiaEdilizia': [1, 1, 2, 2],
            'A_AnnoStipula': [2020, 2020, 2021, 2021],
            'A_MeseStipula': [1, 2, 3, 4],
        })
        
        # Test con categoria completamente nuova
        test = pd.DataFrame({
            'AI_ZonaOmi': ['C', 'C'],  # Zona mai vista
            'AI_Prezzo_Ridistribuito': [300, 400],
            'AI_Superficie': [150, 200],
            'AI_IdTipologiaEdilizia': [3, 3],
            'A_AnnoStipula': [2022, 2022],
            'A_MeseStipula': [1, 2],
        })
        
        stats = fit_contextual_features(train)
        train_transformed = transform_contextual_features(train, stats)
        test_transformed = transform_contextual_features(test, stats)
        
        # Verifica che test abbia le stesse colonne di train
        train_cols = set(train_transformed.columns)
        test_cols = set(test_transformed.columns)
        assert train_cols == test_cols, "Train and test should have same columns after transform"
        
        # Verifica che valori per categoria unseen siano gestiti (NaN o default)
        # Le zone unseen avranno NaN per le statistiche di zona (comportamento corretto)
        assert len(test_transformed) == len(test), "Should not lose rows"
    
    def test_fit_transform_consistency(self):
        """Test che fit_transform_contextual_features sia coerente."""
        train = pd.DataFrame({
            'AI_ZonaOmi': ['A', 'A', 'B', 'B'],
            'AI_Prezzo_Ridistribuito': [100, 200, 150, 250],
            'AI_Superficie': [50, 100, 75, 125],
            'AI_IdTipologiaEdilizia': [1, 1, 2, 2],
            'A_AnnoStipula': [2020, 2020, 2021, 2021],
            'A_MeseStipula': [1, 2, 3, 4],
        })
        
        test = pd.DataFrame({
            'AI_ZonaOmi': ['A', 'B'],
            'AI_Prezzo_Ridistribuito': [120, 180],
            'AI_Superficie': [60, 90],
            'AI_IdTipologiaEdilizia': [1, 2],
            'A_AnnoStipula': [2022, 2022],
            'A_MeseStipula': [1, 2],
        })
        
        # Metodo 1: fit + transform separati
        stats = fit_contextual_features(train)
        train_v1 = transform_contextual_features(train, stats)
        test_v1 = transform_contextual_features(test, stats)
        
        # Metodo 2: fit_transform insieme
        train_v2, _, test_v2, stats2 = fit_transform_contextual_features(
            train_df=train,
            test_df=test
        )
        
        # Verifica che i risultati siano identici
        pd.testing.assert_frame_equal(
            train_v1.sort_index(axis=1), 
            train_v2.sort_index(axis=1),
            check_dtype=False
        )
        pd.testing.assert_frame_equal(
            test_v1.sort_index(axis=1),
            test_v2.sort_index(axis=1),
            check_dtype=False
        )
    
    def test_temporal_features_no_future_leakage(self):
        """Test che temporal features non usino dati dal futuro."""
        # Train con dati 2020-2021
        train = pd.DataFrame({
            'AI_ZonaOmi': ['A'] * 12,
            'AI_Prezzo_Ridistribuito': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210],
            'AI_Superficie': [50] * 12,
            'AI_IdTipologiaEdilizia': [1] * 12,
            'A_AnnoStipula': [2020] * 6 + [2021] * 6,
            'A_MeseStipula': list(range(1, 7)) * 2,
        })
        
        # Test con dati 2022 (futuro)
        test = pd.DataFrame({
            'AI_ZonaOmi': ['A'] * 3,
            'AI_Prezzo_Ridistribuito': [220, 230, 240],
            'AI_Superficie': [50] * 3,
            'AI_IdTipologiaEdilizia': [1] * 3,
            'A_AnnoStipula': [2022] * 3,
            'A_MeseStipula': [1, 2, 3],
        })
        
        stats = fit_contextual_features(train)
        
        # Verifica che temporal stats contengano solo periodi del train
        if 'temporal_price' in stats:
            temporal_keys = set(stats['temporal_price'].keys())
            # Max temporal key nel train
            max_train_key = train['A_AnnoStipula'].max() * 100 + train['A_MeseStipula'].max()
            
            # Nessun temporal key dovrebbe essere dal futuro (test)
            for key in temporal_keys:
                assert key <= max_train_key, \
                    f"Temporal key {key} is from future (beyond train max {max_train_key})"
    
    def test_reproducibility(self):
        """Test che le contextual features siano riproducibili."""
        train = pd.DataFrame({
            'AI_ZonaOmi': ['A', 'A', 'B', 'B'],
            'AI_Prezzo_Ridistribuito': [100, 200, 150, 250],
            'AI_Superficie': [50, 100, 75, 125],
            'AI_IdTipologiaEdilizia': [1, 1, 2, 2],
            'A_AnnoStipula': [2020, 2020, 2021, 2021],
            'A_MeseStipula': [1, 2, 3, 4],
        })
        
        # Due esecuzioni identiche
        stats1 = fit_contextual_features(train)
        result1 = transform_contextual_features(train, stats1)
        
        stats2 = fit_contextual_features(train)
        result2 = transform_contextual_features(train, stats2)
        
        # Risultati dovrebbero essere identici
        pd.testing.assert_frame_equal(
            result1.sort_index(axis=1),
            result2.sort_index(axis=1),
            check_dtype=False
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
