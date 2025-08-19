"""
Test suite per Temporal Advanced utilities.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Aggiungi src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.temporal_advanced import AdvancedTemporalUtils, TemporalSplitter


class TestAdvancedTemporalUtils:
    """Test per AdvancedTemporalUtils."""
    
    @pytest.fixture
    def sample_temporal_data(self):
        """Dataset temporale di test."""
        np.random.seed(42)
        return pd.DataFrame({
            'A_AnnoStipula': [2020, 2020, 2021, 2021, 2022, 2022],
            'A_MeseStipula': [1, 6, 3, 9, 2, 12],
            'giorno': [15, 20, 10, 5, 25, 30],
            'feature1': np.random.normal(0, 1, 6),
            'target': np.random.normal(100, 20, 6)
        })
    
    def test_create_temporal_sort_key_basic(self, sample_temporal_data):
        """Test creazione chiave sorting temporale base."""
        sort_key = AdvancedTemporalUtils.create_temporal_sort_key(
            sample_temporal_data, 'A_AnnoStipula', 'A_MeseStipula'
        )
        
        assert len(sort_key) == 6
        assert sort_key.dtype in ['int64', 'float64']
        
        # Verifica ordine: 2020/1 < 2020/6 < 2021/3 < 2021/9 < 2022/2 < 2022/12
        expected_order = [2020*12+1, 2020*12+6, 2021*12+3, 2021*12+9, 2022*12+2, 2022*12+12]
        actual_order = sort_key.tolist()
        assert actual_order == expected_order
    
    def test_create_temporal_sort_key_with_day(self, sample_temporal_data):
        """Test chiave sorting con giorno."""
        sort_key = AdvancedTemporalUtils.create_temporal_sort_key(
            sample_temporal_data, 'A_AnnoStipula', 'A_MeseStipula', 'giorno'
        )
        
        # Con giorno dovrebbe avere più granularità
        assert len(sort_key.unique()) == 6  # Tutti diversi
    
    def test_temporal_sort_dataframe(self, sample_temporal_data):
        """Test ordinamento temporale DataFrame."""
        # Mescola i dati
        shuffled_data = sample_temporal_data.sample(frac=1, random_state=123).reset_index(drop=True)
        
        sorted_df = AdvancedTemporalUtils.temporal_sort_dataframe(
            shuffled_data, 'A_AnnoStipula', 'A_MeseStipula'
        )
        
        # Verifica ordinamento
        for i in range(len(sorted_df) - 1):
            current_key = sorted_df.iloc[i]['A_AnnoStipula'] * 12 + sorted_df.iloc[i]['A_MeseStipula']
            next_key = sorted_df.iloc[i+1]['A_AnnoStipula'] * 12 + sorted_df.iloc[i+1]['A_MeseStipula']
            assert current_key <= next_key
    
    def test_validate_temporal_split_integrity_valid(self, sample_temporal_data):
        """Test validazione integrità split valido."""
        # Ordina dati
        sorted_data = AdvancedTemporalUtils.temporal_sort_dataframe(
            sample_temporal_data, 'A_AnnoStipula', 'A_MeseStipula'
        )
        
        # Split temporale valido
        X_train = sorted_data.iloc[:2].drop('target', axis=1)
        X_val = sorted_data.iloc[2:4].drop('target', axis=1)
        X_test = sorted_data.iloc[4:].drop('target', axis=1)
        
        result = AdvancedTemporalUtils.validate_temporal_split_integrity(
            X_train, X_val, X_test, 'A_AnnoStipula', 'A_MeseStipula'
        )
        
        assert result['is_valid'] == True
        assert len(result['errors']) == 0
        assert 'temporal_ranges' in result
    
    def test_validate_temporal_split_integrity_overlap(self, sample_temporal_data):
        """Test validazione con overlap temporale."""
        # Crea overlap intenzionale
        X_train = sample_temporal_data.iloc[:4].drop('target', axis=1)  # Include 2022 data
        X_val = sample_temporal_data.iloc[2:4].drop('target', axis=1)   # Overlap con train
        X_test = sample_temporal_data.iloc[4:].drop('target', axis=1)
        
        result = AdvancedTemporalUtils.validate_temporal_split_integrity(
            X_train, X_val, X_test, 'A_AnnoStipula', 'A_MeseStipula'
        )
        
        # Dovrebbe rilevare overlap
        assert result['is_valid'] == False or len(result['warnings']) > 0
    
    def test_detect_temporal_anomalies(self, sample_temporal_data):
        """Test rilevamento anomalie temporali."""
        # Aggiungi gap temporale (manca 2021/6)
        anomaly_data = sample_temporal_data[
            ~((sample_temporal_data['A_AnnoStipula'] == 2021) & 
              (sample_temporal_data['A_MeseStipula'] == 6))
        ].copy()
        
        anomalies = AdvancedTemporalUtils.detect_temporal_anomalies(
            anomaly_data, 'A_AnnoStipula', 'A_MeseStipula', 'target'
        )
        
        assert 'missing_periods' in anomalies
        assert 'outlier_periods' in anomalies
        assert 'target_anomalies' in anomalies
    
    def test_create_temporal_features(self, sample_temporal_data):
        """Test creazione feature temporali."""
        enhanced_df = AdvancedTemporalUtils.create_temporal_features(
            sample_temporal_data, 'A_AnnoStipula', 'A_MeseStipula', 'giorno'
        )
        
        # Verifica nuove feature
        temporal_features = [col for col in enhanced_df.columns if col.startswith('temporal_')]
        assert len(temporal_features) > 5
        
        # Verifica feature specifiche
        assert 'temporal_quarter' in enhanced_df.columns
        assert 'temporal_season' in enhanced_df.columns
        assert 'temporal_month_sin' in enhanced_df.columns
        assert 'temporal_month_cos' in enhanced_df.columns
        
        # Verifica valori ragionevoli
        assert enhanced_df['temporal_quarter'].min() >= 1
        assert enhanced_df['temporal_quarter'].max() <= 4


class TestTemporalSplitter:
    """Test per TemporalSplitter."""
    
    @pytest.fixture
    def large_temporal_data(self):
        """Dataset temporale più grande per test split."""
        np.random.seed(42)
        n_samples = 1000
        
        # Distribuisci su 3 anni
        years = np.random.choice([2020, 2021, 2022], n_samples)
        months = np.random.randint(1, 13, n_samples)
        
        return pd.DataFrame({
            'A_AnnoStipula': years,
            'A_MeseStipula': months,
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.normal(100000, 20000, n_samples)
        })
    
    def test_split_temporal_with_validation(self, large_temporal_data):
        """Test split temporale con validazione."""
        train_df, val_df, test_df, split_info = TemporalSplitter.split_temporal_with_validation(
            large_temporal_data,
            'A_AnnoStipula', 'A_MeseStipula',
            train_fraction=0.6, val_fraction=0.2, test_fraction=0.2
        )
        
        # Verifica dimensioni
        total_samples = len(large_temporal_data)
        assert len(train_df) == int(total_samples * 0.6)
        assert len(val_df) == int(total_samples * 0.2)
        assert len(test_df) == total_samples - len(train_df) - len(val_df)
        
        # Verifica info split
        assert 'splits_info' in split_info
        assert 'validation_results' in split_info
        assert split_info['validation_results']['is_valid'] == True
    
    def test_split_temporal_invalid_fractions(self, large_temporal_data):
        """Test split con frazioni invalide."""
        with pytest.raises(ValueError, match="Frazioni non sommano a 1.0"):
            TemporalSplitter.split_temporal_with_validation(
                large_temporal_data,
                'A_AnnoStipula', 'A_MeseStipula',
                train_fraction=0.6, val_fraction=0.3, test_fraction=0.3  # Somma > 1
            )
    
    def test_split_temporal_missing_columns(self, large_temporal_data):
        """Test split con colonne temporali mancanti."""
        with pytest.raises(ValueError, match="Colonne temporali mancanti"):
            AdvancedTemporalUtils.create_temporal_sort_key(
                large_temporal_data, 'missing_year', 'missing_month'
            )


@pytest.mark.integration
class TestTemporalIntegration:
    """Test integrazione utilities temporali."""
    
    def test_full_temporal_workflow(self):
        """Test workflow temporale completo."""
        # Crea dataset realistico
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        n_samples = len(dates)
        
        df = pd.DataFrame({
            'A_AnnoStipula': dates.year,
            'A_MeseStipula': dates.month,
            'giorno': dates.day,
            'feature1': np.random.normal(0, 1, n_samples),
            'target': np.random.normal(100000, 20000, n_samples)
        })
        
        # 1. Ordina temporalmente
        sorted_df = AdvancedTemporalUtils.temporal_sort_dataframe(
            df, 'A_AnnoStipula', 'A_MeseStipula', 'giorno'
        )
        
        # 2. Crea feature temporali
        enhanced_df = AdvancedTemporalUtils.create_temporal_features(
            sorted_df, 'A_AnnoStipula', 'A_MeseStipula', 'giorno'
        )
        
        # 3. Split temporale
        train_df, val_df, test_df, split_info = TemporalSplitter.split_temporal_with_validation(
            enhanced_df, 'A_AnnoStipula', 'A_MeseStipula'
        )
        
        # 4. Valida integrità
        X_train = train_df.drop('target', axis=1)
        X_val = val_df.drop('target', axis=1)
        X_test = test_df.drop('target', axis=1)
        
        integrity_result = AdvancedTemporalUtils.validate_temporal_split_integrity(
            X_train, X_val, X_test, 'A_AnnoStipula', 'A_MeseStipula'
        )
        
        # Verifica risultati
        assert split_info['validation_results']['is_valid'] == True
        assert integrity_result['is_valid'] == True
        assert len(enhanced_df.columns) > len(df.columns)  # Feature aggiunte