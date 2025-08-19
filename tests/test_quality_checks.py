"""
Test suite per Quality Checks system.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Aggiungi src al path per import
sys.path.append(str(Path(__file__).parent.parent / "src"))

from validation.quality_checks import QualityChecker, DataQualityMetrics


class TestQualityChecker:
    """Test per QualityChecker class."""
    
    @pytest.fixture
    def sample_config(self):
        """Configurazione di test."""
        return {
            'quality_checks': {
                'check_temporal_leakage': True,
                'check_target_leakage': True,
                'check_category_distribution': True,
                'max_category_drift': 0.05,
                'min_temporal_gap_months': 1
            }
        }
    
    @pytest.fixture
    def sample_temporal_data(self):
        """Dataset temporale di test."""
        np.random.seed(42)
        n_samples = 1000
        
        # Crea dati temporali ordinati
        years = np.random.choice([2020, 2021, 2022, 2023], n_samples)
        months = np.random.choice(range(1, 13), n_samples)
        
        return pd.DataFrame({
            'A_AnnoStipula': years,
            'A_MeseStipula': months,
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.normal(100000, 20000, n_samples)
        })
    
    def test_quality_checker_initialization(self, sample_config):
        """Test inizializzazione QualityChecker."""
        checker = QualityChecker(sample_config)
        
        assert checker.enabled_checks['temporal_leakage'] == True
        assert checker.enabled_checks['target_leakage'] == True
        assert checker.max_category_drift == 0.05
    
    def test_temporal_leakage_detection_valid_split(self, sample_config, sample_temporal_data):
        """Test detection temporal leakage con split valido."""
        checker = QualityChecker(sample_config)
        
        # Crea split temporali validi (no overlap)
        data_sorted = sample_temporal_data.sort_values(['A_AnnoStipula', 'A_MeseStipula'])
        
        n = len(data_sorted)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        X_train = data_sorted.iloc[:train_end].drop('target', axis=1)
        X_val = data_sorted.iloc[train_end:val_end].drop('target', axis=1)
        X_test = data_sorted.iloc[val_end:].drop('target', axis=1)
        
        result = checker.check_temporal_leakage(X_train, X_val, X_test)
        
        assert result['is_valid'] == True
        assert len(result['errors']) == 0
    
    def test_target_leakage_detection(self, sample_config):
        """Test detection target leakage."""
        checker = QualityChecker(sample_config)
        
        # Dataset con feature sospetta
        df_with_leakage = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'AI_Prezzo_Originale': [100, 200, 300, 400],  # Sospetta!
            'category': ['A', 'B', 'A', 'B']
        })
        
        result = checker.check_target_leakage(df_with_leakage, 'AI_Prezzo_Ridistribuito')
        
        assert len(result['suspicious_features']) > 0
        assert any('prezzo' in f['pattern_matched'].lower() for f in result['suspicious_features'])
    
    def test_category_distribution_check(self, sample_config):
        """Test check distribuzione categorie."""
        checker = QualityChecker(sample_config)
        
        # Crea dataset con drift categorico
        np.random.seed(42)
        
        # Train: prevalentemente categoria A
        X_train = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C'], 500, p=[0.7, 0.2, 0.1]),
            'feature1': np.random.normal(0, 1, 500)
        })
        
        # Test: prevalentemente categoria B (drift!)
        X_test = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C'], 200, p=[0.2, 0.7, 0.1]),
            'feature1': np.random.normal(0, 1, 200)
        })
        
        result = checker.check_category_distribution(X_train, None, X_test)
        
        # Dovrebbe rilevare drift
        assert 'category' in result['drift_analysis']
        drift_score = result['drift_analysis']['category']['max_drift']
        assert drift_score > checker.max_category_drift
    
    def test_run_all_checks(self, sample_config, sample_temporal_data):
        """Test esecuzione completa tutti i checks."""
        checker = QualityChecker(sample_config)
        
        # Prepara dati
        data_sorted = sample_temporal_data.sort_values(['A_AnnoStipula', 'A_MeseStipula'])
        n = len(data_sorted)
        
        X_train = data_sorted.iloc[:int(n*0.6)].drop('target', axis=1)
        X_val = data_sorted.iloc[int(n*0.6):int(n*0.8)].drop('target', axis=1)
        X_test = data_sorted.iloc[int(n*0.8):].drop('target', axis=1)
        
        y_train = data_sorted.iloc[:int(n*0.6)]['target']
        y_val = data_sorted.iloc[int(n*0.6):int(n*0.8)]['target']
        y_test = data_sorted.iloc[int(n*0.8):]['target']
        
        result = checker.run_all_checks(X_train, X_val, X_test, y_train, y_val, y_test)
        
        assert 'overall_status' in result
        assert 'checks_executed' in result
        assert len(result['checks_executed']) > 0


class TestDataQualityMetrics:
    """Test per DataQualityMetrics."""
    
    @pytest.fixture
    def sample_quality_data(self):
        """Dataset per test qualità."""
        np.random.seed(42)
        return pd.DataFrame({
            'complete_numeric': np.random.normal(0, 1, 100),
            'incomplete_numeric': [np.nan if i % 10 == 0 else np.random.normal(0, 1) for i in range(100)],
            'consistent_category': np.random.choice(['A', 'B', 'C'], 100),
            'mixed_types': [str(i) if i % 2 == 0 else i for i in range(100)],
            'unique_id': range(100),
            'constant_col': ['constant'] * 100
        })
    
    def test_data_quality_score_calculation(self, sample_quality_data):
        """Test calcolo score qualità dati."""
        metrics = DataQualityMetrics.calculate_data_quality_score(sample_quality_data)
        
        assert 'completeness_score' in metrics
        assert 'consistency_score' in metrics
        assert 'uniqueness_score' in metrics
        assert 'validity_score' in metrics
        assert 'overall_score' in metrics
        
        # Completeness dovrebbe essere < 1.0 per colonna incompleta
        assert 0 < metrics['completeness_score'] < 1.0
        
        # Overall score dovrebbe essere ragionevole
        assert 0 <= metrics['overall_score'] <= 1.0
    
    def test_quality_metrics_with_target(self, sample_quality_data):
        """Test metriche qualità con target specificato."""
        sample_quality_data['target'] = np.random.normal(0, 1, 100)
        
        metrics = DataQualityMetrics.calculate_data_quality_score(
            sample_quality_data, target_col='target'
        )
        
        # Target dovrebbe essere escluso da uniqueness
        assert metrics['overall_score'] >= 0
        assert 'details' in metrics


@pytest.mark.integration
class TestQualityChecksIntegration:
    """Test integrazione quality checks con pipeline."""
    
    @pytest.fixture
    def mock_preprocessing_info(self):
        """Info preprocessing mock."""
        return {
            'dataset_evolution': [
                {
                    'step_name': 'feature_extraction',
                    'shape_before': (1000, 50),
                    'shape_after': (1000, 55),
                    'cols_change': 5,
                    'memory_change_mb': 10.5
                },
                {
                    'step_name': 'outlier_removal',
                    'shape_before': (1000, 55),
                    'shape_after': (950, 55),
                    'cols_change': 0,
                    'memory_change_mb': -2.1
                }
            ],
            'steps_info': {
                'feature_extraction': {
                    'features_added': ['geo_x', 'geo_y', 'area_m2'],
                    'extraction_method': 'wkt_parsing'
                }
            }
        }
    
    def test_feature_stability_with_preprocessing_info(self, sample_config, mock_preprocessing_info):
        """Test feature stability con info preprocessing."""
        checker = QualityChecker(sample_config)
        
        result = checker.check_feature_stability(mock_preprocessing_info)
        
        assert 'stability_analysis' in result
        assert result['is_valid'] in [True, False]  # Dipende dai dati
    
    @pytest.mark.slow
    def test_comprehensive_quality_check_pipeline(self, sample_config, sample_temporal_data):
        """Test completo pipeline quality checks."""
        checker = QualityChecker(sample_config)
        
        # Simula preprocessing completo
        data_sorted = sample_temporal_data.sort_values(['A_AnnoStipula', 'A_MeseStipula'])
        n = len(data_sorted)
        
        # Split con potenziale problema
        X_train = data_sorted.iloc[:int(n*0.8)].drop('target', axis=1)  # 80% train
        X_test = data_sorted.iloc[int(n*0.8):].drop('target', axis=1)   # 20% test
        
        y_train = data_sorted.iloc[:int(n*0.8)]['target']
        y_test = data_sorted.iloc[int(n*0.8):]['target']
        
        # Mock preprocessing info
        preprocessing_info = {
            'dataset_evolution': [
                {
                    'step_name': 'test_step',
                    'shape_before': (n, 6),
                    'shape_after': (int(n*0.95), 5),
                    'cols_change': -1,
                    'memory_change_mb': -5.0
                }
            ]
        }
        
        # Esegui tutti i checks
        result = checker.run_all_checks(
            X_train, None, X_test, y_train, None, y_test, preprocessing_info
        )
        
        assert 'overall_status' in result
        assert result['overall_status'] in ['PASSED', 'WARNINGS', 'FAILED', 'CRITICAL_ERRORS']
        assert len(result['checks_executed']) >= 3