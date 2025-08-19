"""
Test suite per integrazione completa pipeline con nuovi moduli.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Aggiungi src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.smart_config import SmartConfigurationManager
from validation.quality_checks import QualityChecker
from preprocessing.pipeline_tracker import PipelineTracker
from utils.temporal_advanced import AdvancedTemporalUtils
from utils.robust_operations import RobustDataOperations


class TestPipelineIntegration:
    """Test integrazione completa pipeline con nuovi moduli."""
    
    @pytest.fixture
    def comprehensive_config(self):
        """Configurazione completa per test."""
        return {
            'paths': {
                'raw_data': 'data/raw',
                'preprocessed_data': 'data/preprocessed',
                'models_dir': 'models'
            },
            'target': {
                'column_candidates': ['AI_Prezzo_Ridistribuito']
            },
            'temporal_split': {
                'year_col': 'A_AnnoStipula',
                'month_col': 'A_MeseStipula',
                'mode': 'fraction',
                'train_fraction': 0.7,
                'valid_fraction': 0.15
            },
            'outliers': {
                'method': 'ensemble',
                'group_by_col': 'AI_IdCategoriaCatastale'
            },
            'quality_checks': {
                'check_temporal_leakage': True,
                'check_target_leakage': True,
                'check_category_distribution': True
            },
            'training': {
                'shap': {'enabled': True, 'sample_size': 100}
            },
            'tracking': {
                'enabled': True,
                'save_intermediate': False
            }
        }
    
    @pytest.fixture
    def realistic_real_estate_data(self):
        """Dataset immobiliare realistico per test."""
        np.random.seed(42)
        n_samples = 500
        
        return pd.DataFrame({
            'A_AnnoStipula': np.random.choice([2020, 2021, 2022, 2023], n_samples),
            'A_MeseStipula': np.random.randint(1, 13, n_samples),
            'AI_IdCategoriaCatastale': np.random.choice(['A02', 'A03', 'A04', 'A10'], n_samples),
            'AI_Superficie': np.random.normal(80, 30, n_samples),
            'AI_Piano': np.random.choice(['P1', 'P2', 'P3', 'PT', 'S1'], n_samples),
            'AI_Prezzo_Ridistribuito': np.random.lognormal(11, 0.5, n_samples),
            'feature_numeric1': np.random.normal(0, 1, n_samples),
            'feature_numeric2': np.random.normal(0, 1, n_samples),
            'feature_categorical': np.random.choice(['Type1', 'Type2', 'Type3'], n_samples),
            'constant_feature': ['CONSTANT'] * n_samples,
            'id_feature': range(n_samples)  # Potenziale data leakage
        })
    
    def test_smart_config_with_real_data(self, comprehensive_config, realistic_real_estate_data):
        """Test SmartConfigurationManager con dati realistici."""
        config_manager = SmartConfigurationManager()
        config_manager.config = comprehensive_config
        
        # Test risoluzione target
        target_resolution = config_manager.resolve_target_columns(realistic_real_estate_data)
        assert target_resolution['target_column'] == 'AI_Prezzo_Ridistribuito'
        assert target_resolution['resolution_method'] == 'direct_match'
        
        # Test risoluzione temporale
        temporal_resolution = config_manager.resolve_temporal_columns(realistic_real_estate_data)
        assert temporal_resolution['temporal_available'] == True
        assert temporal_resolution['year_column'] == 'A_AnnoStipula'
        
        # Test risoluzione categoriche
        categorical_resolution = config_manager.resolve_categorical_columns(realistic_real_estate_data)
        assert len(categorical_resolution['categorical_columns']) > 0
        assert categorical_resolution['category_column_for_outliers'] == 'AI_IdCategoriaCatastale'
    
    def test_quality_checks_with_real_data(self, comprehensive_config, realistic_real_estate_data):
        """Test QualityChecker con dati realistici."""
        checker = QualityChecker(comprehensive_config)
        
        # Prepara split temporali
        sorted_data = AdvancedTemporalUtils.temporal_sort_dataframe(
            realistic_real_estate_data, 'A_AnnoStipula', 'A_MeseStipula'
        )
        
        n = len(sorted_data)
        X_train = sorted_data.iloc[:int(n*0.6)].drop('AI_Prezzo_Ridistribuito', axis=1)
        X_val = sorted_data.iloc[int(n*0.6):int(n*0.8)].drop('AI_Prezzo_Ridistribuito', axis=1)
        X_test = sorted_data.iloc[int(n*0.8):].drop('AI_Prezzo_Ridistribuito', axis=1)
        
        y_train = sorted_data.iloc[:int(n*0.6)]['AI_Prezzo_Ridistribuito']
        y_val = sorted_data.iloc[int(n*0.6):int(n*0.8)]['AI_Prezzo_Ridistribuito']
        y_test = sorted_data.iloc[int(n*0.8):]['AI_Prezzo_Ridistribuito']
        
        # Esegui quality checks
        results = checker.run_all_checks(X_train, X_val, X_test, y_train, y_val, y_test)
        
        assert 'overall_status' in results
        assert len(results['checks_executed']) >= 3
        
        # Dovrebbe rilevare ID feature come sospetta
        if 'target_leakage' in results:
            suspicious_features = results['target_leakage'].get('suspicious_features', [])
            assert any('id' in f['feature'].lower() for f in suspicious_features)
    
    def test_pipeline_tracker_workflow(self, comprehensive_config, realistic_real_estate_data):
        """Test PipelineTracker con workflow completo."""
        tracker = PipelineTracker(comprehensive_config)
        
        # Simula step di preprocessing
        df_initial = realistic_real_estate_data.copy()
        
        # Step 1: Feature extraction
        tracker.track_step_start('feature_extraction')
        df_after_fe = df_initial.copy()
        df_after_fe['new_feature'] = df_after_fe['AI_Superficie'] * 2
        
        fe_info = tracker.track_step_completion(
            'feature_extraction', df_initial, df_after_fe, 
            {'features_added': ['new_feature']}
        )
        assert fe_info['cols_change'] == 1
        
        # Step 2: Outlier removal
        tracker.track_step_start('outlier_removal')
        df_after_outliers = df_after_fe.iloc[:-10].copy()  # Rimuovi 10 outlier
        
        outlier_info = tracker.track_step_completion(
            'outlier_removal', df_after_fe, df_after_outliers,
            {'outliers_removed': 10}
        )
        assert outlier_info['rows_change'] == -10
        
        # Genera report
        report = tracker.generate_comprehensive_report()
        
        assert 'pipeline_summary' in report
        assert 'dataset_evolution' in report
        assert len(report['dataset_evolution']) == 2
        assert report['pipeline_summary']['steps_completed'] == 2
    
    @pytest.mark.slow
    def test_full_pipeline_integration(self, comprehensive_config, realistic_real_estate_data):
        """Test integrazione completa pipeline."""
        # Inizializza componenti
        config_manager = SmartConfigurationManager()
        config_manager.config = comprehensive_config
        
        tracker = PipelineTracker(comprehensive_config)
        checker = QualityChecker(comprehensive_config)
        
        # 1. Ottimizza config per dataset
        optimized_config = config_manager.optimize_config_for_dataset(realistic_real_estate_data)
        assert 'paths' in optimized_config
        
        # 2. Risolvi colonne
        target_resolution = config_manager.resolve_target_columns(realistic_real_estate_data)
        temporal_resolution = config_manager.resolve_temporal_columns(realistic_real_estate_data)
        
        assert target_resolution['target_column'] is not None
        assert temporal_resolution['temporal_available'] == True
        
        # 3. Simula preprocessing con tracking
        tracker.track_step_start('preprocessing')
        
        # Rimozione colonne costanti
        df_cleaned, removal_info = RobustDataOperations.remove_columns_safe(
            realistic_real_estate_data, ['constant_feature']
        )
        
        tracker.track_step_completion(
            'constant_removal', realistic_real_estate_data, df_cleaned, removal_info
        )
        
        # 4. Split temporale
        sorted_data = AdvancedTemporalUtils.temporal_sort_dataframe(
            df_cleaned, 'A_AnnoStipula', 'A_MeseStipula'
        )
        
        n = len(sorted_data)
        X_train = sorted_data.iloc[:int(n*0.6)].drop('AI_Prezzo_Ridistribuito', axis=1)
        X_val = sorted_data.iloc[int(n*0.6):int(n*0.8)].drop('AI_Prezzo_Ridistribuito', axis=1)
        X_test = sorted_data.iloc[int(n*0.8):].drop('AI_Prezzo_Ridistribuito', axis=1)
        
        y_train = sorted_data.iloc[:int(n*0.6)]['AI_Prezzo_Ridistribuito']
        y_val = sorted_data.iloc[int(n*0.6):int(n*0.8)]['AI_Prezzo_Ridistribuito']
        y_test = sorted_data.iloc[int(n*0.8):]['AI_Prezzo_Ridistribuito']
        
        # 5. Quality checks
        quality_results = checker.run_all_checks(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # 6. Genera report finale
        final_report = tracker.generate_comprehensive_report()
        
        # Verifica integrazione
        assert quality_results['overall_status'] in ['PASSED', 'WARNINGS', 'FAILED', 'CRITICAL_ERRORS']
        assert final_report['pipeline_summary']['steps_completed'] >= 1
        assert len(X_train) + len(X_val) + len(X_test) <= len(realistic_real_estate_data)
    
    def test_error_handling_integration(self, comprehensive_config):
        """Test gestione errori integrata."""
        # Dataset problematico
        problematic_data = pd.DataFrame({
            'missing_temporal': [1, 2, 3],  # Mancano colonne temporali
            'target': [100, 200, 300]
        })
        
        config_manager = SmartConfigurationManager()
        config_manager.config = comprehensive_config
        
        # Dovrebbe gestire gracefully colonne temporali mancanti
        temporal_resolution = config_manager.resolve_temporal_columns(problematic_data)
        assert temporal_resolution['temporal_available'] == False
        
        # Quality checker dovrebbe adattarsi
        checker = QualityChecker(comprehensive_config)
        
        X_train = problematic_data.drop('target', axis=1).iloc[:2]
        X_test = problematic_data.drop('target', axis=1).iloc[2:]
        y_train = problematic_data['target'].iloc[:2]
        y_test = problematic_data['target'].iloc[2:]
        
        # Non dovrebbe crashare anche senza colonne temporali
        results = checker.run_all_checks(X_train, None, X_test, y_train, None, y_test)
        assert 'overall_status' in results