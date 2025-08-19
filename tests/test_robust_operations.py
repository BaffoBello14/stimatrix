"""
Test suite per Robust Operations.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Aggiungi src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.robust_operations import RobustDataOperations, RobustColumnAnalyzer, RobustDataValidator


class TestRobustDataOperations:
    """Test per RobustDataOperations."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """DataFrame di test."""
        return pd.DataFrame({
            'existing_col1': [1, 2, 3, 4, 5],
            'existing_col2': ['A', 'B', 'C', 'D', 'E'],
            'constant_col': ['constant'] * 5,
            'target': [100, 200, 300, 400, 500]
        })
    
    def test_safe_column_operation_success(self, sample_dataframe):
        """Test operazione sicura con successo."""
        def test_operation(df, cols):
            return df.drop(columns=cols), {'dropped': cols}
        
        columns_to_remove = ['existing_col1', 'existing_col2']
        result_df, info = RobustDataOperations.safe_column_operation(
            sample_dataframe, columns_to_remove, "TEST_REMOVAL", test_operation
        )
        
        assert info['success'] == True
        assert info['columns_processed_count'] == 2
        assert 'existing_col1' not in result_df.columns
        assert 'existing_col2' not in result_df.columns
        assert 'target' in result_df.columns
    
    def test_safe_column_operation_missing_columns(self, sample_dataframe):
        """Test operazione con colonne mancanti."""
        def test_operation(df, cols):
            return df.drop(columns=cols), {'dropped': cols}
        
        columns_to_remove = ['existing_col1', 'missing_col1', 'missing_col2']
        result_df, info = RobustDataOperations.safe_column_operation(
            sample_dataframe, columns_to_remove, "TEST_REMOVAL", test_operation
        )
        
        assert info['success'] == True
        assert len(info['missing_columns']) == 2
        assert len(info['existing_columns']) == 1
        assert 'existing_col1' not in result_df.columns
    
    def test_safe_column_operation_with_fallback(self, sample_dataframe):
        """Test operazione con fallback."""
        def failing_operation(df, cols):
            raise ValueError("Operazione fallita")
        
        def fallback_operation(df, cols):
            return df, {'fallback_used': True}
        
        columns = ['existing_col1']
        result_df, info = RobustDataOperations.safe_column_operation(
            sample_dataframe, columns, "TEST_WITH_FALLBACK", 
            failing_operation, fallback_operation
        )
        
        assert info['success'] == True
        assert info['fallback_used'] == True
        assert 'original_error' in info
    
    def test_remove_columns_safe(self, sample_dataframe):
        """Test rimozione sicura colonne."""
        columns_to_remove = ['existing_col1', 'missing_col']
        result_df, info = RobustDataOperations.remove_columns_safe(
            sample_dataframe, columns_to_remove
        )
        
        assert info['success'] == True
        assert 'existing_col1' not in result_df.columns
        assert len(info['missing_columns']) == 1
    
    def test_select_columns_safe(self, sample_dataframe):
        """Test selezione sicura colonne."""
        columns_to_select = ['existing_col1', 'target', 'missing_col']
        result_df, info = RobustDataOperations.select_columns_safe(
            sample_dataframe, columns_to_select
        )
        
        assert info['success'] == True
        assert len(result_df.columns) == 2  # Solo existing_col1 e target
        assert 'existing_col1' in result_df.columns
        assert 'target' in result_df.columns


class TestRobustColumnAnalyzer:
    """Test per RobustColumnAnalyzer."""
    
    @pytest.fixture
    def sample_analysis_data(self):
        """Dataset per analisi colonne."""
        np.random.seed(42)
        return pd.DataFrame({
            'constant_95': ['A'] * 95 + ['B'] * 5,
            'constant_99': ['X'] * 99 + ['Y'] * 1,
            'variable': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'numeric1': np.random.normal(0, 1, 100),
            'numeric2': np.random.normal(0, 1, 100),
            'correlated': np.random.normal(0, 1, 100),
            'target': np.random.normal(0, 1, 100)
        })
    
    def test_find_constant_columns(self, sample_analysis_data):
        """Test identificazione colonne costanti."""
        # Crea correlazione alta
        sample_analysis_data['correlated'] = sample_analysis_data['numeric1'] * 0.99 + np.random.normal(0, 0.01, 100)
        
        constant_cols, stats = RobustColumnAnalyzer.find_constant_columns(
            sample_analysis_data, threshold=0.95, exclude_columns=['target']
        )
        
        assert 'constant_99' in constant_cols
        assert 'target' not in constant_cols  # Escluso
        assert len(stats) > 0
        
        # Verifica statistiche
        for col in constant_cols:
            assert stats[col]['max_frequency'] >= 0.95
    
    def test_find_highly_correlated_columns(self, sample_analysis_data):
        """Test identificazione colonne correlate."""
        # Crea correlazione perfetta
        sample_analysis_data['perfect_corr'] = sample_analysis_data['numeric1']
        
        corr_cols, stats = RobustColumnAnalyzer.find_highly_correlated_columns(
            sample_analysis_data, threshold=0.95, exclude_columns=['target']
        )
        
        assert 'perfect_corr' in corr_cols
        assert 'target' not in corr_cols
    
    def test_find_columns_by_pattern(self, sample_analysis_data):
        """Test ricerca colonne per pattern."""
        patterns = ['*const*', 'numeric*', 'target']
        
        results = RobustColumnAnalyzer.find_columns_by_pattern(
            sample_analysis_data, patterns, case_sensitive=False
        )
        
        assert len(results['*const*']) >= 2  # constant_95, constant_99
        assert len(results['numeric*']) >= 2  # numeric1, numeric2
        assert 'target' in results['target']
    
    def test_analyze_missing_values(self, sample_analysis_data):
        """Test analisi valori mancanti."""
        # Aggiungi valori mancanti
        sample_analysis_data.loc[:50, 'numeric1'] = np.nan  # 51% missing
        sample_analysis_data.loc[:90, 'variable'] = np.nan  # 91% missing
        
        cols_to_remove, stats = RobustColumnAnalyzer.analyze_missing_values(
            sample_analysis_data, threshold=0.9
        )
        
        assert 'variable' in cols_to_remove  # 91% missing > 90%
        assert 'numeric1' not in cols_to_remove  # 51% missing < 90%
        
        # Verifica statistiche
        assert stats['variable']['null_percentage'] > 90
        assert stats['numeric1']['null_percentage'] > 50


class TestRobustDataValidator:
    """Test per RobustDataValidator."""
    
    def test_validate_dataframe_healthy(self):
        """Test validazione DataFrame sano."""
        healthy_df = pd.DataFrame({
            'col1': [1, 2, 3, 4],
            'col2': ['A', 'B', 'C', 'D']
        })
        
        result = RobustDataValidator.validate_dataframe(healthy_df)
        
        assert result['is_valid'] == True
        assert len(result['errors']) == 0
        assert result['shape'] == (4, 2)
    
    def test_validate_dataframe_empty(self):
        """Test validazione DataFrame vuoto."""
        empty_df = pd.DataFrame()
        
        result = RobustDataValidator.validate_dataframe(empty_df)
        
        assert result['is_valid'] == False
        assert len(result['errors']) > 0
        assert 'DataFrame vuoto' in result['errors']
    
    def test_validate_dataframe_with_issues(self):
        """Test validazione DataFrame con problemi."""
        problematic_df = pd.DataFrame({
            'mixed_types': [1, 'string', 3.14, None],
            'col1': [1, 2, 3, 4]
        })
        # Aggiungi colonna duplicata
        problematic_df['col1_duplicate'] = problematic_df['col1']
        problematic_df.columns = ['mixed_types', 'col1', 'col1']  # Nome duplicato
        
        result = RobustDataValidator.validate_dataframe(problematic_df)
        
        assert len(result['warnings']) > 0
    
    def test_validate_columns_exist(self):
        """Test validazione esistenza colonne."""
        df = pd.DataFrame({
            'existing1': [1, 2, 3],
            'existing2': [4, 5, 6]
        })
        
        required_cols = ['existing1', 'existing2', 'missing1']
        result = RobustDataValidator.validate_columns_exist(df, required_cols)
        
        assert result['is_valid'] == False
        assert len(result['existing_columns']) == 2
        assert len(result['missing_columns']) == 1
        assert result['success_rate'] == 2/3


@pytest.mark.parametrize("threshold,expected_constant", [
    (0.9, ['constant_95', 'constant_99']),
    (0.95, ['constant_99']),
    (0.99, ['constant_99']),
    (0.999, [])
])
def test_constant_detection_thresholds(threshold, expected_constant):
    """Test detection colonne costanti con soglie diverse."""
    df = pd.DataFrame({
        'constant_95': ['A'] * 95 + ['B'] * 5,
        'constant_99': ['X'] * 99 + ['Y'] * 1,
        'variable': np.random.choice(['A', 'B', 'C'], 100),
        'target': range(100)
    })
    
    constant_cols, _ = RobustColumnAnalyzer.find_constant_columns(
        df, threshold=threshold, exclude_columns=['target']
    )
    
    for expected_col in expected_constant:
        assert expected_col in constant_cols