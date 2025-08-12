"""Tests for preprocessing pipeline functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add src to path for testing
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.pipeline import (
    choose_target,
    apply_log_target_if,
    run_preprocessing
)
from preprocessing.transformers import temporal_split, temporal_key
from preprocessing.outliers import detect_outliers, OutlierConfig
from preprocessing.imputation import impute_missing, ImputationConfig


class TestTargetSelection:
    """Test target variable selection logic."""
    
    def test_choose_target_first_candidate(self, sample_real_estate_data):
        """Test target selection with first candidate available."""
        config = {"target": {"column_candidates": ["AI_Prezzo_Ridistribuito", "Another_Target"]}}
        
        target_col = choose_target(sample_real_estate_data, config)
        
        assert target_col == "AI_Prezzo_Ridistribuito"
    
    def test_choose_target_second_candidate(self):
        """Test target selection with second candidate."""
        df = pd.DataFrame({
            "AI_Prezzo_Alternative": [100, 200, 300],
            "Other_Col": [1, 2, 3]
        })
        config = {"target": {"column_candidates": ["AI_Prezzo_Ridistribuito", "AI_Prezzo_Alternative"]}}
        
        target_col = choose_target(df, config)
        
        assert target_col == "AI_Prezzo_Alternative"
    
    def test_choose_target_not_found(self):
        """Test target selection when no candidates are found."""
        df = pd.DataFrame({
            "Some_Col": [1, 2, 3],
            "Another_Col": ["a", "b", "c"]
        })
        config = {"target": {"column_candidates": ["Non_Existent_Target"]}}
        
        with pytest.raises(ValueError, match="Nessuna colonna target trovata"):
            choose_target(df, config)
    
    def test_choose_target_default_config(self, sample_real_estate_data):
        """Test target selection with default configuration."""
        config = {}  # Empty config should use defaults
        
        target_col = choose_target(sample_real_estate_data, config)
        
        assert target_col == "AI_Prezzo_Ridistribuito"


class TestTargetTransformation:
    """Test target variable transformations."""
    
    def test_apply_log_transform_enabled(self):
        """Test log transformation when enabled."""
        config = {"target": {"log_transform": True}}
        y = pd.Series([100, 200, 300, 400])
        
        y_transformed, transform_info = apply_log_target_if(config, y)
        
        assert transform_info["log"] is True
        assert len(y_transformed) == len(y)
        # Check that transformation was applied
        assert not y_transformed.equals(y)
        # Log transformation should be monotonic
        assert y_transformed.is_monotonic_increasing
    
    def test_apply_log_transform_disabled(self):
        """Test no transformation when disabled."""
        config = {"target": {"log_transform": False}}
        y = pd.Series([100, 200, 300, 400])
        
        y_transformed, transform_info = apply_log_target_if(config, y)
        
        assert transform_info["log"] is False
        assert y_transformed.equals(y)
    
    def test_apply_log_transform_negative_values(self):
        """Test log transformation with negative values."""
        config = {"target": {"log_transform": True}}
        y = pd.Series([-10, 0, 10, 100])
        
        y_transformed, transform_info = apply_log_target_if(config, y)
        
        assert transform_info["log"] is True
        # Should handle negative values by clipping to small positive value
        assert (y_transformed >= 0).all()


class TestTemporalSplit:
    """Test temporal splitting functionality."""
    
    def test_temporal_key_creation(self, sample_real_estate_data):
        """Test temporal key creation."""
        key = temporal_key(
            sample_real_estate_data,
            "A_AnnoStipula",
            "A_MeseStipula"
        )
        
        assert len(key) == len(sample_real_estate_data)
        # Key should be year*100 + month
        assert (key >= 202001).any()  # 2020, January
        assert (key <= 202312).any()  # 2023, December
    
    def test_temporal_split_fraction_mode(self, sample_real_estate_data):
        """Test temporal split in fraction mode."""
        from preprocessing.transformers import TemporalSplitConfig
        
        config = TemporalSplitConfig(
            mode="fraction",
            train_fraction=0.8,
            year_col="A_AnnoStipula",
            month_col="A_MeseStipula"
        )
        
        train_df, test_df = temporal_split(sample_real_estate_data, config)
        
        total_samples = len(sample_real_estate_data)
        assert len(train_df) + len(test_df) == total_samples
        assert len(train_df) / total_samples == pytest.approx(0.8, abs=0.1)
    
    def test_temporal_split_date_mode(self, sample_real_estate_data):
        """Test temporal split in date mode."""
        from preprocessing.transformers import TemporalSplitConfig
        
        config = TemporalSplitConfig(
            mode="date",
            test_start_year=2023,
            test_start_month=1,
            year_col="A_AnnoStipula",
            month_col="A_MeseStipula"
        )
        
        train_df, test_df = temporal_split(sample_real_estate_data, config)
        
        # Test set should only contain data from 2023+
        test_years = test_df["A_AnnoStipula"]
        assert (test_years >= 2023).all()
        
        # Train set should contain data before 2023
        train_years = train_df["A_AnnoStipula"]
        assert (train_years < 2023).any()


class TestOutlierDetection:
    """Test outlier detection functionality."""
    
    def test_outlier_detection_iqr(self, sample_real_estate_data):
        """Test IQR-based outlier detection."""
        config = OutlierConfig(
            method="iqr",
            iqr_factor=1.5,
            group_by_col=None
        )
        
        outlier_mask = detect_outliers(
            sample_real_estate_data,
            "AI_Prezzo_Ridistribuito",
            config
        )
        
        assert len(outlier_mask) == len(sample_real_estate_data)
        assert isinstance(outlier_mask, pd.Series)
        assert outlier_mask.dtype == bool
        # Should detect some outliers
        assert outlier_mask.sum() > 0
    
    def test_outlier_detection_zscore(self, sample_real_estate_data):
        """Test Z-score based outlier detection."""
        config = OutlierConfig(
            method="zscore",
            z_thresh=3.0,
            group_by_col=None
        )
        
        outlier_mask = detect_outliers(
            sample_real_estate_data,
            "AI_Prezzo_Ridistribuito",
            config
        )
        
        assert len(outlier_mask) == len(sample_real_estate_data)
        assert outlier_mask.dtype == bool
    
    def test_outlier_detection_grouped(self, sample_real_estate_data):
        """Test grouped outlier detection."""
        config = OutlierConfig(
            method="iqr",
            iqr_factor=1.5,
            group_by_col="AI_IdCategoriaCatastale",
            min_group_size=10
        )
        
        outlier_mask = detect_outliers(
            sample_real_estate_data,
            "AI_Prezzo_Ridistribuito",
            config
        )
        
        assert len(outlier_mask) == len(sample_real_estate_data)
        assert outlier_mask.dtype == bool


class TestImputationLogic:
    """Test missing value imputation."""
    
        def test_imputation_numeric(self):
        """Test numeric imputation."""
        df = pd.DataFrame({
            "numeric_col": [1.0, 2.0, np.nan, 4.0, np.nan],
            "group_col": [1, 1, 1, 2, 2]
        })

        config = ImputationConfig(
            numeric_strategy="median",
            categorical_strategy="most_frequent",
            group_by_col=None
        )
    
        df_imputed = impute_missing(df, config)
        
        assert not df_imputed["numeric_col"].isna().any()
        # Median of [1, 2, 4] is 2
        assert df_imputed["numeric_col"].iloc[2] == 2.0
    
        def test_imputation_categorical(self):
        """Test categorical imputation."""
        df = pd.DataFrame({
            "cat_col": ["A", "B", None, "A", None],
            "group_col": [1, 1, 1, 2, 2]
        })

        config = ImputationConfig(
            numeric_strategy="median",
            categorical_strategy="most_frequent",
            group_by_col=None
        )

        df_imputed = impute_missing(df, config)
        
        assert not df_imputed["cat_col"].isna().any()
        # Most frequent is "A"
        assert df_imputed["cat_col"].iloc[2] == "A"
    
        def test_imputation_grouped(self):
        """Test grouped imputation."""
        df = pd.DataFrame({
            "numeric_col": [1.0, 2.0, np.nan, 10.0, np.nan],
            "group_col": [1, 1, 1, 2, 2]
        })

        config = ImputationConfig(
            numeric_strategy="mean",
            categorical_strategy="most_frequent",
            group_by_col="group_col"
        )

        df_imputed = impute_missing(df, config)
        
        assert not df_imputed["numeric_col"].isna().any()
        # Group 1: mean of [1, 2] = 1.5
        assert df_imputed["numeric_col"].iloc[2] == 1.5
        # Group 2: only one value, should use global mean or handle appropriately
        assert not pd.isna(df_imputed["numeric_col"].iloc[4])


class TestPreprocessingPipeline:
    """Integration tests for the complete preprocessing pipeline."""
    
    def test_preprocessing_pipeline_minimal(self, temp_dir, sample_real_estate_data):
        """Test minimal preprocessing pipeline execution."""
        # Setup test data
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()
        sample_real_estate_data.to_parquet(raw_dir / "raw.parquet")
        
        # Minimal config
        config = {
            "paths": {
                "raw_data": str(raw_dir),
                "preprocessed_data": str(temp_dir / "preprocessed")
            },
            "target": {"column_candidates": ["AI_Prezzo_Ridistribuito"]},
            "temporal_split": {
                "year_col": "A_AnnoStipula",
                "month_col": "A_MeseStipula",
                "mode": "fraction",
                "train_fraction": 0.8,
                "valid_fraction": 0.1
            },
            "profiles": {
                "tree": {"enabled": True, "output_prefix": "tree"}
            },
            "feature_extraction": {"geometry": True, "json": True},
            "outliers": {"method": "iqr"},
            "imputation": {"numeric_strategy": "median"},
            "encoding": {"max_ohe_cardinality": 10}
        }
        
        # Should not raise exceptions
        result_path = run_preprocessing(config)
        
        assert result_path.exists()
        # Check that output files were created
        preprocessed_dir = temp_dir / "preprocessed"
        assert (preprocessed_dir / "X_train_tree.parquet").exists()
        assert (preprocessed_dir / "y_train_tree.parquet").exists()
        assert (preprocessed_dir / "X_test_tree.parquet").exists()
        assert (preprocessed_dir / "y_test_tree.parquet").exists()
    
    def test_preprocessing_pipeline_no_raw_data(self, temp_dir):
        """Test pipeline behavior with no raw data."""
        config = {
            "paths": {
                "raw_data": str(temp_dir / "empty_raw"),
                "preprocessed_data": str(temp_dir / "preprocessed")
            }
        }
        
        with pytest.raises(FileNotFoundError, match="Nessun file parquet trovato"):
            run_preprocessing(config)
    
    @patch('preprocessing.pipeline.logger')
    def test_preprocessing_logging(self, mock_logger, temp_dir, sample_real_estate_data):
        """Test that preprocessing logs appropriately."""
        # Setup test data
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()
        sample_real_estate_data.to_parquet(raw_dir / "raw.parquet")
        
        config = {
            "paths": {
                "raw_data": str(raw_dir),
                "preprocessed_data": str(temp_dir / "preprocessed")
            },
            "target": {"column_candidates": ["AI_Prezzo_Ridistribuito"]},
            "temporal_split": {"mode": "fraction", "train_fraction": 0.8, "valid_fraction": 0.0},
            "profiles": {"tree": {"enabled": True, "output_prefix": "tree"}},
            "feature_extraction": {"geometry": False, "json": False},
            "outliers": {"method": "iqr"},
            "imputation": {"numeric_strategy": "median"},
            "encoding": {"max_ohe_cardinality": 10}
        }
        
        run_preprocessing(config)
        
        # Check that info logging was called
        assert mock_logger.info.called
        # Should log data loading, feature extraction, etc.
        logged_messages = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Caricamento raw completato" in msg for msg in logged_messages)


class TestDataValidation:
    """Test data validation during preprocessing."""
    
    def test_empty_dataframe_handling(self, temp_dir):
        """Test handling of empty dataframes."""
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()
        
        # Create empty dataframe
        empty_df = pd.DataFrame()
        empty_df.to_parquet(raw_dir / "raw.parquet")
        
        config = {
            "paths": {
                "raw_data": str(raw_dir),
                "preprocessed_data": str(temp_dir / "preprocessed")
            },
            "target": {"column_candidates": ["AI_Prezzo_Ridistribuito"]}
        }
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, KeyError)):
            run_preprocessing(config)
    
    def test_missing_target_column(self, temp_dir):
        """Test behavior when target column is missing."""
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()
        
        # Create dataframe without target column
        df = pd.DataFrame({
            "some_col": [1, 2, 3],
            "another_col": ["a", "b", "c"]
        })
        df.to_parquet(raw_dir / "raw.parquet")
        
        config = {
            "paths": {
                "raw_data": str(raw_dir),
                "preprocessed_data": str(temp_dir / "preprocessed")
            },
            "target": {"column_candidates": ["NonExistentTarget"]}
        }
        
        with pytest.raises(ValueError, match="Nessuna colonna target trovata"):
            run_preprocessing(config)


if __name__ == "__main__":
    pytest.main([__file__])