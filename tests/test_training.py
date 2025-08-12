"""Tests for training functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add src to path for testing
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.train import run_training, _load_xy, _catboost_cat_features
from training.model_zoo import build_estimator
from training.metrics import regression_metrics, overfit_diagnostics
from training.tuner import tune_model


class TestModelZoo:
    """Test model building functionality."""
    
    def test_build_estimator_linear(self):
        """Test building linear models."""
        estimator = build_estimator("linear", {})
        
        from sklearn.linear_model import LinearRegression
        assert isinstance(estimator, LinearRegression)
    
    def test_build_estimator_ridge(self):
        """Test building ridge regression."""
        params = {"alpha": 1.0}
        estimator = build_estimator("ridge", params)
        
        from sklearn.linear_model import Ridge
        assert isinstance(estimator, Ridge)
        assert estimator.alpha == 1.0
    
    def test_build_estimator_random_forest(self):
        """Test building random forest."""
        params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
        estimator = build_estimator("rf", params)
        
        from sklearn.ensemble import RandomForestRegressor
        assert isinstance(estimator, RandomForestRegressor)
        assert estimator.n_estimators == 100
        assert estimator.max_depth == 5
        assert estimator.random_state == 42
    
    def test_build_estimator_xgboost(self):
        """Test building XGBoost model."""
        params = {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
        estimator = build_estimator("xgboost", params)
        
        import xgboost as xgb
        assert isinstance(estimator, xgb.XGBRegressor)
        assert estimator.n_estimators == 100
        assert estimator.learning_rate == 0.1
    
    def test_build_estimator_unknown(self):
        """Test building unknown model type."""
        with pytest.raises(ValueError, match="Unknown model key"):
            build_estimator("unknown_model", {})


class TestMetrics:
    """Test metrics calculation."""
    
    def test_regression_metrics_perfect(self):
        """Test metrics with perfect predictions."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
        y_pred = pd.Series([1.0, 2.0, 3.0, 4.0])
        
        metrics = regression_metrics(y_true, y_pred)
        
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["mse"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["mape"] == pytest.approx(0.0, abs=1e-10)
    
    def test_regression_metrics_realistic(self):
        """Test metrics with realistic predictions."""
        y_true = pd.Series([100, 200, 300, 400])
        y_pred = pd.Series([110, 190, 310, 390])
        
        metrics = regression_metrics(y_true, y_pred)
        
        assert 0.8 < metrics["r2"] < 1.0  # Should be quite good
        assert metrics["rmse"] > 0
        assert metrics["mse"] > 0
        assert metrics["mae"] > 0
        assert metrics["mape"] > 0
    
    def test_regression_metrics_with_zeros(self):
        """Test metrics handling zeros in target."""
        y_true = pd.Series([0.0, 1.0, 2.0, 3.0])
        y_pred = pd.Series([0.1, 1.1, 1.9, 2.9])
        
        metrics = regression_metrics(y_true, y_pred)
        
        # Should handle division by zero in MAPE
        assert not np.isnan(metrics["mape"])
        assert not np.isinf(metrics["mape"])
    
    def test_overfit_diagnostics(self):
        """Test overfitting diagnostics."""
        # Perfect fit on train, poor on validation
        train_metrics = {"r2": 0.99, "rmse": 0.1}
        val_metrics = {"r2": 0.5, "rmse": 2.0}
        
        diagnostics = overfit_diagnostics(train_metrics, val_metrics)
        
        assert "gap_r2" in diagnostics
        assert "ratio_rmse" in diagnostics
        assert diagnostics["gap_r2"] == pytest.approx(0.49)
        assert diagnostics["ratio_rmse"] == pytest.approx(20.0)


class TestDataLoading:
    """Test data loading utilities."""
    
    def test_load_xy_with_validation(self, sample_training_data):
        """Test loading training data with validation set."""
        # Create validation files
        X_val = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 20),
            "feature2": np.random.normal(0, 1, 20),
            "feature3": np.random.randint(0, 5, 20)
        })
        y_val = pd.DataFrame({
            "AI_Prezzo_Ridistribuito": np.random.lognormal(12, 0.5, 20)
        })
        
        X_val.to_parquet(sample_training_data / "X_val_tree.parquet")
        y_val.to_parquet(sample_training_data / "y_val_tree.parquet")
        
        X_train, y_train, X_val_loaded, y_val_loaded, X_test, y_test = _load_xy(
            sample_training_data, "tree"
        )
        
        assert len(X_train) == 100
        assert len(y_train) == 100
        assert len(X_val_loaded) == 20
        assert len(y_val_loaded) == 20
        assert len(X_test) == 30
        assert len(y_test) == 30
        
        # Check target column extraction
        assert y_train.name == "AI_Prezzo_Ridistribuito"
        assert y_val_loaded.name == "AI_Prezzo_Ridistribuito"
        assert y_test.name == "AI_Prezzo_Ridistribuito"
    
    def test_load_xy_without_validation(self, sample_training_data):
        """Test loading training data without validation set."""
        X_train, y_train, X_val, y_val, X_test, y_test = _load_xy(
            sample_training_data, "tree"
        )
        
        assert len(X_train) == 100
        assert len(y_train) == 100
        assert X_val is None
        assert y_val is None
        assert len(X_test) == 30
        assert len(y_test) == 30
    
    def test_catboost_cat_features(self, temp_dir):
        """Test CatBoost categorical features detection."""
        # Create categorical columns file
        cat_file = temp_dir / "categorical_columns_catboost.txt"
        cat_file.write_text("feature1\nfeature3\n")
        
        X = pd.DataFrame({
            "feature1": ["A", "B", "C"],
            "feature2": [1, 2, 3],
            "feature3": ["X", "Y", "Z"]
        })
        
        cat_indices = _catboost_cat_features(temp_dir, "catboost", X)
        
        assert cat_indices == [0, 2]  # Indices of feature1 and feature3
    
    def test_catboost_cat_features_fallback(self, temp_dir):
        """Test CatBoost categorical features fallback to dtype inference."""
        X = pd.DataFrame({
            "numeric_col": [1, 2, 3],
            "object_col": ["A", "B", "C"],
            "category_col": pd.Categorical(["X", "Y", "Z"])
        })
        
        cat_indices = _catboost_cat_features(temp_dir, "catboost", X)
        
        # Should detect object and category columns
        assert 1 in cat_indices  # object_col
        assert 2 in cat_indices  # category_col
        assert 0 not in cat_indices  # numeric_col


class TestTuning:
    """Test hyperparameter tuning."""
    
    @patch('training.tuner.optuna.create_study')
    def test_tune_model_basic(self, mock_create_study):
        """Test basic model tuning."""
        # Mock Optuna study
        mock_study = MagicMock()
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 1.0
        mock_trial.suggest_int.return_value = 100
        mock_trial.suggest_categorical.return_value = "sqrt"
        
        mock_study.best_params = {"alpha": 1.0}
        mock_study.best_value = 0.85
        mock_create_study.return_value = mock_study
        
        # Sample data
        X = pd.DataFrame(np.random.randn(100, 3))
        y = pd.Series(np.random.randn(100))
        
        search_space = {
            "alpha": {"type": "float", "low": 0.1, "high": 10.0, "log": True}
        }
        
        result = tune_model(
            "ridge", X, y, None, None, "r2", 10, None, "tpe", 42, {}, search_space
        )
        
        assert result.best_params == {"alpha": 1.0}
        assert result.best_value == 0.85
        mock_study.optimize.assert_called_once()


class TestTrainingPipeline:
    """Test complete training pipeline."""
    
    def test_run_training_minimal(self, sample_training_data):
        """Test minimal training pipeline execution."""
        config = {
            "paths": {
                "preprocessed_data": str(sample_training_data),
                "models_dir": str(sample_training_data / "models")
            },
            "training": {
                "primary_metric": "r2",
                "report_metrics": ["r2", "rmse"],
                "seed": 42,
                "models": {
                    "linear": {
                        "enabled": True,
                        "profile": "tree",
                        "trials": 1,
                        "base_params": {},
                        "fit_params": {},
                        "search_space": {}
                    }
                },
                "shap": {"enabled": False},
                "ensembles": {
                    "voting": {"enabled": False},
                    "stacking": {"enabled": False}
                }
            }
        }
        
        results = run_training(config)
        
        assert "models" in results
        assert "linear" in results["models"]
        assert "metrics_test" in results["models"]["linear"]
        assert "metrics_train" in results["models"]["linear"]
    
    def test_run_training_with_validation(self, sample_training_data):
        """Test training with validation set."""
        # Create validation files
        X_val = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 20),
            "feature2": np.random.normal(0, 1, 20),
            "feature3": np.random.randint(0, 5, 20)
        })
        y_val = pd.DataFrame({
            "AI_Prezzo_Ridistribuito": np.random.lognormal(12, 0.5, 20)
        })
        
        X_val.to_parquet(sample_training_data / "X_val_tree.parquet")
        y_val.to_parquet(sample_training_data / "y_val_tree.parquet")
        
        config = {
            "paths": {
                "preprocessed_data": str(sample_training_data),
                "models_dir": str(sample_training_data / "models")
            },
            "training": {
                "primary_metric": "r2",
                "report_metrics": ["r2", "rmse"],
                "seed": 42,
                "models": {
                    "ridge": {
                        "enabled": True,
                        "profile": "tree",
                        "trials": 5,
                        "base_params": {},
                        "fit_params": {},
                        "search_space": {
                            "alpha": {"type": "float", "low": 0.1, "high": 10.0, "log": True}
                        }
                    }
                },
                "shap": {"enabled": False},
                "ensembles": {
                    "voting": {"enabled": False},
                    "stacking": {"enabled": False}
                }
            }
        }
        
        results = run_training(config)
        
        assert "models" in results
        assert "ridge" in results["models"]
        model_results = results["models"]["ridge"]
        
        # Should have train and test metrics
        assert "metrics_train" in model_results
        assert "metrics_test" in model_results
        
        # Should have overfitting diagnostics
        assert "overfit" in model_results
    
    @patch('training.train.logger')
    def test_training_logging(self, mock_logger, sample_training_data):
        """Test that training logs appropriately."""
        config = {
            "paths": {
                "preprocessed_data": str(sample_training_data),
                "models_dir": str(sample_training_data / "models")
            },
            "training": {
                "primary_metric": "r2",
                "seed": 42,
                "models": {
                    "linear": {
                        "enabled": True,
                        "profile": "tree",
                        "trials": 1,
                        "base_params": {},
                        "fit_params": {},
                        "search_space": {}
                    }
                },
                "shap": {"enabled": False},
                "ensembles": {"voting": {"enabled": False}, "stacking": {"enabled": False}}
            }
        }
        
        run_training(config)
        
        # Check that logging was called
        assert mock_logger.info.called
    
    def test_training_no_models_enabled(self, sample_training_data):
        """Test training with no models enabled."""
        config = {
            "paths": {
                "preprocessed_data": str(sample_training_data),
                "models_dir": str(sample_training_data / "models")
            },
            "training": {
                "primary_metric": "r2",
                "seed": 42,
                "models": {
                    "linear": {"enabled": False}
                },
                "shap": {"enabled": False},
                "ensembles": {"voting": {"enabled": False}, "stacking": {"enabled": False}}
            }
        }
        
        results = run_training(config)
        
        # Should return empty results but not crash
        assert "models" in results
        assert len(results["models"]) == 0
    
    def test_training_missing_data_files(self, temp_dir):
        """Test training with missing data files."""
        config = {
            "paths": {
                "preprocessed_data": str(temp_dir),
                "models_dir": str(temp_dir / "models")
            },
            "training": {
                "models": {
                    "linear": {
                        "enabled": True,
                        "profile": "tree",
                        "trials": 1
                    }
                }
            }
        }
        
        # Should handle missing files gracefully with empty results
        results = run_training(config)
        
        # Should return empty models when data files are missing
        assert "models" in results
        assert len(results["models"]) == 0


class TestErrorHandling:
    """Test error handling in training."""
    
    def test_invalid_model_type(self):
        """Test handling of invalid model types."""
        with pytest.raises(ValueError):
            build_estimator("invalid_model", {})
    
    def test_metrics_with_invalid_data(self):
        """Test metrics calculation with invalid data."""
        y_true = pd.Series([1, 2, np.nan, 4])
        y_pred = pd.Series([1.1, 1.9, 3.1, 3.9])
        
        # Should handle NaN values appropriately
        metrics = regression_metrics(y_true, y_pred)
        
        # Should not contain NaN or inf values
        for metric_name, value in metrics.items():
            assert not np.isnan(value), f"Metric {metric_name} is NaN"
            assert not np.isinf(value), f"Metric {metric_name} is infinite"
    
    def test_empty_training_data(self):
        """Test training with empty data."""
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
        
        # Should handle empty data gracefully
        with pytest.raises((ValueError, IndexError)):
            regression_metrics(y, pd.Series(dtype=float))


if __name__ == "__main__":
    pytest.main([__file__])