"""Tests for diagnostics module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.diagnostics import (
    residual_analysis,
    drift_detection,
    prediction_intervals,
    calculate_psi,
    compute_residuals
)


class TestDiagnostics:
    """Test diagnostic functions."""
    
    def test_compute_residuals(self):
        """Test residual computation."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        
        residuals = compute_residuals(y_true, y_pred)
        expected = np.array([-10, 10, -10])
        
        np.testing.assert_array_equal(residuals, expected)
    
    def test_calculate_psi_no_drift(self):
        """Test PSI calculation with no drift."""
        # Same distribution
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)
        
        psi = calculate_psi(expected, actual)
        
        # Should be very low (< 0.1 for no drift)
        assert psi < 0.15
    
    def test_calculate_psi_with_drift(self):
        """Test PSI calculation with significant drift."""
        # Different distributions
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(2, 1.5, 1000)  # Shifted mean and variance
        
        psi = calculate_psi(expected, actual)
        
        # Should detect drift (> 0.15)
        assert psi > 0.15
    
    def test_residual_analysis_basic(self):
        """Test basic residual analysis."""
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            y_true = np.array([100, 200, 300, 400, 500])
            y_pred = np.array([110, 190, 310, 390, 510])
            X = pd.DataFrame({"group": ["A", "A", "B", "B", "C"]})
            
            config = {
                "diagnostics": {
                    "residual_analysis": {
                        "enabled": True,
                        "by_groups": ["group"],
                        "save_worst_predictions": True,
                        "top_n_worst": 3,
                        "plots": []  # No plots for test
                    }
                }
            }
            
            results = residual_analysis("test_model", y_true, y_pred, X, config, output_dir)
            
            # Check overall stats
            assert "overall" in results
            assert "mean" in results["overall"]
            assert "std" in results["overall"]
            
            # Check group analysis
            assert "by_group" in results
            assert "group" in results["by_group"]
            assert "A" in results["by_group"]["group"]
            assert "B" in results["by_group"]["group"]
            
            # Check worst predictions file
            worst_file = output_dir / "test_model_worst_predictions.csv"
            assert worst_file.exists()
    
    def test_residual_analysis_disabled(self):
        """Test that residual analysis can be disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            y_true = np.array([100, 200, 300])
            y_pred = np.array([110, 190, 310])
            
            config = {
                "diagnostics": {
                    "residual_analysis": {
                        "enabled": False
                    }
                }
            }
            
            results = residual_analysis("test_model", y_true, y_pred, None, config, output_dir)
            
            # Should return empty dict
            assert results == {}
    
    def test_drift_detection_basic(self):
        """Test basic drift detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create similar distributions (no drift)
            X_train = pd.DataFrame({
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(10, 2, 100)
            })
            
            X_test = pd.DataFrame({
                "feature1": np.random.normal(0, 1, 50),
                "feature2": np.random.normal(10, 2, 50)
            })
            
            config = {
                "diagnostics": {
                    "drift_detection": {
                        "enabled": True,
                        "methods": ["psi", "ks_test"],
                        "alert_threshold": 0.15,
                        "save_report": True
                    }
                }
            }
            
            results = drift_detection(X_train, X_test, config, output_dir)
            
            # Check structure
            assert "features" in results
            assert "alerts" in results
            assert "summary" in results
            
            # Should have checked both features
            assert "feature1" in results["features"]
            assert "feature2" in results["features"]
            
            # Check summary
            assert results["summary"]["total_features_checked"] == 2
            
            # Check report file
            report_file = output_dir / "drift_report.json"
            assert report_file.exists()
    
    def test_drift_detection_with_drift(self):
        """Test drift detection with actual drift."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Create shifted distributions (drift)
            X_train = pd.DataFrame({
                "feature1": np.random.normal(0, 1, 100),
            })
            
            X_test = pd.DataFrame({
                "feature1": np.random.normal(5, 1, 50),  # Shifted by 5
            })
            
            config = {
                "diagnostics": {
                    "drift_detection": {
                        "enabled": True,
                        "methods": ["psi", "ks_test"],
                        "alert_threshold": 0.15,
                        "save_report": True
                    }
                }
            }
            
            results = drift_detection(X_train, X_test, config, output_dir)
            
            # Should detect drift
            assert len(results["alerts"]) > 0
            assert results["summary"]["total_alerts"] > 0
    
    def test_prediction_intervals_basic(self):
        """Test prediction intervals computation."""
        # Create a simple model
        from sklearn.linear_model import LinearRegression
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Generate data
            np.random.seed(42)
            X_train = np.random.rand(100, 3)
            y_train = X_train.sum(axis=1) + np.random.normal(0, 0.1, 100)
            
            X_test = np.random.rand(50, 3)
            y_test = X_test.sum(axis=1) + np.random.normal(0, 0.1, 50)
            
            # Fit model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            config = {
                "uncertainty": {
                    "prediction_intervals": {
                        "enabled": True,
                        "method": "residual_bootstrap",
                        "n_bootstraps": 50,  # Lower for test
                        "confidence_levels": [0.8, 0.9]
                    }
                }
            }
            
            results = prediction_intervals(
                model, X_train, y_train, X_test, y_test, 
                config, "test_model", output_dir
            )
            
            # Check structure
            assert "80%" in results
            assert "90%" in results
            
            # Check 80% interval
            assert "coverage" in results["80%"]
            assert "average_width" in results["80%"]
            assert "target_coverage" in results["80%"]
            
            # Coverage should be close to target
            assert results["80%"]["target_coverage"] == 0.8
            assert 0.6 < results["80%"]["coverage"] < 1.0  # Reasonable range
            
            # 90% interval should be wider
            assert results["90%"]["average_width"] > results["80%"]["average_width"]
            
            # Check file saved
            interval_file = output_dir / "test_model_prediction_intervals.json"
            assert interval_file.exists()
    
    def test_prediction_intervals_disabled(self):
        """Test that prediction intervals can be disabled."""
        from sklearn.linear_model import LinearRegression
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            X_train = np.random.rand(50, 3)
            y_train = X_train.sum(axis=1)
            X_test = np.random.rand(20, 3)
            y_test = X_test.sum(axis=1)
            
            model = LinearRegression().fit(X_train, y_train)
            
            config = {
                "uncertainty": {
                    "prediction_intervals": {
                        "enabled": False
                    }
                }
            }
            
            results = prediction_intervals(
                model, X_train, y_train, X_test, y_test,
                config, "test_model", output_dir
            )
            
            # Should return empty dict
            assert results == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
