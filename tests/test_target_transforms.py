"""Tests for target transformations."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.target_transforms import (
    apply_target_transform,
    inverse_target_transform,
    get_transform_name,
)


class TestTargetTransforms:
    """Test target variable transformations."""
    
    def test_none_transform(self):
        """Test no transformation."""
        y = pd.Series([100, 200, 300])
        y_trans, metadata = apply_target_transform(y, transform_type="none")
        
        assert metadata["transform"] == "none"
        np.testing.assert_array_almost_equal(y_trans, y)
        
        # Inverse should be identity
        y_back = inverse_target_transform(y_trans, metadata)
        np.testing.assert_array_almost_equal(y_back, y)
    
    def test_log_transform(self):
        """Test log1p transformation."""
        y = pd.Series([0, 100, 1000])
        y_trans, metadata = apply_target_transform(y, transform_type="log")
        
        assert metadata["transform"] == "log"
        expected = np.log1p(y)
        np.testing.assert_array_almost_equal(y_trans, expected)
        
        # Inverse should recover original
        y_back = inverse_target_transform(y_trans, metadata)
        np.testing.assert_array_almost_equal(y_back, y, decimal=10)
    
    def test_log10_transform(self):
        """Test log10 transformation."""
        y = pd.Series([10, 100, 1000])
        y_trans, metadata = apply_target_transform(y, transform_type="log10", log10_offset=1.0)
        
        assert metadata["transform"] == "log10"
        assert metadata["log10_offset"] == 1.0
        expected = np.log10(y + 1.0)
        np.testing.assert_array_almost_equal(y_trans, expected)
        
        # Inverse
        y_back = inverse_target_transform(y_trans, metadata)
        np.testing.assert_array_almost_equal(y_back, y, decimal=10)
    
    def test_sqrt_transform(self):
        """Test square root transformation."""
        y = pd.Series([0, 100, 400])
        y_trans, metadata = apply_target_transform(y, transform_type="sqrt")
        
        assert metadata["transform"] == "sqrt"
        expected = np.sqrt(y)
        np.testing.assert_array_almost_equal(y_trans, expected)
        
        # Inverse
        y_back = inverse_target_transform(y_trans, metadata)
        np.testing.assert_array_almost_equal(y_back, y, decimal=10)
    
    def test_boxcox_transform(self):
        """Test Box-Cox transformation."""
        y = pd.Series([100, 200, 300, 400, 500])
        y_trans, metadata = apply_target_transform(y, transform_type="boxcox")
        
        assert metadata["transform"] == "boxcox"
        assert "lambda" in metadata
        assert isinstance(metadata["lambda"], float)
        
        # Inverse should recover original
        y_back = inverse_target_transform(y_trans, metadata)
        np.testing.assert_array_almost_equal(y_back, y, decimal=6)
    
    def test_boxcox_with_shift(self):
        """Test Box-Cox handles non-positive values with shift."""
        y = pd.Series([-50, 0, 50, 100])
        y_trans, metadata = apply_target_transform(y, transform_type="boxcox")
        
        assert metadata["transform"] == "boxcox"
        assert "shift" in metadata
        assert metadata["shift"] > 0  # Should have applied shift
        
        # Inverse should recover original
        y_back = inverse_target_transform(y_trans, metadata)
        np.testing.assert_array_almost_equal(y_back, y, decimal=6)
    
    def test_yeojohnson_transform(self):
        """Test Yeo-Johnson transformation."""
        y = pd.Series([100, 200, 300, 400, 500])
        y_trans, metadata = apply_target_transform(y, transform_type="yeojohnson")
        
        assert metadata["transform"] == "yeojohnson"
        assert "lambda" in metadata
        assert isinstance(metadata["lambda"], float)
        
        # Inverse should recover original
        y_back = inverse_target_transform(y_trans, metadata)
        np.testing.assert_array_almost_equal(y_back, y, decimal=6)
    
    def test_yeojohnson_with_negatives(self):
        """Test Yeo-Johnson works with negative values."""
        y = pd.Series([-100, -50, 0, 50, 100, 200])
        y_trans, metadata = apply_target_transform(y, transform_type="yeojohnson")
        
        assert metadata["transform"] == "yeojohnson"
        
        # Inverse should recover original
        y_back = inverse_target_transform(y_trans, metadata)
        np.testing.assert_array_almost_equal(y_back, y, decimal=6)
    
    def test_transform_preserves_series_type(self):
        """Test that Series input returns Series output."""
        y = pd.Series([100, 200, 300], name="price")
        y_trans, metadata = apply_target_transform(y, transform_type="log")
        
        assert isinstance(y_trans, pd.Series)
        assert y_trans.name == "price"
        
        y_back = inverse_target_transform(y_trans, metadata)
        assert isinstance(y_back, pd.Series)
        assert y_back.name == "price"
    
    def test_transform_with_numpy_array(self):
        """Test that numpy array input works."""
        y = np.array([100, 200, 300])
        y_trans, metadata = apply_target_transform(y, transform_type="log")
        
        assert isinstance(y_trans, np.ndarray)
        
        y_back = inverse_target_transform(y_trans, metadata)
        assert isinstance(y_back, np.ndarray)
        np.testing.assert_array_almost_equal(y_back, y)
    
    def test_get_transform_name(self):
        """Test transform name formatting."""
        assert get_transform_name({"transform": "none"}) == "No transformation"
        assert get_transform_name({"transform": "log"}) == "Log1p"
        assert get_transform_name({"transform": "sqrt"}) == "Square root"
        
        # NOTE: get_transform_name now looks for both old keys (boxcox_lambda) and new keys (lambda)
        boxcox_meta = {"transform": "boxcox", "lambda": 0.5}
        assert "Box-Cox" in get_transform_name(boxcox_meta)
        # Lambda value should be formatted in the name
        
        yj_meta = {"transform": "yeojohnson", "lambda": 1.2}
        assert "Yeo-Johnson" in get_transform_name(yj_meta)
    
    # NOTE: validate_transform_compatibility was removed
    # Transformations now handle edge cases automatically:
    # - sqrt: clamps negatives to 0
    # - boxcox: auto-shifts if needed
    # - yeojohnson: works with any values
    # Test removed as function no longer exists
    
    def test_invalid_transform_type(self):
        """Test that invalid transform type raises error."""
        y = pd.Series([100, 200, 300])
        
        with pytest.raises(ValueError, match="Unknown transform type"):
            apply_target_transform(y, transform_type="invalid")
        
        with pytest.raises(ValueError, match="Unknown transform type"):
            inverse_target_transform(y, {"transform": "invalid"})
    
    def test_reproducibility(self):
        """Test that same input produces same output."""
        y = pd.Series([100, 200, 300, 400, 500])
        
        # Box-Cox should be deterministic for same input
        y_trans1, meta1 = apply_target_transform(y.copy(), transform_type="boxcox")
        y_trans2, meta2 = apply_target_transform(y.copy(), transform_type="boxcox")
        
        np.testing.assert_array_almost_equal(y_trans1, y_trans2)
        np.testing.assert_almost_equal(meta1["lambda"], meta2["lambda"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
