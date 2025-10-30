"""Tests to verify overflow prevention in inverse transformations."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.target_transforms import (
    apply_target_transform,
    inverse_target_transform
)


class TestOverflowPrevention:
    """Test that inverse transformations don't cause overflow."""
    
    def test_large_log_values_no_overflow(self):
        """Test that large log-transformed values don't overflow on inverse."""
        # Create large log-transformed values (that would overflow with np.expm1)
        y_log = pd.Series([50, 100, 200, 500, 700])  # These would overflow with np.expm1
        metadata = {"transform": "log"}
        
        # This should NOT raise overflow warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_original = inverse_target_transform(y_log, metadata)
            
            # Check no overflow warnings
            overflow_warnings = [warn for warn in w if "overflow" in str(warn.message)]
            assert len(overflow_warnings) == 0, f"Got overflow warnings: {overflow_warnings}"
        
        # Result should be finite
        assert np.all(np.isfinite(y_original))
    
    def test_extreme_boxcox_values(self):
        """Test Box-Cox with extreme values."""
        # Very large prices
        y = pd.Series([1e6, 1e7, 1e8])
        y_trans, metadata = apply_target_transform(y, transform_type="boxcox")
        
        # Inverse should work without overflow
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_back = inverse_target_transform(y_trans, metadata)
            
            overflow_warnings = [warn for warn in w if "overflow" in str(warn.message)]
            assert len(overflow_warnings) == 0
        
        # Should recover original (with some tolerance for very large numbers)
        np.testing.assert_allclose(y_back, y, rtol=1e-4)
    
    def test_extreme_yeojohnson_values(self):
        """Test Yeo-Johnson with extreme values."""
        # Mix of very large and very small values
        y = pd.Series([-1e6, -1e3, 0, 1e3, 1e6, 1e8])
        y_trans, metadata = apply_target_transform(y, transform_type="yeojohnson")
        
        # Inverse should work without overflow
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_back = inverse_target_transform(y_trans, metadata)
            
            overflow_warnings = [warn for warn in w if "overflow" in str(warn.message)]
            assert len(overflow_warnings) == 0
        
        # Should recover original
        np.testing.assert_allclose(y_back, y, rtol=1e-4)
    
    def test_log_with_realistic_price_range(self):
        """Test log transformation with realistic real estate prices."""
        # Realistic price range (euros)
        y = pd.Series([50000, 100000, 200000, 500000, 1000000, 5000000])
        y_trans, metadata = apply_target_transform(y, transform_type="log")
        
        # Inverse should not overflow
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_back = inverse_target_transform(y_trans, metadata)
            
            overflow_warnings = [warn for warn in w if "overflow" in str(warn.message)]
            assert len(overflow_warnings) == 0
        
        # Should match original closely
        np.testing.assert_allclose(y_back, y, rtol=1e-10)
    
    def test_sqrt_large_values(self):
        """Test sqrt with large values."""
        y = pd.Series([1e6, 1e8, 1e10])
        y_trans, metadata = apply_target_transform(y, transform_type="sqrt")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_back = inverse_target_transform(y_trans, metadata)
            
            overflow_warnings = [warn for warn in w if "overflow" in str(warn.message)]
            assert len(overflow_warnings) == 0
        
        np.testing.assert_allclose(y_back, y, rtol=1e-10)
    
    def test_log10_large_values(self):
        """Test log10 with large values."""
        y = pd.Series([1e6, 1e7, 1e8])
        y_trans, metadata = apply_target_transform(y, transform_type="log10", log10_offset=1.0)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_back = inverse_target_transform(y_trans, metadata)
            
            overflow_warnings = [warn for warn in w if "overflow" in str(warn.message)]
            assert len(overflow_warnings) == 0
        
        np.testing.assert_allclose(y_back, y, rtol=1e-10)
    
    def test_array_vs_series_consistency(self):
        """Test that array and series inputs behave consistently."""
        y_arr = np.array([100000, 200000, 500000])
        y_ser = pd.Series(y_arr)
        
        # Apply same transformation
        y_arr_trans, meta_arr = apply_target_transform(y_arr, transform_type="boxcox")
        y_ser_trans, meta_ser = apply_target_transform(y_ser, transform_type="boxcox")
        
        # Lambda should be the same (deterministic)
        np.testing.assert_almost_equal(
            meta_arr["boxcox_lambda"],
            meta_ser["boxcox_lambda"]
        )
        
        # Inverse should give same results
        y_arr_back = inverse_target_transform(y_arr_trans, meta_arr)
        y_ser_back = inverse_target_transform(y_ser_trans, meta_ser)
        
        np.testing.assert_allclose(y_arr_back, y_ser_back.values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
