#!/usr/bin/env python3
"""
Test script to verify LightGBM feature names warning fix.
This script simulates the warning scenario and tests the fix.
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

def test_lightgbm_feature_names_warning():
    """Test if LightGBM feature names warning is resolved."""
    print("Testing LightGBM feature names warning fix...")
    
    # Create sample data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    
    # Convert to DataFrame with feature names (simulating preprocessed data)
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Training data type: {type(X_train)}")
    print(f"Test data type: {type(X_test)}")
    
    # Test BEFORE fix (this would generate the warning)
    print("\n--- Testing BEFORE fix (expected warning) ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        lgbm_bad = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
        lgbm_bad.fit(X_train, y_train)  # Fit with DataFrame (feature names)
        _ = lgbm_bad.predict(X_test.values)  # Predict with numpy array (no feature names)
        
        warning_found = any("X does not have valid feature names" in str(warning.message) for warning in w)
        if warning_found:
            print("✓ Warning reproduced successfully")
            for warning in w:
                if "X does not have valid feature names" in str(warning.message):
                    print(f"  Warning: {warning.message}")
        else:
            print("✗ Warning not found (unexpected)")
    
    # Test AFTER fix (this should NOT generate the warning)
    print("\n--- Testing AFTER fix (no warning expected) ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        lgbm_good = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
        lgbm_good.fit(X_train.values, y_train.values)  # Fit with numpy arrays (no feature names)
        _ = lgbm_good.predict(X_test.values)  # Predict with numpy arrays (no feature names)
        
        warning_found = any("X does not have valid feature names" in str(warning.message) for warning in w)
        if not warning_found:
            print("✓ No warning generated - fix working correctly!")
        else:
            print("✗ Warning still present (fix may not be working)")
            for warning in w:
                if "X does not have valid feature names" in str(warning.message):
                    print(f"  Warning: {warning.message}")
    
    # Test that model performance is equivalent
    print("\n--- Testing model performance equivalence ---")
    lgbm1 = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
    lgbm2 = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
    
    # Train one with DataFrames, one with numpy arrays
    lgbm1.fit(X_train, y_train)
    lgbm2.fit(X_train.values, y_train.values)
    
    # Predict with consistent formats
    pred1 = lgbm1.predict(X_test)
    pred2 = lgbm2.predict(X_test.values)
    
    # Calculate R² scores
    from sklearn.metrics import r2_score
    r2_1 = r2_score(y_test, pred1)
    r2_2 = r2_score(y_test, pred2)
    
    print(f"R² with DataFrame fit: {r2_1:.6f}")
    print(f"R² with numpy fit: {r2_2:.6f}")
    print(f"Difference: {abs(r2_1 - r2_2):.6f}")
    
    if abs(r2_1 - r2_2) < 1e-10:  # Very small tolerance for numerical differences
        print("✓ Model performance is equivalent!")
    else:
        print("✗ Model performance differs (may be due to random seed differences)")
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("The fix ensures LightGBM always uses .values (numpy arrays)")
    print("for both training and prediction, eliminating the feature names warning")
    print("while maintaining equivalent model performance.")
    print("="*50)

if __name__ == "__main__":
    test_lightgbm_feature_names_warning()