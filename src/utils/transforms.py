"""Target transformation utilities with inverse transformation support."""
from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox

from utils.logger import get_logger

logger = get_logger(__name__)


def apply_target_transform(
    y: np.ndarray | pd.Series,
    transform_type: str = "log",
    **kwargs
) -> Tuple[np.ndarray | pd.Series, Dict[str, Any]]:
    """
    Apply transformation to target variable.
    
    Args:
        y: Target values
        transform_type: Type of transformation ('none', 'log', 'log10', 'sqrt', 'boxcox', 'yeojohnson')
        **kwargs: Additional parameters (e.g., log10_offset)
    
    Returns:
        Tuple of (transformed values, metadata dict)
    """
    is_series = isinstance(y, pd.Series)
    if is_series:
        index = y.index
        y_arr = y.values
    else:
        y_arr = np.asarray(y)
    
    metadata = {"transform": transform_type}
    
    if transform_type == "none":
        y_transformed = y_arr
    
    elif transform_type == "log":
        # log1p handles zeros
        y_transformed = np.log1p(y_arr)
        logger.info("Applied log1p transformation to target")
    
    elif transform_type == "log10":
        offset = float(kwargs.get("log10_offset", 1.0))
        y_transformed = np.log10(y_arr + offset)
        metadata["log10_offset"] = offset
        logger.info(f"Applied log10 transformation to target (offset={offset})")
    
    elif transform_type == "sqrt":
        if np.any(y_arr < 0):
            logger.warning("Negative values found for sqrt transform, clipping to 0")
            y_arr = np.maximum(y_arr, 0)
        y_transformed = np.sqrt(y_arr)
        logger.info("Applied sqrt transformation to target")
    
    elif transform_type == "boxcox":
        if np.any(y_arr <= 0):
            logger.warning("Non-positive values found for Box-Cox, adding offset")
            offset = abs(y_arr.min()) + 1
            y_arr = y_arr + offset
            metadata["boxcox_offset"] = float(offset)
        
        y_transformed, lmbda = stats.boxcox(y_arr)
        metadata["boxcox_lambda"] = float(lmbda)
        logger.info(f"Applied Box-Cox transformation to target (lambda={lmbda:.4f})")
    
    elif transform_type == "yeojohnson":
        y_transformed, lmbda = stats.yeojohnson(y_arr)
        metadata["yeojohnson_lambda"] = float(lmbda)
        logger.info(f"Applied Yeo-Johnson transformation to target (lambda={lmbda:.4f})")
    
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    if is_series:
        return pd.Series(y_transformed, index=index), metadata
    else:
        return y_transformed, metadata


def inverse_transform_target(
    y_transformed: np.ndarray | pd.Series,
    transform_meta: Dict[str, Any]
) -> np.ndarray | pd.Series:
    """
    Apply inverse transformation to bring predictions back to original scale.
    
    Args:
        y_transformed: Transformed target values (predictions in transformed space)
        transform_meta: Metadata dict with transform type and parameters
    
    Returns:
        Target values in original scale
    """
    transform_type = transform_meta.get("transform", "none")
    
    is_series = isinstance(y_transformed, pd.Series)
    if is_series:
        index = y_transformed.index
        y_arr = y_transformed.values
    else:
        y_arr = np.asarray(y_transformed)
    
    if transform_type == "none":
        y_original = y_arr
    
    elif transform_type == "log":
        # Inverse of log1p is expm1
        y_original = np.expm1(y_arr)
    
    elif transform_type == "log10":
        # Inverse of log10(y + offset) is 10^y - offset
        offset = transform_meta.get("log10_offset", 1.0)
        y_original = np.power(10, y_arr) - offset
    
    elif transform_type == "sqrt":
        # Inverse of sqrt is square
        y_original = np.square(y_arr)
    
    elif transform_type == "boxcox":
        # Inverse Box-Cox transformation
        lmbda = transform_meta.get("boxcox_lambda")
        if lmbda is None:
            raise ValueError("Box-Cox lambda not found in transform_meta")
        y_original = inv_boxcox(y_arr, lmbda)
        
        # Apply offset if present
        offset = transform_meta.get("boxcox_offset", 0)
        if offset > 0:
            y_original = y_original - offset
    
    elif transform_type == "yeojohnson":
        # Inverse Yeo-Johnson transformation
        lmbda = transform_meta.get("yeojohnson_lambda")
        if lmbda is None:
            raise ValueError("Yeo-Johnson lambda not found in transform_meta")
        y_original = inverse_yeojohnson(y_arr, lmbda)
    
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    if is_series:
        return pd.Series(y_original, index=index)
    else:
        return y_original


def inverse_yeojohnson(y: np.ndarray, lmbda: float) -> np.ndarray:
    """
    Inverse Yeo-Johnson transformation.
    
    Args:
        y: Transformed values
        lmbda: Lambda parameter from forward transform
    
    Returns:
        Original values
    """
    x = np.zeros_like(y)
    
    # Case 1: y >= 0 and lambda != 0
    mask1 = (y >= 0) & (lmbda != 0)
    if np.any(mask1):
        x[mask1] = np.power(y[mask1] * lmbda + 1, 1 / lmbda) - 1
    
    # Case 2: y >= 0 and lambda == 0
    mask2 = (y >= 0) & (lmbda == 0)
    if np.any(mask2):
        x[mask2] = np.expm1(y[mask2])
    
    # Case 3: y < 0 and lambda != 2
    mask3 = (y < 0) & (lmbda != 2)
    if np.any(mask3):
        x[mask3] = 1 - np.power(-(2 - lmbda) * y[mask3] + 1, 1 / (2 - lmbda))
    
    # Case 4: y < 0 and lambda == 2
    mask4 = (y < 0) & (lmbda == 2)
    if np.any(mask4):
        x[mask4] = -np.expm1(-y[mask4])
    
    return x


def get_transform_name(transform_meta: Dict[str, Any]) -> str:
    """Get human-readable name of transformation."""
    transform_type = transform_meta.get("transform", "none")
    
    if transform_type == "none":
        return "No transformation"
    elif transform_type == "log":
        return "Log1p"
    elif transform_type == "log10":
        offset = transform_meta.get("log10_offset", 1.0)
        return f"Log10 (offset={offset})"
    elif transform_type == "sqrt":
        return "Square root"
    elif transform_type == "boxcox":
        lmbda = transform_meta.get("boxcox_lambda", "?")
        return f"Box-Cox (λ={lmbda:.4f})" if lmbda != "?" else "Box-Cox"
    elif transform_type == "yeojohnson":
        lmbda = transform_meta.get("yeojohnson_lambda", "?")
        return f"Yeo-Johnson (λ={lmbda:.4f})" if lmbda != "?" else "Yeo-Johnson"
    else:
        return f"Unknown ({transform_type})"
