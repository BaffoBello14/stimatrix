"""Target variable transformations with inverse transformation support.

Supported transformations:
- none: No transformation
- log: log1p (log(1+x))
- log10: log10(x + offset)
- sqrt: Square root
- boxcox: Box-Cox transformation (requires y > 0)
- yeojohnson: Yeo-Johnson transformation (works with any y)
"""
from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import get_logger

logger = get_logger(__name__)


def boxcox_transform(values: np.ndarray, lmbda: float, shift: float = 0.0) -> np.ndarray:
    """Apply Box-Cox transform with explicit lambda and shift."""
    shifted = values + shift
    if lmbda == 0:
        return np.log(shifted)
    return (np.power(shifted, lmbda) - 1) / lmbda


def yeojohnson_transform(values: np.ndarray, lmbda: float) -> np.ndarray:
    """Apply Yeo-Johnson transform with explicit lambda."""
    mask_pos = values >= 0
    out = np.zeros_like(values, dtype=float)
    if mask_pos.any():
        pos = values[mask_pos]
        if lmbda == 0:
            out[mask_pos] = np.log(pos + 1)
        else:
            out[mask_pos] = ((pos + 1) ** lmbda - 1) / lmbda
    if (~mask_pos).any():
        neg = values[~mask_pos]
        if lmbda == 2:
            out[~mask_pos] = -np.log(1 - neg)
        else:
            out[~mask_pos] = -(((1 - neg) ** (2 - lmbda)) - 1) / (2 - lmbda)
    return out


def apply_target_transform(
    y: pd.Series | np.ndarray,
    transform_type: str = "none",
    **kwargs
) -> Tuple[pd.Series | np.ndarray, Dict[str, Any]]:
    """
    Apply transformation to target variable.
    
    Args:
        y: Target variable
        transform_type: Type of transformation
            - 'none': No transformation
            - 'log': log1p (handles zeros)
            - 'log10': log10(y + offset)
            - 'sqrt': Square root
            - 'boxcox': Box-Cox (auto-estimates lambda)
            - 'yeojohnson': Yeo-Johnson (auto-estimates lambda)
        **kwargs: Additional parameters (e.g., log10_offset)
    
    Returns:
        Tuple of (transformed y, metadata dict)
    """
    is_series = isinstance(y, pd.Series)
    y_values = y.values if is_series else np.asarray(y)
    index = y.index if is_series else None
    
    metadata: Dict[str, Any] = {"transform": transform_type}
    
    if transform_type == "none":
        y_transformed = y_values
        logger.info("✨ Target: No transformation applied")
    
    elif transform_type == "log":
        # log1p handles zeros automatically
        y_transformed = np.log1p(y_values)
        logger.info(f"✨ Target: log1p transformation applied (y → log(1+y))")
    
    elif transform_type == "log10":
        offset = float(kwargs.get("log10_offset", 1.0))
        metadata["log10_offset"] = offset
        y_transformed = np.log10(y_values + offset)
        logger.info(f"✨ Target: log10 transformation applied (y → log10(y+{offset}))")
    
    elif transform_type == "sqrt":
        # Check for negative values
        if np.any(y_values < 0):
            logger.warning("⚠️  Target contains negative values. sqrt will produce NaN for negatives.")
        y_transformed = np.sqrt(np.maximum(y_values, 0))
        logger.info("✨ Target: sqrt transformation applied")
    
    elif transform_type == "boxcox":
        original = y_values.copy()
        shift = 0.0
        if np.any(original <= 0):
            min_val = original.min()
            shift = abs(min_val) + 1.0
            logger.warning(f"⚠️  Box-Cox requires y > 0. Shifting by {shift:.2f}")
        shifted = original + shift
        _, lambda_fitted = stats.boxcox(shifted)
        metadata["lambda"] = float(lambda_fitted)
        metadata["shift"] = float(shift)
        y_transformed = boxcox_transform(original, lambda_fitted, shift)
        logger.info(
            "✨ Target: Box-Cox transformation applied (λ=%.4f, shift=%.4f)",
            lambda_fitted,
            shift,
        )

    elif transform_type == "yeojohnson":
        original = y_values.copy()
        _, lambda_fitted = stats.yeojohnson(original)
        metadata["lambda"] = float(lambda_fitted)
        metadata["shift"] = 0.0
        y_transformed = yeojohnson_transform(original, lambda_fitted)
        logger.info("✨ Target: Yeo-Johnson transformation applied (λ=%.4f)", lambda_fitted)
    
    else:
        raise ValueError(
            f"Unknown transform type: {transform_type}. "
            f"Valid options: none, log, log10, sqrt, boxcox, yeojohnson"
        )
    
    # Return in same format as input
    if is_series:
        return pd.Series(y_transformed, index=index, name=y.name), metadata
    else:
        return y_transformed, metadata


def inverse_target_transform(
    y_transformed: pd.Series | np.ndarray,
    metadata: Dict[str, Any]
) -> pd.Series | np.ndarray:
    """
    Apply inverse transformation to bring predictions back to original scale.
    
    Args:
        y_transformed: Transformed target values (predictions)
        metadata: Metadata dict from apply_target_transform()
    
    Returns:
        Target values in original scale
    """
    transform_type = metadata.get("transform", "none")
    
    is_series = isinstance(y_transformed, pd.Series)
    y_values = y_transformed.values if is_series else np.asarray(y_transformed)
    index = y_transformed.index if is_series else None
    
    if transform_type == "none":
        y_original = y_values
    
    elif transform_type == "log":
        # Inverse of log1p is expm1
        y_original = np.expm1(y_values)
    
    elif transform_type == "log10":
        offset = metadata.get("log10_offset", 1.0)
        y_original = np.power(10, y_values) - offset
    
    elif transform_type == "sqrt":
        # Inverse of sqrt is square
        y_original = np.square(y_values)
    
    elif transform_type == "boxcox":
        lambda_val = metadata.get("lambda", metadata.get("boxcox_lambda"))
        if lambda_val is None:
            raise ValueError("Box-Cox lambda not found in metadata")
        shift = metadata.get("shift", metadata.get("boxcox_shift", 0.0))
        if lambda_val == 0:
            y_original = np.exp(y_values) - shift
        else:
            y_original = np.power(lambda_val * y_values + 1.0, 1.0 / lambda_val) - shift

    elif transform_type == "yeojohnson":
        lambda_val = metadata.get("lambda", metadata.get("yeojohnson_lambda"))
        if lambda_val is None:
            raise ValueError("Yeo-Johnson lambda not found in metadata")

        y_original = _inverse_yeojohnson(y_values, lambda_val)
    
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    # Return in same format as input
    if is_series:
        return pd.Series(y_original, index=index, name=y_transformed.name)
    else:
        return y_original


def _inverse_yeojohnson(y: np.ndarray, lmbda: float) -> np.ndarray:
    """
    Inverse Yeo-Johnson transformation.
    
    Reference: Yeo & Johnson (2000)
    """
    x = np.zeros_like(y, dtype=float)
    
    # Case 1: y >= 0 and lambda != 0
    mask1 = (y >= 0) & (lmbda != 0)
    if np.any(mask1):
        x[mask1] = np.power(y[mask1] * lmbda + 1, 1.0 / lmbda) - 1
    
    # Case 2: y >= 0 and lambda == 0
    mask2 = (y >= 0) & (lmbda == 0)
    if np.any(mask2):
        x[mask2] = np.expm1(y[mask2])
    
    # Case 3: y < 0 and lambda != 2
    mask3 = (y < 0) & (lmbda != 2)
    if np.any(mask3):
        x[mask3] = 1 - np.power(-(2 - lmbda) * y[mask3] + 1, 1.0 / (2 - lmbda))
    
    # Case 4: y < 0 and lambda == 2
    mask4 = (y < 0) & (lmbda == 2)
    if np.any(mask4):
        x[mask4] = -np.expm1(-y[mask4])
    
    return x


def get_transform_name(metadata: Dict[str, Any]) -> str:
    """Get human-readable name of transformation."""
    transform_type = metadata.get("transform", "none")
    
    if transform_type == "none":
        return "No transformation"
    elif transform_type == "log":
        return "Log1p"
    elif transform_type == "log10":
        offset = metadata.get("log10_offset", 1.0)
        return f"Log10 (offset={offset})"
    elif transform_type == "sqrt":
        return "Square root"
    elif transform_type == "boxcox":
        lmbda = metadata.get("boxcox_lambda", "?")
        return f"Box-Cox (λ={lmbda:.4f})" if lmbda != "?" else "Box-Cox"
    elif transform_type == "yeojohnson":
        lmbda = metadata.get("yeojohnson_lambda", "?")
        return f"Yeo-Johnson (λ={lmbda:.4f})" if lmbda != "?" else "Yeo-Johnson"
    else:
        return f"Unknown ({transform_type})"


