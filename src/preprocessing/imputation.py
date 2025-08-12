from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ImputationConfig:
    numeric_strategy: str = "median"  # median or mean
    categorical_strategy: str = "most_frequent"
    group_by_col: Optional[str] = None  # e.g., AI_IdCategoriaCatastale


@dataclass
class FittedImputers:
    numeric_fill_values: Dict[str, Any]
    categorical_fill_values: Dict[str, Any]
    group_by_col: Optional[str]


def _fit_fill_values(df: pd.DataFrame, cfg: ImputationConfig) -> FittedImputers:
    numeric_fill: Dict[str, Any] = {}
    categorical_fill: Dict[str, Any] = {}

    if cfg.group_by_col and cfg.group_by_col in df.columns:
        group_key = cfg.group_by_col
        grouped = df.groupby(group_key)
        # Compute fill series per (group, column)
        for col in df.select_dtypes(include=[np.number]).columns:
            if cfg.numeric_strategy == "mean":
                vals = grouped[col].median() if grouped[col].median().notna().any() else grouped[col].mean()
            else:
                vals = grouped[col].median()
            numeric_fill[col] = vals  # pandas Series indexed by group
        for col in df.select_dtypes(include=["object", "category"]).columns:
            modes = grouped[col].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else (s.dropna().iloc[0] if not s.dropna().empty else "UNKNOWN"))
            categorical_fill[col] = modes
    else:
        for col in df.select_dtypes(include=[np.number]).columns:
            numeric_fill[col] = (df[col].mean() if cfg.numeric_strategy == "mean" else df[col].median())
        for col in df.select_dtypes(include=["object", "category"]).columns:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "UNKNOWN"
            categorical_fill[col] = mode_val

    return FittedImputers(
        numeric_fill_values=numeric_fill,
        categorical_fill_values=categorical_fill,
        group_by_col=cfg.group_by_col if cfg.group_by_col in df.columns else None,
    )


def _apply_fill_values(df: pd.DataFrame, fitted: FittedImputers) -> pd.DataFrame:
    result = df.copy()
    if fitted.group_by_col and fitted.group_by_col in result.columns and isinstance(next(iter(fitted.numeric_fill_values.values()), None), pd.Series):
        group_key = fitted.group_by_col
        grouped = result.groupby(group_key)
        # Numeric
        for col, series_vals in fitted.numeric_fill_values.items():
            try:
                result[col] = grouped[col].transform(lambda s: s.fillna(series_vals.get(s.name)))
            except Exception:
                # Fallback to global median if any issue
                fallback = series_vals.median() if isinstance(series_vals, pd.Series) else series_vals
                result[col] = result[col].fillna(fallback)
        # Categorical
        for col, series_vals in fitted.categorical_fill_values.items():
            try:
                result[col] = grouped[col].transform(lambda s: s.fillna(series_vals.get(s.name, "UNKNOWN")))
            except Exception:
                fallback = series_vals.mode().iloc[0] if isinstance(series_vals, pd.Series) and not series_vals.mode().empty else "UNKNOWN"
                if fallback is None:
                    fallback = "UNKNOWN"
                result[col] = result[col].fillna(fallback)
    else:
        for col, val in fitted.numeric_fill_values.items():
            result[col] = result[col].fillna(val)
        for col, val in fitted.categorical_fill_values.items():
            fill_val = val if val is not None else "UNKNOWN"
            result[col] = result[col].fillna(fill_val)
    return result


# Backwards-compatible simple API (not train/test safe)
def impute_missing(df: pd.DataFrame, cfg: ImputationConfig) -> pd.DataFrame:
    fitted = _fit_fill_values(df, cfg)
    return _apply_fill_values(df, fitted)


# Train/test safe API

def fit_imputers(df_train: pd.DataFrame, cfg: ImputationConfig) -> FittedImputers:
    return _fit_fill_values(df_train, cfg)


def transform_with_imputers(df: pd.DataFrame, fitted: FittedImputers) -> pd.DataFrame:
    return _apply_fill_values(df, fitted)