from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import datetime

import numpy as np
import pandas as pd

from utils.logger import get_logger
from preprocessing.constants import MISSING_CATEGORY_SENTINEL

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
    # Convert any datetime.date or datetime.datetime objects to strings before imputation
    # This prevents type errors in downstream processing
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(100)
        if len(sample) > 0:
            sample_types = set(type(x) for x in sample)
            if datetime.date in sample_types or datetime.datetime in sample_types:
                df[col] = df[col].apply(lambda x: x.isoformat() if isinstance(x, (datetime.date, datetime.datetime)) else x)
                logger.info(f"[imputation] Converted datetime objects to strings in column: {col}")
    
    numeric_fill: Dict[str, Any] = {}
    categorical_fill: Dict[str, Any] = {}

    if cfg.group_by_col and cfg.group_by_col in df.columns:
        group_key = cfg.group_by_col
        grouped = df.groupby(group_key)
        # Compute fill series per (group, column)
        for col in df.select_dtypes(include=[np.number]).columns:
            if cfg.numeric_strategy == "mean":
                vals = grouped[col].median()
                # If a group's median is NaN (all NaN), fallback to group mean
                vals = vals.where(vals.notna(), grouped[col].mean())
            else:
                vals = grouped[col].median()
                # If a group's median is NaN (all NaN), fallback to group mean
                vals = vals.where(vals.notna(), grouped[col].mean())
            # If still NaN (group entirely NaN), fallback to global statistic
            global_fallback = (df[col].mean() if cfg.numeric_strategy == "mean" else df[col].median())
            vals = vals.fillna(global_fallback)
            numeric_fill[col] = vals  # pandas Series indexed by group
        for col in df.select_dtypes(include=["object", "category"]).columns:
            modes = grouped[col].agg(
                lambda s: s.mode().iloc[0]
                if not s.mode().empty
                else (s.dropna().iloc[0] if not s.dropna().empty else MISSING_CATEGORY_SENTINEL)
            )
            # Fallback to global mode/sentinel per gruppi completamente NaN
            global_mode = (df[col].mode().iloc[0] if not df[col].mode().empty else MISSING_CATEGORY_SENTINEL)
            modes = modes.fillna(global_mode)
            categorical_fill[col] = modes
    else:
        for col in df.select_dtypes(include=[np.number]).columns:
            numeric_fill[col] = (df[col].mean() if cfg.numeric_strategy == "mean" else df[col].median())
        for col in df.select_dtypes(include=["object", "category"]).columns:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else MISSING_CATEGORY_SENTINEL
            categorical_fill[col] = mode_val

    return FittedImputers(
        numeric_fill_values=numeric_fill,
        categorical_fill_values=categorical_fill,
        group_by_col=cfg.group_by_col if cfg.group_by_col in df.columns else None,
    )


def _apply_fill_values(df: pd.DataFrame, fitted: FittedImputers) -> pd.DataFrame:
    result = df.copy()
    
    # Convert any datetime.date or datetime.datetime objects to strings
    for col in result.select_dtypes(include=["object"]).columns:
        sample = result[col].dropna().head(100)
        if len(sample) > 0:
            sample_types = set(type(x) for x in sample)
            if datetime.date in sample_types or datetime.datetime in sample_types:
                result[col] = result[col].apply(lambda x: x.isoformat() if isinstance(x, (datetime.date, datetime.datetime)) else x)
                logger.debug(f"[imputation] Converted datetime objects to strings in column: {col}")
    
    if fitted.group_by_col and fitted.group_by_col in result.columns and isinstance(next(iter(fitted.numeric_fill_values.values()), None), pd.Series):
        group_key = fitted.group_by_col
        grouped = result.groupby(group_key)
        # Numeric
        for col, series_vals in fitted.numeric_fill_values.items():
            try:
                # Use per-group value, fallback to series median (which we pre-filled) if missing
                result[col] = grouped[col].transform(lambda s: s.fillna(series_vals.get(s.name)))
            except Exception:
                fallback = series_vals.median() if isinstance(series_vals, pd.Series) else series_vals
                result[col] = result[col].fillna(fallback)
        # Categorical
        for col, series_vals in fitted.categorical_fill_values.items():
            try:
                result[col] = grouped[col].transform(
                    lambda s: s.fillna(series_vals.get(s.name, MISSING_CATEGORY_SENTINEL)).infer_objects(copy=False)
                )
            except Exception:
                fallback = (
                    series_vals.mode().iloc[0]
                    if isinstance(series_vals, pd.Series) and not series_vals.mode().empty
                    else MISSING_CATEGORY_SENTINEL
                )
                if fallback is None:
                    fallback = MISSING_CATEGORY_SENTINEL
                result[col] = result[col].fillna(fallback).infer_objects(copy=False)
    else:
        for col, val in fitted.numeric_fill_values.items():
            result[col] = result[col].fillna(val)
        for col, val in fitted.categorical_fill_values.items():
            fill_val = val if val is not None else MISSING_CATEGORY_SENTINEL
            result[col] = result[col].fillna(fill_val).infer_objects(copy=False)
    return result


# Train/test safe API

def fit_imputers(df_train: pd.DataFrame, cfg: ImputationConfig) -> FittedImputers:
    return _fit_fill_values(df_train, cfg)


def transform_with_imputers(df: pd.DataFrame, fitted: FittedImputers) -> pd.DataFrame:
    return _apply_fill_values(df, fitted)