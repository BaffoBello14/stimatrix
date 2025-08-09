from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from category_encoders.ordinal import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EncodingPlan:
    one_hot_cols: List[str]
    ordinal_cols: List[str]


@dataclass
class FittedEncoders:
    one_hot: OneHotEncoder | None
    ordinal: Dict[str, OrdinalEncoder]
    one_hot_input_cols: List[str]


def plan_encodings(df: pd.DataFrame, max_ohe_cardinality: int = 12) -> EncodingPlan:
    one_hot_cols: List[str] = []
    ordinal_cols: List[str] = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        nunique = df[col].nunique(dropna=True)
        if nunique <= max_ohe_cardinality:
            one_hot_cols.append(col)
        else:
            # If column looks numeric-like strings, we can ordinal-encode by sorted unique
            sample = df[col].dropna().astype(str).head(50)
            looks_numeric = sample.apply(lambda s: s.replace(',', '.').replace(' ', '').lstrip('+-').replace('.', '', 1).isdigit()).mean() > 0.7
            if looks_numeric:
                ordinal_cols.append(col)
            else:
                # fallback: treat as ordinal with arbitrary order (stable)
                ordinal_cols.append(col)
    return EncodingPlan(one_hot_cols=one_hot_cols, ordinal_cols=ordinal_cols)


def fit_apply_encoders(df: pd.DataFrame, plan: EncodingPlan) -> Tuple[pd.DataFrame, FittedEncoders, List[str]]:
    cols_to_drop: List[str] = []
    enc_one_hot: OneHotEncoder | None = None
    enc_ordinals: Dict[str, OrdinalEncoder] = {}

    result = df.copy()

    # Ordinal encoding per column (handles NaN with -1)
    for col in plan.ordinal_cols:
        enc = OrdinalEncoder(cols=[col], handle_unknown="impute", handle_missing="return_nan")
        transformed = enc.fit_transform(result[[col]])
        new_col = f"{col}__ord"
        result[new_col] = transformed[col].astype(float)
        enc_ordinals[col] = enc
        cols_to_drop.append(col)

    # OneHot on all OHE columns jointly to avoid duplicate categories
    if plan.one_hot_cols:
        enc_one_hot = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float64)
        ohe_arr = enc_one_hot.fit_transform(result[plan.one_hot_cols])
        ohe_cols = enc_one_hot.get_feature_names_out(plan.one_hot_cols)
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=result.index)
        result = pd.concat([result.drop(columns=plan.one_hot_cols), ohe_df], axis=1)
        cols_to_drop.extend([c for c in plan.one_hot_cols if c in result.columns])

    return result, FittedEncoders(one_hot=enc_one_hot, ordinal=enc_ordinals, one_hot_input_cols=plan.one_hot_cols), cols_to_drop


def transform_with_encoders(df: pd.DataFrame, fitted: FittedEncoders) -> pd.DataFrame:
    result = df.copy()

    # Ordinal
    for col, enc in fitted.ordinal.items():
        if col in result.columns:
            transformed = enc.transform(result[[col]])
            new_col = f"{col}__ord"
            result[new_col] = transformed[col].astype(float)
            result = result.drop(columns=[col])
        else:
            # Column missing -> create placeholder with NaN
            result[f"{col}__ord"] = np.nan

    # OneHot
    if fitted.one_hot is not None and fitted.one_hot_input_cols:
        # Ensure all expected input columns exist
        for col in fitted.one_hot_input_cols:
            if col not in result.columns:
                result[col] = None
        ohe_arr = fitted.one_hot.transform(result[fitted.one_hot_input_cols])
        ohe_cols = fitted.one_hot.get_feature_names_out(fitted.one_hot_input_cols)
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=result.index)
        result = pd.concat([result.drop(columns=fitted.one_hot_input_cols, errors='ignore'), ohe_df], axis=1)

    return result