"""Advanced multi-strategy categorical encoding based on cardinality."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder as CategoryTargetEncoder
from sklearn.preprocessing import OneHotEncoder

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EncodingPlan:
    """Plan for encoding categorical variables based on cardinality."""
    one_hot_cols: List[str]
    target_cols: List[str]
    frequency_cols: List[str]
    ordinal_cols: List[str]
    drop_cols: List[str]  # High cardinality columns


@dataclass
class FittedEncoders:
    """Container for fitted encoders."""
    one_hot: OneHotEncoder | None
    target: Dict[str, CategoryTargetEncoder]  # Per-column target encoders
    frequency: Dict[str, Dict[str, float]]    # Per-column frequency maps
    ordinal: Dict[str, OrdinalEncoder]        # Per-column ordinal encoders
    one_hot_input_cols: List[str]
    target_input_cols: List[str]
    frequency_input_cols: List[str]
    ordinal_input_cols: List[str]


def plan_encodings(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> EncodingPlan:
    """
    Plan encoding strategy for each categorical column based on cardinality.
    
    Strategy:
    - ≤one_hot_max unique values → One-Hot Encoding
    - target_encoding_range → Target Encoding (with smoothing)
    - frequency_encoding_range → Frequency Encoding
    - ordinal_encoding_range → Ordinal Encoding
    - >drop_above → DROP (too high cardinality)
    
    Args:
        df: Input DataFrame
        config: Configuration dict with encoding parameters
    
    Returns:
        EncodingPlan with columns assigned to each strategy
    """
    enc_cfg = config.get('encoding', {})
    
    one_hot_max = int(enc_cfg.get('one_hot_max', 10))
    target_range = enc_cfg.get('target_encoding_range', [11, 30])
    freq_range = enc_cfg.get('frequency_encoding_range', [31, 100])
    ord_range = enc_cfg.get('ordinal_encoding_range', [101, 200])
    drop_above = int(enc_cfg.get('drop_above', 200))
    
    one_hot_cols = []
    target_cols = []
    frequency_cols = []
    ordinal_cols = []
    drop_cols = []
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        nunique = df[col].nunique(dropna=True)
        
        if nunique <= one_hot_max:
            one_hot_cols.append(col)
        elif target_range[0] <= nunique <= target_range[1]:
            target_cols.append(col)
        elif freq_range[0] <= nunique <= freq_range[1]:
            frequency_cols.append(col)
        elif ord_range[0] <= nunique <= ord_range[1]:
            ordinal_cols.append(col)
        elif nunique > drop_above:
            drop_cols.append(col)
        else:
            # Fallback to ordinal for edge cases
            ordinal_cols.append(col)
    
    logger.info(
        f"Encoding plan: "
        f"OneHot={len(one_hot_cols)} cols (≤{one_hot_max} unique), "
        f"Target={len(target_cols)} cols ({target_range[0]}-{target_range[1]} unique), "
        f"Frequency={len(frequency_cols)} cols ({freq_range[0]}-{freq_range[1]} unique), "
        f"Ordinal={len(ordinal_cols)} cols ({ord_range[0]}-{ord_range[1]} unique), "
        f"Drop={len(drop_cols)} cols (>{drop_above} unique)"
    )
    
    return EncodingPlan(
        one_hot_cols=one_hot_cols,
        target_cols=target_cols,
        frequency_cols=frequency_cols,
        ordinal_cols=ordinal_cols,
        drop_cols=drop_cols
    )


def fit_apply_encoders(
    X_train: pd.DataFrame,
    y_train: pd.Series | None,
    plan: EncodingPlan,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, FittedEncoders]:
    """
    Fit and apply encoders to training data.
    
    Args:
        X_train: Training features
        y_train: Training target (required for target encoding, None otherwise)
        plan: Encoding plan from plan_encodings()
        config: Configuration dict
    
    Returns:
        Tuple of (encoded DataFrame, fitted encoders)
    """
    result = X_train.copy()
    
    enc_one_hot = None
    enc_target = {}
    enc_frequency = {}
    enc_ordinal = {}
    
    # 1. Drop high cardinality columns
    if plan.drop_cols:
        result = result.drop(columns=plan.drop_cols, errors='ignore')
        logger.info(f"✂️  Dropped {len(plan.drop_cols)} high-cardinality columns (>{config.get('encoding', {}).get('drop_above', 200)} unique)")
    
    # 2. One-Hot Encoding
    if plan.one_hot_cols:
        valid_cols = [c for c in plan.one_hot_cols if c in result.columns]
        if valid_cols:
            enc_one_hot = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',
                dtype=np.float64
            )
            ohe_arr = enc_one_hot.fit_transform(result[valid_cols])
            ohe_cols = enc_one_hot.get_feature_names_out(valid_cols)
            ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=result.index)
            result = pd.concat([result.drop(columns=valid_cols), ohe_df], axis=1)
            logger.info(f"🔢 One-Hot Encoded {len(valid_cols)} columns → {len(ohe_cols)} binary features")
    
    # 3. Target Encoding (requires y_train)
    if plan.target_cols and y_train is not None:
        enc_cfg = config.get('encoding', {}).get('target_encoder', {})
        smoothing = float(enc_cfg.get('smoothing', 1.0))
        min_samples = int(enc_cfg.get('min_samples_leaf', 1))
        
        for col in plan.target_cols:
            if col not in result.columns:
                continue
            
            try:
                enc = CategoryTargetEncoder(
                    cols=[col],
                    smoothing=smoothing,
                    min_samples_leaf=min_samples,
                    handle_unknown='value',
                    handle_missing='value'
                )
                
                transformed = enc.fit_transform(result[[col]], y_train)
                new_col = f'{col}__target'
                result[new_col] = transformed[col].astype(float)
                enc_target[col] = enc
                result = result.drop(columns=[col])
            except Exception as e:
                logger.warning(f"⚠️  Target encoding failed for {col}: {e}. Using frequency encoding instead.")
                # Fallback to frequency
                freq = result[col].value_counts(normalize=True).to_dict()
                new_col = f'{col}__freq'
                result[new_col] = result[col].map(freq).fillna(0).astype(float)
                enc_frequency[col] = freq
                result = result.drop(columns=[col])
        
        if enc_target:
            logger.info(f"🎯 Target Encoded {len(enc_target)} columns (smoothing={smoothing})")
    elif plan.target_cols and y_train is None:
        # Fallback to frequency if no target provided
        logger.warning(f"⚠️  Target encoding requested for {len(plan.target_cols)} columns but no y_train provided. Using frequency encoding.")
        for col in plan.target_cols:
            if col not in result.columns:
                continue
            freq = result[col].value_counts(normalize=True).to_dict()
            new_col = f'{col}__freq'
            result[new_col] = result[col].map(freq).fillna(0).astype(float)
            enc_frequency[col] = freq
            result = result.drop(columns=[col])
    
    # 4. Frequency Encoding
    if plan.frequency_cols:
        for col in plan.frequency_cols:
            if col not in result.columns:
                continue
            
            # Calculate frequency (normalized counts)
            freq = result[col].value_counts(normalize=True).to_dict()
            new_col = f'{col}__freq'
            result[new_col] = result[col].map(freq).fillna(0).astype(float)
            enc_frequency[col] = freq
            result = result.drop(columns=[col])
        
        logger.info(f"📊 Frequency Encoded {len(plan.frequency_cols)} columns")
    
    # 5. Ordinal Encoding
    if plan.ordinal_cols:
        for col in plan.ordinal_cols:
            if col not in result.columns:
                continue
            
            try:
                enc = OrdinalEncoder(
                    cols=[col],
                    handle_unknown='value',
                    handle_missing='return_nan'
                )
                transformed = enc.fit_transform(result[[col]])
                new_col = f'{col}__ord'
                result[new_col] = transformed[col].astype(float)
                enc_ordinal[col] = enc
                result = result.drop(columns=[col])
            except Exception as e:
                logger.warning(f"⚠️  Ordinal encoding failed for {col}: {e}. Dropping column.")
                result = result.drop(columns=[col])
        
        if enc_ordinal:
            logger.info(f"🔢 Ordinal Encoded {len(enc_ordinal)} columns")
    
    fitted = FittedEncoders(
        one_hot=enc_one_hot,
        target=enc_target,
        frequency=enc_frequency,
        ordinal=enc_ordinal,
        one_hot_input_cols=plan.one_hot_cols,
        target_input_cols=plan.target_cols,
        frequency_input_cols=plan.frequency_cols,
        ordinal_input_cols=plan.ordinal_cols
    )
    
    return result, fitted


def transform_with_encoders(
    X: pd.DataFrame,
    fitted: FittedEncoders
) -> pd.DataFrame:
    """
    Transform data using fitted encoders (for validation/test sets).
    
    Args:
        X: Input features
        fitted: Fitted encoders from fit_apply_encoders()
    
    Returns:
        Encoded DataFrame
    """
    result = X.copy()
    
    # Drop high cardinality columns (same as in training)
    all_encoded_cols = (
        fitted.one_hot_input_cols +
        fitted.target_input_cols +
        fitted.frequency_input_cols +
        fitted.ordinal_input_cols
    )
    other_cat_cols = result.select_dtypes(include=['object', 'category']).columns
    to_drop = [c for c in other_cat_cols if c not in all_encoded_cols]
    if to_drop:
        result = result.drop(columns=to_drop, errors='ignore')
    
    # 1. One-Hot Encoding
    if fitted.one_hot is not None and fitted.one_hot_input_cols:
        # Handle missing columns (fill with None for unknown category)
        for col in fitted.one_hot_input_cols:
            if col not in result.columns:
                result[col] = None
        
        ohe_arr = fitted.one_hot.transform(result[fitted.one_hot_input_cols])
        ohe_cols = fitted.one_hot.get_feature_names_out(fitted.one_hot_input_cols)
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=result.index)
        result = pd.concat([result.drop(columns=fitted.one_hot_input_cols, errors='ignore'), ohe_df], axis=1)
    
    # 2. Target Encoding
    for col, enc in fitted.target.items():
        if col in result.columns:
            # Transform without target (uses fitted mapping)
            try:
                transformed = enc.transform(result[[col]])
                new_col = f'{col}__target'
                result[new_col] = transformed[col].astype(float)
                result = result.drop(columns=[col])
            except Exception as e:
                logger.warning(f"⚠️  Target encoding transform failed for {col}: {e}. Using global mean.")
                new_col = f'{col}__target'
                result[new_col] = 0.0  # Neutral value
                result = result.drop(columns=[col], errors='ignore')
        else:
            # Missing column → use global mean (neutral value)
            new_col = f'{col}__target'
            result[new_col] = 0.0
    
    # 3. Frequency Encoding
    for col, freq_map in fitted.frequency.items():
        if col in result.columns:
            new_col = f'{col}__freq'
            result[new_col] = result[col].map(freq_map).fillna(0).astype(float)
            result = result.drop(columns=[col])
        else:
            # Missing column → 0 frequency
            new_col = f'{col}__freq'
            result[new_col] = 0.0
    
    # 4. Ordinal Encoding
    for col, enc in fitted.ordinal.items():
        if col in result.columns:
            try:
                transformed = enc.transform(result[[col]])
                new_col = f'{col}__ord'
                result[new_col] = transformed[col].astype(float).fillna(-1)
                result = result.drop(columns=[col])
            except Exception as e:
                logger.warning(f"⚠️  Ordinal encoding transform failed for {col}: {e}. Using -1.")
                new_col = f'{col}__ord'
                result[new_col] = -1.0
                result = result.drop(columns=[col], errors='ignore')
        else:
            # Missing column → -1 (unknown)
            new_col = f'{col}__ord'
            result[new_col] = -1.0
    
    return result
