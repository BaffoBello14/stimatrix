from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OutlierConfig:
    method: str = "iqr"  # options: iqr, zscore, iso_forest
    z_thresh: float = 4.0
    iqr_factor: float = 1.5
    iso_forest_contamination: float = 0.02
    group_by_col: Optional[str] = None  # e.g., 'AI_IdCategoriaCatastale'


def _inliers_iqr(values: pd.Series, iqr_factor: float) -> pd.Series:
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    low = q1 - iqr_factor * iqr
    high = q3 + iqr_factor * iqr
    return (values >= low) & (values <= high)


def _inliers_zscore(values: pd.Series, z_thresh: float) -> pd.Series:
    mean = values.mean()
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(True, index=values.index)
    z = (values - mean) / std
    return z.abs() <= z_thresh


def _inliers_isoforest(values: pd.Series, contamination: float) -> pd.Series:
    valid = values.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return pd.Series(True, index=values.index)
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(valid.to_frame())  # 1 for inliers, -1 for outliers
    mask_valid = preds == 1
    mask = pd.Series(True, index=values.index)
    mask.loc[valid.index] = mask_valid
    return mask


def detect_outliers(
    df: pd.DataFrame,
    target_col: str,
    config: OutlierConfig,
) -> pd.Series:
    logger.info(
        f"Outlier detection method={config.method}, group_by={config.group_by_col}, target={target_col}"
    )
    if config.group_by_col and config.group_by_col in df.columns:
        masks: List[pd.Series] = []
        for cat, group_idx in df.groupby(config.group_by_col).groups.items():
            values = df.loc[group_idx, target_col]
            if config.method == "iqr":
                inliers = _inliers_iqr(values, config.iqr_factor)
            elif config.method == "zscore":
                inliers = _inliers_zscore(values, config.z_thresh)
            elif config.method == "iso_forest":
                inliers = _inliers_isoforest(values, config.iso_forest_contamination)
            else:
                inliers = pd.Series(True, index=values.index)
            masks.append(inliers.reindex(df.index, fill_value=False))
        # Combine per-group masks
        combined = pd.Series(False, index=df.index)
        for m in masks:
            combined = combined | m
        return combined
    else:
        values = df[target_col]
        if config.method == "iqr":
            return _inliers_iqr(values, config.iqr_factor)
        if config.method == "zscore":
            return _inliers_zscore(values, config.z_thresh)
        if config.method == "iso_forest":
            return _inliers_isoforest(values, config.iso_forest_contamination)
        return pd.Series(True, index=df.index)