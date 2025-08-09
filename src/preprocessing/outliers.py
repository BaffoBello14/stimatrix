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
    method: str = "iqr"  # options: iqr, zscore, iso_forest, ensemble
    z_thresh: float = 4.0
    iqr_factor: float = 1.5
    iso_forest_contamination: float = 0.02
    group_by_col: Optional[str] = None  # e.g., 'AI_IdCategoriaCatastale'
    min_group_size: int = 30
    fallback_strategy: str = "skip"  # 'skip' or 'global'


def _inliers_iqr(values: pd.Series, iqr_factor: float) -> pd.Series:
    v = values.astype(float)
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return pd.Series(True, index=v.index)
    low = q1 - iqr_factor * iqr
    high = q3 + iqr_factor * iqr
    return (v >= low) & (v <= high)


def _inliers_zscore(values: pd.Series, z_thresh: float) -> pd.Series:
    v = values.astype(float)
    mean = v.mean()
    std = v.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(True, index=v.index)
    z = (v - mean) / std
    return z.abs() <= z_thresh


def _inliers_isoforest(values: pd.Series, contamination: float) -> pd.Series:
    v = values.astype(float).replace([np.inf, -np.inf], np.nan)
    valid = v.dropna()
    if valid.empty:
        return pd.Series(True, index=v.index)
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(valid.to_frame())  # 1 for inliers, -1 for outliers
    mask_valid = preds == 1
    mask = pd.Series(True, index=v.index)
    mask.loc[valid.index] = mask_valid
    return mask


def _ensemble_inliers(values: pd.Series, cfg: OutlierConfig) -> pd.Series:
    m_iqr = _inliers_iqr(values, cfg.iqr_factor)
    m_z = _inliers_zscore(values, cfg.z_thresh)
    m_iso = _inliers_isoforest(values, cfg.iso_forest_contamination)
    # Outlier if at least 2 methods say outlier -> inlier if at least 2 say inlier
    votes_inlier = m_iqr.astype(int) + m_z.astype(int) + m_iso.astype(int)
    return votes_inlier >= 2


def _detect_inliers_series(values: pd.Series, cfg: OutlierConfig) -> pd.Series:
    if cfg.method == "iqr":
        return _inliers_iqr(values, cfg.iqr_factor)
    if cfg.method == "zscore":
        return _inliers_zscore(values, cfg.z_thresh)
    if cfg.method == "iso_forest":
        return _inliers_isoforest(values, cfg.iso_forest_contamination)
    if cfg.method == "ensemble":
        return _ensemble_inliers(values, cfg)
    # default safe
    return pd.Series(True, index=values.index)


def detect_outliers(
    df: pd.DataFrame,
    target_col: str,
    config: OutlierConfig,
) -> pd.Series:
    logger.info(
        f"Outlier detection method={config.method}, group_by={config.group_by_col}, target={target_col}, min_group_size={config.min_group_size}, fallback={config.fallback_strategy}"
    )
    # Return mask of inliers (True = keep)
    if config.group_by_col and config.group_by_col in df.columns:
        inliers_mask = pd.Series(False, index=df.index)
        groups = df.groupby(config.group_by_col).groups
        for cat, idx in groups.items():
            idx = pd.Index(idx)
            values = df.loc[idx, target_col]
            if len(idx) < config.min_group_size:
                if config.fallback_strategy == "global":
                    # compute on full df
                    inliers = _detect_inliers_series(df[target_col], config).reindex(idx)
                else:
                    # skip -> keep all in small group
                    inliers = pd.Series(True, index=idx)
            else:
                inliers = _detect_inliers_series(values, config)
            inliers_mask.loc[idx] = inliers
        return inliers_mask
    else:
        return _detect_inliers_series(df[target_col], config)