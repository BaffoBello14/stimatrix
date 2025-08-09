from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScalingConfig:
    with_mean: bool = True
    with_std: bool = True


@dataclass
class PCAConfig:
    enabled: bool = False
    n_components: float | int = 0.95  # keep 95% variance by default
    random_state: int = 42


@dataclass
class TemporalSplitConfig:
    year_col: str = "A_AnnoStipula"
    month_col: str = "A_MeseStipula"
    test_start_year: int = 2023
    test_start_month: int = 1


@dataclass
class CorrelationFilterConfig:
    numeric_threshold: float = 0.98


@dataclass
class FittedTransforms:
    scaler: Optional[StandardScaler]
    pca: Optional[PCA]


def temporal_key(df: pd.DataFrame, year_col: str, month_col: str) -> pd.Series:
    year = df[year_col].fillna(0).astype(int)
    month = df[month_col].fillna(1).astype(int)
    return year * 100 + month


def temporal_split(
    df: pd.DataFrame,
    split_cfg: TemporalSplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    key = temporal_key(df, split_cfg.year_col, split_cfg.month_col)
    threshold = split_cfg.test_start_year * 100 + split_cfg.test_start_month
    train_df = df[key < threshold].copy()
    test_df = df[key >= threshold].copy()
    return train_df, test_df


def scale_and_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaling_cfg: ScalingConfig,
    pca_cfg: PCAConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, FittedTransforms]:
    scaler = StandardScaler(with_mean=scaling_cfg.with_mean, with_std=scaling_cfg.with_std)
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    if pca_cfg.enabled:
        pca = PCA(n_components=pca_cfg.n_components, random_state=pca_cfg.random_state)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        comp_cols = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]
        X_train_df = pd.DataFrame(X_train_pca, columns=comp_cols, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_pca, columns=comp_cols, index=X_test.index)
        return X_train_df, X_test_df, FittedTransforms(scaler=scaler, pca=pca)

    return X_train_scaled, X_test_scaled, FittedTransforms(scaler=scaler, pca=None)


def remove_highly_correlated(X: pd.DataFrame, threshold: float = 0.98) -> Tuple[pd.DataFrame, List[str]]:
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return X.drop(columns=to_drop, errors="ignore"), to_drop


def drop_non_descriptive(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    to_drop: List[str] = []
    # Drop constant columns
    nunique = X.nunique(dropna=False)
    constants = nunique[nunique <= 1].index.tolist()
    if constants:
        to_drop.extend(constants)
    # Drop columns with excessive missingness (> 98%)
    na_frac = X.isna().mean()
    too_missing = na_frac[na_frac > 0.98].index.tolist()
    if too_missing:
        to_drop.extend(too_missing)
    return X.drop(columns=to_drop, errors="ignore"), to_drop