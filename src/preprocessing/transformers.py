from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScalingConfig:
    scaler_type: str = "standard"  # 'standard' | 'robust' | 'none'
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
    # mode: 'date' uses fixed year/month threshold; 'fraction' uses time-ordered percentage
    mode: str = "date"
    test_start_year: int = 2023
    test_start_month: int = 1
    train_fraction: float = 0.8
    valid_fraction: float = 0.0  # 0 means no validation split


@dataclass
class CorrelationFilterConfig:
    numeric_threshold: float = 0.98


@dataclass
class FittedTransforms:
    scaler: Optional[object]
    pca: Optional[PCA]


@dataclass
class WinsorConfig:
    enabled: bool = False
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99


@dataclass
class FittedWinsorizer:
    lower_bounds: pd.Series
    upper_bounds: pd.Series


def temporal_key(df: pd.DataFrame, year_col: str, month_col: str) -> pd.Series:
    year = df[year_col].fillna(0).astype(int)
    month = df[month_col].fillna(1).astype(int)
    return year * 100 + month


def temporal_split(
    df: pd.DataFrame,
    split_cfg: TemporalSplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if split_cfg.mode == "fraction":
        key = temporal_key(df, split_cfg.year_col, split_cfg.month_col)
        order = key.sort_values().index
        n = len(order)
        cut = int(np.floor(split_cfg.train_fraction * n))
        train_idx = order[:cut]
        test_idx = order[cut:]
        return df.loc[train_idx].copy(), df.loc[test_idx].copy()

    key = temporal_key(df, split_cfg.year_col, split_cfg.month_col)
    threshold = split_cfg.test_start_year * 100 + split_cfg.test_start_month
    train_df = df[key < threshold].copy()
    test_df = df[key >= threshold].copy()
    return train_df, test_df


def temporal_split_3way(
    df: pd.DataFrame,
    split_cfg: TemporalSplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_cfg.valid_fraction <= 0:
        train_df, test_df = temporal_split(df, split_cfg)
        return train_df, pd.DataFrame(columns=df.columns).astype(df.dtypes.to_dict()), test_df

    key = temporal_key(df, split_cfg.year_col, split_cfg.month_col)

    if split_cfg.mode == "fraction":
        order = key.sort_values().index
        n = len(order)
        n_train = int(np.floor(split_cfg.train_fraction * n))
        n_val = int(np.floor(split_cfg.valid_fraction * n))
        n_train = max(1, min(n - 2, n_train))
        n_val = max(1, min(n - n_train - 1, n_val))
        train_idx = order[:n_train]
        val_idx = order[n_train:n_train + n_val]
        test_idx = order[n_train + n_val:]
        return df.loc[train_idx].copy(), df.loc[val_idx].copy(), df.loc[test_idx].copy()

    # mode == 'date' -> create validation immediately before test period (10% of train by default if valid_fraction==0 -> handled above)
    threshold = split_cfg.test_start_year * 100 + split_cfg.test_start_month
    pre_test = df[key < threshold].copy()
    if pre_test.empty:
        return pre_test, pre_test, df[key >= threshold].copy()
    # take tail as validation according to valid_fraction
    pre_key = temporal_key(pre_test, split_cfg.year_col, split_cfg.month_col)
    order = pre_key.sort_values().index
    n = len(order)
    n_val = int(np.floor(split_cfg.valid_fraction * n))
    n_val = max(1, min(n - 1, n_val))
    val_idx = order[-n_val:]
    train_idx = order[:-n_val]
    test_df = df[key >= threshold].copy()
    return pre_test.loc[train_idx].copy(), pre_test.loc[val_idx].copy(), test_df


def fit_winsorizer(X_train: pd.DataFrame, cfg: WinsorConfig) -> FittedWinsorizer:
    if not cfg.enabled:
        return FittedWinsorizer(lower_bounds=pd.Series(dtype=float), upper_bounds=pd.Series(dtype=float))
    lb = X_train.quantile(cfg.lower_quantile)
    ub = X_train.quantile(cfg.upper_quantile)
    return FittedWinsorizer(lower_bounds=lb, upper_bounds=ub)


def apply_winsorizer(X: pd.DataFrame, fitted: FittedWinsorizer) -> pd.DataFrame:
    if fitted.lower_bounds.empty and fitted.upper_bounds.empty:
        return X
    Y = X.copy()
    for col in Y.columns:
        if col in fitted.lower_bounds.index:
            Y[col] = Y[col].clip(lower=fitted.lower_bounds[col], upper=fitted.upper_bounds[col])
    return Y


def scale_and_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaling_cfg: ScalingConfig,
    pca_cfg: PCAConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, FittedTransforms]:
    # Choose scaler
    scaler: Optional[object]
    if scaling_cfg.scaler_type == "none":
        scaler = None
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
    elif scaling_cfg.scaler_type == "robust":
        scaler = RobustScaler(with_centering=scaling_cfg.with_mean, with_scaling=scaling_cfg.with_std)
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    else:
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


def drop_non_descriptive(X: pd.DataFrame, na_threshold: float = 0.98) -> Tuple[pd.DataFrame, List[str]]:
    to_drop: List[str] = []
    # Drop constant columns
    nunique = X.nunique(dropna=False)
    constants = nunique[nunique <= 1].index.tolist()
    if constants:
        to_drop.extend(constants)
    # Drop columns with excessive missingness (> na_threshold)
    na_frac = X.isna().mean()
    too_missing = na_frac[na_frac > na_threshold].index.tolist()
    if too_missing:
        to_drop.extend(too_missing)
    return X.drop(columns=to_drop, errors="ignore"), to_drop