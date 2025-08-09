from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger
from preprocessing.feature_extractors import extract_geometry_features, maybe_extract_json_features
from preprocessing.outliers import OutlierConfig, detect_outliers
from preprocessing.encoders import plan_encodings, fit_apply_encoders, transform_with_encoders
from preprocessing.imputation import ImputationConfig, impute_missing, fit_imputers, transform_with_imputers
from preprocessing.transformers import (
    TemporalSplitConfig,
    ScalingConfig,
    PCAConfig,
    temporal_split,
    temporal_split_3way,
    fit_winsorizer,
    apply_winsorizer,
    scale_and_pca,
    remove_highly_correlated,
    drop_non_descriptive,
)
from preprocessing.report import dataframe_profile, save_report

logger = get_logger(__name__)


def choose_target(df: pd.DataFrame, config: Dict[str, Any]) -> str:
    # Prefer redistributed price if present, else fall back to AI_Valore or similar if available
    preferred = config.get("target", {}).get("column_candidates", [
        "AI_Prezzo_Ridistribuito",
        "AI_Prezzo",
    ])
    for c in preferred:
        if c in df.columns:
            return c
    # Fallback: look for price-like columns
    price_like = [c for c in df.columns if "Prezzo" in c or "Valore" in c]
    if not price_like:
        raise ValueError("Nessuna colonna target trovata (Prezzo/Valore)")
    return price_like[0]


def apply_log_target_if(config: Dict[str, Any], y: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    use_log = bool(config.get("target", {}).get("log_transform", False))
    if not use_log:
        return y, {"log": False}
    y_pos = y.clip(lower=1e-6)
    return np.log1p(y_pos), {"log": True}


def run_preprocessing(config: Dict[str, Any]) -> Path:
    paths = config.get("paths", {})
    raw_dir = Path(paths.get("raw_data", "data/raw"))
    pre_dir = Path(paths.get("preprocessed_data", "data/preprocessed"))
    reports_dir = pre_dir / "reports"
    pre_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_files = list(raw_dir.glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"Nessun file parquet trovato in {raw_dir}")

    df = pd.read_parquet(raw_files[0])

    # Initial cleanup
    df = df.dropna(axis=1, how="all")

    # Feature extraction from JSON and geometry-like WKT columns
    fe_cfg = config.get("feature_extraction", {})
    if bool(fe_cfg.get("geometry", True)):
        df, drop_geom = extract_geometry_features(df)
    else:
        drop_geom = []
    if bool(fe_cfg.get("json", True)):
        df, drop_json = maybe_extract_json_features(df)
    else:
        drop_json = []
    cols_to_drop_now = list(set(drop_geom + drop_json))
    if cols_to_drop_now:
        df = df.drop(columns=cols_to_drop_now, errors="ignore")

    # Create temporal key to enable split later (do not leak across time)
    if "A_AnnoStipula" in df.columns and "A_MeseStipula" in df.columns:
        df["TemporalKey"] = df["A_AnnoStipula"].astype(int) * 100 + df["A_MeseStipula"].astype(int)

    # Decide target
    target_col = choose_target(df, config)

    # Create combined for split key if needed
    Xy_full = df.copy()

    # Temporal split FIRST to avoid leakage
    split_cfg_dict = config.get("temporal_split", {})
    split_cfg = TemporalSplitConfig(
        year_col=split_cfg_dict.get("year_col", "A_AnnoStipula"),
        month_col=split_cfg_dict.get("month_col", "A_MeseStipula"),
        test_start_year=int(split_cfg_dict.get("test_start_year", 2023)),
        test_start_month=int(split_cfg_dict.get("test_start_month", 1)),
    )
    # Use 3-way split (val optional if valid_fraction>0)
    train_df, val_df, test_df = temporal_split_3way(Xy_full, split_cfg)

    # Outlier detection ONLY on train target (optionally per category)
    out_cfg_dict = config.get("outliers", {})
    out_cfg = OutlierConfig(
        method=out_cfg_dict.get("method", "iqr"),
        z_thresh=float(out_cfg_dict.get("z_thresh", 4.0)),
        iqr_factor=float(out_cfg_dict.get("iqr_factor", 1.5)),
        iso_forest_contamination=float(out_cfg_dict.get("iso_forest_contamination", 0.02)),
        group_by_col=out_cfg_dict.get("group_by_col", "AI_IdCategoriaCatastale"),
        min_group_size=int(out_cfg_dict.get("min_group_size", 30)),
        fallback_strategy=str(out_cfg_dict.get("fallback_strategy", "skip")),
    )
    inliers_mask = detect_outliers(train_df, target_col, out_cfg)
    train_df = train_df.loc[inliers_mask].copy()

    # Separate X and y
    y_train = train_df[target_col].astype(float)
    y_test = test_df[target_col].astype(float)
    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])
    X_val = None
    y_val = None
    if not val_df.empty:
        y_val = val_df[target_col].astype(float)
        X_val = val_df.drop(columns=[target_col])

    # Optional log-transform of target
    y_train, log_meta = apply_log_target_if(config, y_train)
    if log_meta.get("log"):
        # Apply same transform to test for consistency
        y_test = np.log1p(y_test.clip(lower=1e-6))

    # Fit imputers on train, transform train/test
    imp_cfg_dict = config.get("imputation", {})
    imp_cfg = ImputationConfig(
        numeric_strategy=imp_cfg_dict.get("numeric_strategy", "median"),
        categorical_strategy=imp_cfg_dict.get("categorical_strategy", "most_frequent"),
        group_by_col=imp_cfg_dict.get("group_by_col", "AI_IdCategoriaCatastale"),
    )
    fitted_imputers = fit_imputers(X_train, imp_cfg)
    X_train = transform_with_imputers(X_train, fitted_imputers)
    X_test = transform_with_imputers(X_test, fitted_imputers)
    if X_val is not None:
        X_val = transform_with_imputers(X_val, fitted_imputers)

    # Encoding plan on train columns, fit on train and transform test
    plan = plan_encodings(X_train, max_ohe_cardinality=int(config.get("encoding", {}).get("max_ohe_cardinality", 12)))
    X_train, encoders, _ = fit_apply_encoders(X_train, plan)
    X_test = transform_with_encoders(X_test, encoders)
    if X_val is not None:
        X_val = transform_with_encoders(X_val, encoders)

    # Cast remaining object columns that look numeric (decide on train; apply to both)
    cols_to_coerce: List[str] = []
    for col in X_train.select_dtypes(include=["object"]).columns:
        coerced = pd.to_numeric(X_train[col].astype(str).str.replace(",", "."), errors="coerce")
        if coerced.notna().mean() > 0.8:
            cols_to_coerce.append(col)
            X_train[col] = coerced
    for col in cols_to_coerce:
        if col in X_test.columns:
            X_test[col] = pd.to_numeric(X_test[col].astype(str).str.replace(",", "."), errors="coerce")
        if X_val is not None and col in X_val.columns:
            X_val[col] = pd.to_numeric(X_val[col].astype(str).str.replace(",", "."), errors="coerce")

    # Remove non-descriptive columns (train-based)
    X_train, removed_non_descr = drop_non_descriptive(X_train)
    X_test = X_test.drop(columns=[c for c in removed_non_descr if c in X_test.columns], errors="ignore")
    if X_val is not None:
        X_val = X_val.drop(columns=[c for c in removed_non_descr if c in X_val.columns], errors="ignore")

    # Prepare numeric matrices and align columns
    X_train_num = X_train.select_dtypes(include=[np.number]).fillna(0)
    X_test_num = X_test.select_dtypes(include=[np.number]).fillna(0)
    X_test_num = X_test_num.reindex(columns=X_train_num.columns, fill_value=0.0)
    X_val_num = None
    if X_val is not None:
        X_val_num = X_val.select_dtypes(include=[np.number]).fillna(0)
        X_val_num = X_val_num.reindex(columns=X_train_num.columns, fill_value=0.0)

    # Optional winsorization (fit on train, apply to all)
    wins_cfg_dict = config.get("winsorization", {})
    from preprocessing.transformers import WinsorConfig
    wins_cfg = WinsorConfig(
        enabled=bool(wins_cfg_dict.get("enabled", False)),
        lower_quantile=float(wins_cfg_dict.get("lower_quantile", 0.01)),
        upper_quantile=float(wins_cfg_dict.get("upper_quantile", 0.99)),
    )
    winsorizer = fit_winsorizer(X_train_num, wins_cfg)
    X_train_num = apply_winsorizer(X_train_num, winsorizer)
    X_test_num = apply_winsorizer(X_test_num, winsorizer)
    if X_val_num is not None:
        X_val_num = apply_winsorizer(X_val_num, winsorizer)

    # Scaling and optional PCA (fit only on train)
    scaling_cfg_dict = config.get("scaling", {})
    scaling_cfg = ScalingConfig(
        scaler_type=scaling_cfg_dict.get("scaler_type", "standard"),
        with_mean=bool(scaling_cfg_dict.get("with_mean", True)),
        with_std=bool(scaling_cfg_dict.get("with_std", True)),
    )
    pca_cfg_dict = config.get("pca", {})
    pca_cfg = PCAConfig(
        enabled=bool(pca_cfg_dict.get("enabled", False)),
        n_components=pca_cfg_dict.get("n_components", 0.95),
        random_state=int(pca_cfg_dict.get("random_state", 42)),
    )
    X_train, X_test, fitted = scale_and_pca(X_train_num, X_test_num, scaling_cfg, pca_cfg)
    if X_val_num is not None:
        # Transform val using fitted scaler/pca
        X_val_scaled = pd.DataFrame(fitted.scaler.transform(X_val_num), columns=X_val_num.columns, index=X_val_num.index)
        if fitted.pca is not None:
            X_val_pca = fitted.pca.transform(X_val_scaled)
            X_val = pd.DataFrame(X_val_pca, columns=[f"PC{i+1}" for i in range(X_val_pca.shape[1])], index=X_val_num.index)
        else:
            X_val = X_val_scaled

    # Remove highly correlated columns (train-based)
    X_train, dropped_corr = remove_highly_correlated(X_train, threshold=float(config.get("correlation", {}).get("numeric_threshold", 0.98)))
    X_test = X_test.drop(columns=[c for c in dropped_corr if c in X_test.columns], errors="ignore")
    if X_val is not None:
        X_val = X_val.drop(columns=[c for c in dropped_corr if c in X_val.columns], errors="ignore")

    # Save outputs
    pre_filename = paths.get("preprocessed_filename", "preprocessed.parquet")
    out_path = pre_dir / pre_filename
    # For backward compatibility, keep combined preprocessed with features + target
    frames = [X_train.assign(**{target_col: y_train})]
    if X_val is not None and y_val is not None:
        frames.append(X_val.assign(**{target_col: y_val}))
    frames.append(X_test.assign(**{target_col: y_test}))
    combined = pd.concat(frames, axis=0)
    combined.to_parquet(out_path, index=False)

    # Also save split datasets for training convenience
    X_train_path = pre_dir / "X_train.parquet"
    X_test_path = pre_dir / "X_test.parquet"
    y_train_path = pre_dir / "y_train.parquet"
    y_test_path = pre_dir / "y_test.parquet"
    X_train.to_parquet(X_train_path, index=False)
    X_test.to_parquet(X_test_path, index=False)
    pd.DataFrame({target_col: y_train}).to_parquet(y_train_path, index=False)
    pd.DataFrame({target_col: y_test}).to_parquet(y_test_path, index=False)
    if X_val is not None and y_val is not None:
        X_val_path = pre_dir / "X_val.parquet"
        y_val_path = pre_dir / "y_val.parquet"
        X_val.to_parquet(X_val_path, index=False)
        pd.DataFrame({target_col: y_val}).to_parquet(y_val_path, index=False)

    # Report
    save_report(
        reports_dir / "preprocessing.md",
        sections={
            "Raw profile": dataframe_profile(df),
            "After feature extraction": dataframe_profile(X_train),
            "Train features": dataframe_profile(X_train),
            "Test features": dataframe_profile(X_test),
        },
    )

    logger.info(f"Preprocessing completato: {out_path}")
    return out_path