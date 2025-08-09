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
    # Prepare base copies before per-profile transformations
    base_train = X_train.copy()
    base_test = X_test.copy()
    base_val = X_val.copy() if X_val is not None else None

    profiles_cfg = config.get("profiles", {
        "scaled": {"enabled": True, "output_prefix": "scaled"},
        "tree": {"enabled": False, "output_prefix": "tree"},
        "catboost": {"enabled": False, "output_prefix": "catboost"},
    })

    first_profile_saved = None

    def save_profile(X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr: pd.Series, y_te: pd.Series, X_va: pd.DataFrame | None, y_va: pd.Series | None, prefix: str):
        X_tr.to_parquet(pre_dir / f"X_train_{prefix}.parquet", index=False)
        X_te.to_parquet(pre_dir / f"X_test_{prefix}.parquet", index=False)
        pd.DataFrame({target_col: y_tr}).to_parquet(pre_dir / f"y_train_{prefix}.parquet", index=False)
        pd.DataFrame({target_col: y_te}).to_parquet(pre_dir / f"y_test_{prefix}.parquet", index=False)
        if X_va is not None and y_va is not None:
            X_va.to_parquet(pre_dir / f"X_val_{prefix}.parquet", index=False)
            pd.DataFrame({target_col: y_va}).to_parquet(pre_dir / f"y_val_{prefix}.parquet", index=False)

    # Helper: numeric coercion based on train
    def coerce_numeric_like(train_df: pd.DataFrame, other_dfs: List[pd.DataFrame | None]) -> Tuple[pd.DataFrame, List[pd.DataFrame | None]]:
        train = train_df.copy()
        cols_to_coerce: List[str] = []
        for col in train.select_dtypes(include=["object"]).columns:
            coerced = pd.to_numeric(train[col].astype(str).str.replace(",", "."), errors="coerce")
            if coerced.notna().mean() > 0.8:
                cols_to_coerce.append(col)
                train[col] = coerced
        outs: List[pd.DataFrame | None] = []
        for df_ in other_dfs:
            if df_ is None:
                outs.append(None)
            else:
                tmp = df_.copy()
                for col in cols_to_coerce:
                    if col in tmp.columns:
                        tmp[col] = pd.to_numeric(tmp[col].astype(str).str.replace(",", "."), errors="coerce")
                outs.append(tmp)
        return train, outs

    # Profile: scaled
    if profiles_cfg.get("scaled", {}).get("enabled", False):
        enc_max = int(profiles_cfg.get("scaled", {}).get("encoding", {}).get("max_ohe_cardinality", config.get("encoding", {}).get("max_ohe_cardinality", 12)))
        X_tr = base_train.copy(); X_te = base_test.copy(); X_va = base_val.copy() if base_val is not None else None
        plan = plan_encodings(X_tr, max_ohe_cardinality=enc_max)
        X_tr, encoders, _ = fit_apply_encoders(X_tr, plan)
        X_te = transform_with_encoders(X_te, encoders)
        if X_va is not None:
            X_va = transform_with_encoders(X_va, encoders)
        X_tr, [X_te, X_va] = coerce_numeric_like(X_tr, [X_te, X_va])
        X_tr, removed_nd = drop_non_descriptive(X_tr)
        X_te = X_te.drop(columns=[c for c in removed_nd if c in X_te.columns], errors="ignore")
        if X_va is not None:
            X_va = X_va.drop(columns=[c for c in removed_nd if c in X_va.columns], errors="ignore")
        # Align numeric and apply winsor/scaling/pca
        X_tr_num = X_tr.select_dtypes(include=[np.number]).fillna(0)
        X_te_num = X_te.select_dtypes(include=[np.number]).fillna(0).reindex(columns=X_tr_num.columns, fill_value=0)
        X_va_num = None
        if X_va is not None:
            X_va_num = X_va.select_dtypes(include=[np.number]).fillna(0).reindex(columns=X_tr_num.columns, fill_value=0)
        # Winsor
        wins_cfg_dict = profiles_cfg.get("scaled", {}).get("winsorization", config.get("winsorization", {}))
        from preprocessing.transformers import WinsorConfig
        wins_cfg = WinsorConfig(
            enabled=bool(wins_cfg_dict.get("enabled", False)),
            lower_quantile=float(wins_cfg_dict.get("lower_quantile", 0.01)),
            upper_quantile=float(wins_cfg_dict.get("upper_quantile", 0.99)),
        )
        winsorizer = fit_winsorizer(X_tr_num, wins_cfg)
        X_tr_num = apply_winsorizer(X_tr_num, winsorizer)
        X_te_num = apply_winsorizer(X_te_num, winsorizer)
        if X_va_num is not None:
            X_va_num = apply_winsorizer(X_va_num, winsorizer)
        # Scaling/PCA
        sc_dict = profiles_cfg.get("scaled", {}).get("scaling", config.get("scaling", {}))
        scaling_cfg = ScalingConfig(
            scaler_type=sc_dict.get("scaler_type", "standard"),
            with_mean=bool(sc_dict.get("with_mean", True)),
            with_std=bool(sc_dict.get("with_std", True)),
        )
        pca_dict = profiles_cfg.get("scaled", {}).get("pca", config.get("pca", {}))
        pca_cfg = PCAConfig(
            enabled=bool(pca_dict.get("enabled", False)),
            n_components=pca_dict.get("n_components", 0.95),
            random_state=int(pca_dict.get("random_state", 42)),
        )
        X_tr_f, X_te_f, fitted = scale_and_pca(X_tr_num, X_te_num, scaling_cfg, pca_cfg)
        if X_va_num is not None:
            X_va_scaled = pd.DataFrame(fitted.scaler.transform(X_va_num), columns=X_va_num.columns, index=X_va_num.index) if fitted.scaler is not None else X_va_num
            if fitted.pca is not None:
                vals = fitted.pca.transform(X_va_scaled)
                X_va_f = pd.DataFrame(vals, columns=[f"PC{i+1}" for i in range(vals.shape[1])], index=X_va_scaled.index)
            else:
                X_va_f = X_va_scaled
        else:
            X_va_f = None
        # Correlation prune
        corr_thr = float(profiles_cfg.get("scaled", {}).get("correlation", {}).get("numeric_threshold", config.get("correlation", {}).get("numeric_threshold", 0.98)))
        X_tr_f, dropped_corr = remove_highly_correlated(X_tr_f, threshold=corr_thr)
        X_te_f = X_te_f.drop(columns=[c for c in dropped_corr if c in X_te_f.columns], errors="ignore")
        if X_va_f is not None:
            X_va_f = X_va_f.drop(columns=[c for c in dropped_corr if c in X_va_f.columns], errors="ignore")
        prefix = profiles_cfg.get("scaled", {}).get("output_prefix", "scaled")
        save_profile(X_tr_f, X_te_f, y_train, y_test, X_va_f, y_val, prefix)
        if first_profile_saved is None:
            first_profile_saved = prefix

    # Profile: tree (no scaling/PCA)
    if profiles_cfg.get("tree", {}).get("enabled", False):
        enc_max = int(profiles_cfg.get("tree", {}).get("encoding", {}).get("max_ohe_cardinality", config.get("encoding", {}).get("max_ohe_cardinality", 12)))
        X_tr = base_train.copy(); X_te = base_test.copy(); X_va = base_val.copy() if base_val is not None else None
        plan = plan_encodings(X_tr, max_ohe_cardinality=enc_max)
        X_tr, encoders, _ = fit_apply_encoders(X_tr, plan)
        X_te = transform_with_encoders(X_te, encoders)
        if X_va is not None:
            X_va = transform_with_encoders(X_va, encoders)
        X_tr, [X_te, X_va] = coerce_numeric_like(X_tr, [X_te, X_va])
        X_tr, removed_nd = drop_non_descriptive(X_tr)
        X_te = X_te.drop(columns=[c for c in removed_nd if c in X_te.columns], errors="ignore")
        if X_va is not None:
            X_va = X_va.drop(columns=[c for c in removed_nd if c in X_va.columns], errors="ignore")
        # Align columns (no scaling/pca)
        cols = X_tr.columns
        X_te = X_te.reindex(columns=cols, fill_value=0)
        if X_va is not None:
            X_va = X_va.reindex(columns=cols, fill_value=0)
        # Optional numeric-only correlation prune
        corr_thr = float(profiles_cfg.get("tree", {}).get("correlation", {}).get("numeric_threshold", config.get("correlation", {}).get("numeric_threshold", 0.98)))
        X_tr_num = X_tr.select_dtypes(include=[np.number])
        X_tr_num_pruned, dropped_corr = remove_highly_correlated(X_tr_num, threshold=corr_thr)
        keep_cols = list(dropped_corr)
        # Build mask of kept numeric columns
        kept_num_cols = X_tr_num_pruned.columns
        X_tr_final = pd.concat([X_tr[kept_num_cols], X_tr.drop(columns=kept_num_cols, errors="ignore").select_dtypes(exclude=[np.number])], axis=1)
        X_te_final = pd.concat([X_te[kept_num_cols], X_te.drop(columns=kept_num_cols, errors="ignore").select_dtypes(exclude=[np.number])], axis=1)
        if X_va is not None:
            X_va_final = pd.concat([X_va[kept_num_cols], X_va.drop(columns=kept_num_cols, errors="ignore").select_dtypes(exclude=[np.number])], axis=1)
        else:
            X_va_final = None
        prefix = profiles_cfg.get("tree", {}).get("output_prefix", "tree")
        save_profile(X_tr_final, X_te_final, y_train, y_test, X_va_final, y_val, prefix)
        if first_profile_saved is None:
            first_profile_saved = prefix

    # Profile: catboost (preserve categoricals)
    if profiles_cfg.get("catboost", {}).get("enabled", False):
        X_tr = base_train.copy(); X_te = base_test.copy(); X_va = base_val.copy() if base_val is not None else None
        # Imputation already done; keep categoricals
        # Coerce numeric-like strings to numeric
        X_tr, [X_te, X_va] = coerce_numeric_like(X_tr, [X_te, X_va])
        # Drop non-descriptive
        X_tr, removed_nd = drop_non_descriptive(X_tr)
        X_te = X_te.drop(columns=[c for c in removed_nd if c in X_te.columns], errors="ignore")
        if X_va is not None:
            X_va = X_va.drop(columns=[c for c in removed_nd if c in X_va.columns], errors="ignore")
        # Numeric-only correlation prune
        corr_thr = float(profiles_cfg.get("catboost", {}).get("correlation", {}).get("numeric_threshold", config.get("correlation", {}).get("numeric_threshold", 0.98)))
        X_tr_num = X_tr.select_dtypes(include=[np.number])
        X_tr_num_pruned, dropped_corr = remove_highly_correlated(X_tr_num, threshold=corr_thr)
        kept_num_cols = X_tr_num_pruned.columns
        X_tr_final = pd.concat([X_tr[kept_num_cols], X_tr.drop(columns=kept_num_cols, errors="ignore").select_dtypes(exclude=[np.number])], axis=1)
        X_te_final = pd.concat([X_te[kept_num_cols], X_te.drop(columns=kept_num_cols, errors="ignore").select_dtypes(exclude=[np.number])], axis=1)
        if X_va is not None:
            X_va_final = pd.concat([X_va[kept_num_cols], X_va.drop(columns=kept_num_cols, errors="ignore").select_dtypes(exclude=[np.number])], axis=1)
        else:
            X_va_final = None
        prefix = profiles_cfg.get("catboost", {}).get("output_prefix", "catboost")
        save_profile(X_tr_final, X_te_final, y_train, y_test, X_va_final, y_val, prefix)
        # Save list of categorical columns for catboost
        cat_cols = X_tr_final.select_dtypes(include=["object", "category"]).columns.tolist()
        (pre_dir / f"categorical_columns_{prefix}.txt").write_text("\n".join(cat_cols), encoding="utf-8")
        if first_profile_saved is None:
            first_profile_saved = prefix

    # Backward-compatible symlinks (copy) to default filenames using first enabled profile
    if first_profile_saved is not None:
        # Save combined
        X_train_bc = pd.read_parquet(pre_dir / f"X_train_{first_profile_saved}.parquet")
        X_test_bc = pd.read_parquet(pre_dir / f"X_test_{first_profile_saved}.parquet")
        y_train_bc = pd.read_parquet(pre_dir / f"y_train_{first_profile_saved}.parquet")[target_col]
        y_test_bc = pd.read_parquet(pre_dir / f"y_test_{first_profile_saved}.parquet")[target_col]
        frames = [X_train_bc.assign(**{target_col: y_train_bc})]
        if (pre_dir / f"X_val_{first_profile_saved}.parquet").exists() and (pre_dir / f"y_val_{first_profile_saved}.parquet").exists():
            X_val_bc = pd.read_parquet(pre_dir / f"X_val_{first_profile_saved}.parquet")
            y_val_bc = pd.read_parquet(pre_dir / f"y_val_{first_profile_saved}.parquet")[target_col]
            frames.append(X_val_bc.assign(**{target_col: y_val_bc}))
        frames.append(X_test_bc.assign(**{target_col: y_test_bc}))
        combined = pd.concat(frames, axis=0)
        out_path = pre_dir / paths.get("preprocessed_filename", "preprocessed.parquet")
        combined.to_parquet(out_path, index=False)
        # Default names without suffix
        (pre_dir / "X_train.parquet").write_bytes((pre_dir / f"X_train_{first_profile_saved}.parquet").read_bytes())
        (pre_dir / "X_test.parquet").write_bytes((pre_dir / f"X_test_{first_profile_saved}.parquet").read_bytes())
        (pre_dir / "y_train.parquet").write_bytes((pre_dir / f"y_train_{first_profile_saved}.parquet").read_bytes())
        (pre_dir / "y_test.parquet").write_bytes((pre_dir / f"y_test_{first_profile_saved}.parquet").read_bytes())
        if (pre_dir / f"X_val_{first_profile_saved}.parquet").exists():
            (pre_dir / "X_val.parquet").write_bytes((pre_dir / f"X_val_{first_profile_saved}.parquet").read_bytes())
        if (pre_dir / f"y_val_{first_profile_saved}.parquet").exists():
            (pre_dir / "y_val.parquet").write_bytes((pre_dir / f"y_val_{first_profile_saved}.parquet").read_bytes())

    # Report
    save_report(
        reports_dir / "preprocessing.md",
        sections={
            "Raw profile": dataframe_profile(df),
            "After feature extraction": dataframe_profile(base_train),
            "Train features": dataframe_profile(base_train),
            "Test features": dataframe_profile(base_test),
        },
    )

    logger.info(f"Preprocessing completato: {out_path}")
    return out_path