from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import datetime

import numpy as np
import pandas as pd
import re
import fnmatch

from utils.logger import get_logger
from utils.io import save_json
from preprocessing.feature_extractors import extract_geometry_features, maybe_extract_json_features, create_missing_pattern_flags
from preprocessing.outliers import OutlierConfig, detect_outliers
from preprocessing.encoders import plan_encodings, fit_apply_encoders, transform_with_encoders
from preprocessing.imputation import ImputationConfig, fit_imputers, transform_with_imputers
from preprocessing.contextual_features import fit_transform_contextual_features
from preprocessing.target_transforms import (
    apply_target_transform,
    inverse_target_transform,
    get_transform_name,
    boxcox_transform,
    yeojohnson_transform,
)
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
from joblib import dump

from preprocessing.constants import (
    MISSING_CATEGORY_SENTINEL,
    DATETIME_SAMPLE_SIZE,
    DATETIME_ISO_FORMAT,
    DEFAULT_TEMPORAL_TRAIN_FRACTION,
    DEFAULT_TEMPORAL_VALID_FRACTION,
)

logger = get_logger(__name__)

def convert_datetime_columns_to_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all datetime-like columns to ISO formatted strings.

    Handles both native pandas datetime dtypes and object columns containing
    `datetime.date` / `datetime.datetime` instances.
    """
    if df is None:
        return None  # type: ignore[return-value]

    converted_cols: List[str] = []
    converted_object_cols: List[str] = []

    result = df.copy()

    datetime_cols = [col for col in result.columns if pd.api.types.is_datetime64_any_dtype(result[col])]
    for col in datetime_cols:
        series = result[col]
        result[col] = series.dt.strftime(DATETIME_ISO_FORMAT)
        result.loc[series.isna(), col] = None
        converted_cols.append(col)

    for col in result.select_dtypes(include=["object"]).columns:
        sample = result[col].dropna().head(DATETIME_SAMPLE_SIZE)
        if sample.empty:
            continue
        if any(isinstance(value, (datetime.date, datetime.datetime)) for value in sample):
            result[col] = result[col].apply(
                lambda x: x.isoformat() if isinstance(x, (datetime.date, datetime.datetime))
                else (None if pd.isna(x) else x)
            )
            converted_object_cols.append(col)

    if converted_cols or converted_object_cols:
        summary_cols = converted_cols + converted_object_cols
        preview = ", ".join(summary_cols[:5])
        suffix = " ..." if len(summary_cols) > 5 else ""
        logger.debug(
            "Converted %s datetime-like columns to strings: %s%s",
            len(summary_cols),
            preview,
            suffix,
        )

    return result


def apply_data_filters(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Applica filtri al dataset INTERO (prima dello split).
    
    Usato per sperimentazione: filtrare sottoinsiemi specifici e
    valutare se modelli specializzati performano meglio.
    
    Args:
        df: DataFrame raw
        config: Config con sezione 'data_filters'
    
    Returns:
        DataFrame filtrato
    """
    filters = config.get('data_filters', {})
    if not filters or not any(v is not None for k, v in filters.items() if k not in ['description', 'experiment_name']):
        # No filters active
        return df
    
    initial_rows = len(df)
    
    # Temporali
    if filters.get('anno_min') and 'A_AnnoStipula' in df.columns:
        df = df[df['A_AnnoStipula'] >= filters['anno_min']]
    if filters.get('anno_max') and 'A_AnnoStipula' in df.columns:
        df = df[df['A_AnnoStipula'] <= filters['anno_max']]
    if filters.get('mese_min') and 'A_MeseStipula' in df.columns:
        df = df[df['A_MeseStipula'] >= filters['mese_min']]
    if filters.get('mese_max') and 'A_MeseStipula' in df.columns:
        df = df[df['A_MeseStipula'] <= filters['mese_max']]
    
    # Target (prezzo)
    if filters.get('prezzo_min') and 'AI_Prezzo_Ridistribuito' in df.columns:
        df = df[df['AI_Prezzo_Ridistribuito'] >= filters['prezzo_min']]
    if filters.get('prezzo_max') and 'AI_Prezzo_Ridistribuito' in df.columns:
        df = df[df['AI_Prezzo_Ridistribuito'] <= filters['prezzo_max']]
    
    # Caratteristiche immobile
    if filters.get('superficie_min') and 'AI_Superficie' in df.columns:
        df = df[df['AI_Superficie'] >= filters['superficie_min']]
    if filters.get('superficie_max') and 'AI_Superficie' in df.columns:
        df = df[df['AI_Superficie'] <= filters['superficie_max']]
    
    if filters.get('locali_min') and 'AI_NumeroLocali' in df.columns:
        df = df[df['AI_NumeroLocali'] >= filters['locali_min']]
    if filters.get('locali_max') and 'AI_NumeroLocali' in df.columns:
        df = df[df['AI_NumeroLocali'] <= filters['locali_max']]
    
    if filters.get('piano_min') and 'AI_Piano' in df.columns:
        df = df[df['AI_Piano'] >= filters['piano_min']]
    if filters.get('piano_max') and 'AI_Piano' in df.columns:
        df = df[df['AI_Piano'] <= filters['piano_max']]
    
    # Geografiche
    if filters.get('zone_incluse') and 'AI_ZonaOmi' in df.columns:
        df = df[df['AI_ZonaOmi'].isin(filters['zone_incluse'])]
    if filters.get('zone_escluse') and 'AI_ZonaOmi' in df.columns:
        df = df[~df['AI_ZonaOmi'].isin(filters['zone_escluse'])]
    
    if filters.get('tipologie_incluse') and 'AI_IdTipologiaEdilizia' in df.columns:
        df = df[df['AI_IdTipologiaEdilizia'].isin(filters['tipologie_incluse'])]
    if filters.get('tipologie_escluse') and 'AI_IdTipologiaEdilizia' in df.columns:
        df = df[~df['AI_IdTipologiaEdilizia'].isin(filters['tipologie_escluse'])]
    
    # Qualità dati
    if filters.get('max_missing_ratio'):
        missing_ratio = df.isnull().sum(axis=1) / len(df.columns)
        df = df[missing_ratio <= filters['max_missing_ratio']]
    
    if filters.get('remove_outliers_iqr') and 'AI_Prezzo_Ridistribuito' in df.columns:
        Q1 = df['AI_Prezzo_Ridistribuito'].quantile(0.25)
        Q3 = df['AI_Prezzo_Ridistribuito'].quantile(0.75)
        IQR = Q3 - Q1
        iqr_factor = filters.get('iqr_factor', 1.5)
        lower = Q1 - iqr_factor * IQR
        upper = Q3 + iqr_factor * IQR
        df = df[(df['AI_Prezzo_Ridistribuito'] >= lower) & 
                (df['AI_Prezzo_Ridistribuito'] <= upper)]
    
    final_rows = len(df)
    removed = initial_rows - final_rows
    pct_removed = (removed / initial_rows * 100) if initial_rows > 0 else 0
    
    # Log e warnings
    experiment_name = filters.get('experiment_name', 'custom')
    description = filters.get('description', 'Filtered dataset')
    
    logger.info(f"📊 Data Filters Applied: {experiment_name}")
    logger.info(f"   Description: {description}")
    logger.info(f"   Initial rows: {initial_rows:,}")
    logger.info(f"   Final rows: {final_rows:,}")
    logger.info(f"   Removed: {removed:,} ({pct_removed:.1f}%)")
    
    # Warning se troppi dati rimossi
    if pct_removed > 70:
        logger.warning(f"⚠️  Warning: {pct_removed:.1f}% dei dati rimossi! Dataset molto ristretto.")
    if final_rows < 500:
        logger.warning(f"⚠️  Warning: Solo {final_rows} campioni rimanenti! Potrebbe essere insufficiente.")
    
    # Log filtri attivi
    active_filters = [k for k, v in filters.items() 
                      if v is not None and k not in ['description', 'experiment_name', 'iqr_factor']]
    if active_filters:
        logger.info(f"   Active filters: {', '.join(active_filters)}")
    
    return df


def choose_target(df: pd.DataFrame, config: Dict[str, Any]) -> str:
    # Prefer redistributed price if present
    preferred = config.get("target", {}).get("column_candidates", [
        "AI_Prezzo_Ridistribuito",
    ])
    for c in preferred:
        if c in df.columns:
            return c
    # Strict: raise with info
    available = [c for c in df.columns if "Prezzo" in c or "Valore" in c]
    raise ValueError(f"Nessuna colonna target trovata tra {preferred}. Disponibili simili: {available[:10]}")


def apply_target_transform_from_config(config: Dict[str, Any], y: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    """Apply target transformation based on config settings."""
    target_cfg = config.get("target", {})
    
    # Get transform type (new config format)
    transform_type = target_cfg.get("transform", "none")
    
    # Apply transformation
    kwargs = {}
    if transform_type == "log10":
        kwargs["log10_offset"] = target_cfg.get("log10_offset", 1.0)
    
    return apply_target_transform(y, transform_type=transform_type, **kwargs)


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
    logger.info(f"Caricamento raw completato: rows={len(df)}, cols={len(df.columns)}")

    # Apply data filters for experimentation (if configured)
    df = apply_data_filters(df, config)

    # Apply temporal and zone filtering to reduce drift
    temporal_cfg = config.get("temporal_filter", {})
    if temporal_cfg.get("enabled", False):
        initial_rows = len(df)
        
        # Filter by year
        min_year = temporal_cfg.get("min_year")
        if min_year and "A_AnnoStipula" in df.columns:
            df = df[df["A_AnnoStipula"] >= min_year]
            logger.info(f"Filtro temporale: A_AnnoStipula >= {min_year} → {len(df)} righe ({len(df)/initial_rows*100:.1f}%)")
        
        # Optional: filter by month
        min_month = temporal_cfg.get("min_month")
        if min_month and "A_MeseStipula" in df.columns:
            df = df[df["A_MeseStipula"] >= min_month]
            logger.info(f"Filtro mese: A_MeseStipula >= {min_month} → {len(df)} righe")
        
        # Filter out problematic zones
        exclude_zones = temporal_cfg.get("exclude_zones", [])
        if exclude_zones and "AI_ZonaOmi" in df.columns:
            df = df[~df["AI_ZonaOmi"].isin(exclude_zones)]
            logger.info(f"Filtro zone: escluse {exclude_zones} → {len(df)} righe ({len(df)/initial_rows*100:.1f}%)")
        
        # Filter out problematic building types
        exclude_tipologie = temporal_cfg.get("exclude_tipologie", [])
        if exclude_tipologie and "AI_IdTipologiaEdilizia" in df.columns:
            df = df[~df["AI_IdTipologiaEdilizia"].isin(exclude_tipologie)]
            logger.info(f"Filtro tipologie: escluse {exclude_tipologie} → {len(df)} righe ({len(df)/initial_rows*100:.1f}%)")
        
        removed_rows = initial_rows - len(df)
        logger.info(f"✅ Temporal filter: rimossi {removed_rows} campioni ({removed_rows/initial_rows*100:.1f}%)")
        if "AI_ZonaOmi" in df.columns:
            logger.info(f"   Dataset finale: {len(df)} righe su {df['AI_ZonaOmi'].nunique()} zone OMI")
        if "AI_IdTipologiaEdilizia" in df.columns:
            logger.info(f"   Tipologie residenziali: {sorted(df['AI_IdTipologiaEdilizia'].unique())}")

    # Convert datetime columns to strings once to prevent NaT-related issues downstream
    df = convert_datetime_columns_to_strings(df)

    # Initial cleanup
    prev_cols = len(df.columns)
    df = df.dropna(axis=1, how="all")
    logger.info(f"Drop colonne completamente vuote: {prev_cols - len(df.columns)} rimosse")

    # Feature extraction from JSON and geometry-like WKT columns
    fe_cfg = config.get("feature_extraction", {})
    logger.info(f"Feature extraction - geometry={bool(fe_cfg.get('geometry', True))}, json={bool(fe_cfg.get('json', True))}")
    if bool(fe_cfg.get("geometry", True)):
        df, drop_geom = extract_geometry_features(df)
    else:
        drop_geom = []
    if bool(fe_cfg.get("json", True)):
        df, drop_json = maybe_extract_json_features(df)
    else:
        drop_json = []
    # Extract GeoJSON-derived features (Polygon Feature/FeatureCollection)
    try:
        from preprocessing.feature_extractors import extract_geojson_polygon_features
        df, drop_geojson = extract_geojson_polygon_features(df)
    except Exception as _exc:
        drop_geojson = []
    cols_to_drop_now = list(set(drop_geom + drop_json + drop_geojson))
    if cols_to_drop_now:
        df = df.drop(columns=cols_to_drop_now, errors="ignore")
    logger.info(f"Estrazione feature: aggiunte/derivate, dropped_raw={len(cols_to_drop_now)} -> cols={len(df.columns)}")
    
    # Create missing pattern flags (e.g., has_CENED for energy certificates)
    df = create_missing_pattern_flags(df, config)

    # Feature pruning (generic): drop configured columns
    prune_cfg = config.get("feature_pruning", {})
    prune_cols_to_drop = prune_cfg.get(
        "drop_columns",
        [
            "A_ImmobiliPrincipaliConSuperficieValorizzata",
            "AI_SuperficieCalcolata",
            "AI_SuperficieVisuraTotale",
            "AI_SuperficieVisuraTotaleE",
            "AI_SuperficieVisuraTotaleAttuale",
            "AI_SuperficieVisuraTotaleEAttuale",
        ],
    )
    df = df.drop(columns=[c for c in prune_cols_to_drop if c in df.columns], errors="ignore")
    # Drop useless geo SRID (constant)
    df = df.drop(columns=[c for c in ["PC_PoligonoMetricoSrid"] if c in df.columns], errors="ignore")

    # AI_Piano features
    if "AI_Piano" in df.columns:
        try:
            from preprocessing.floor_parser import extract_floor_features_series
        except ImportError:
            extract_floor_features_series = None
        if extract_floor_features_series is not None:
            floor_feats = extract_floor_features_series(df["AI_Piano"])
            df = pd.concat([df, floor_feats], axis=1)
    # Remove AI_Piano raw after feature extraction
    df = df.drop(columns=[c for c in ["AI_Piano"] if c in df.columns], errors="ignore")

    # Keep only numeric part of AI_Civico
    if "AI_Civico" in df.columns:
        civico_num = df["AI_Civico"].astype(str).str.extract(r"(\d+)", expand=False)
        df["AI_Civico_num"] = pd.to_numeric(civico_num, errors="coerce")
        df = df.drop(columns=["AI_Civico"], errors="ignore")

    # Create temporal key to enable split later (do not leak across time)
    if "A_AnnoStipula" in df.columns and "A_MeseStipula" in df.columns:
        df["TemporalKey"] = df["A_AnnoStipula"].astype(int) * 100 + df["A_MeseStipula"].astype(int)
        logger.info("Creata colonna TemporalKey (A_AnnoStipula*100 + A_MeseStipula)")

    # Decide target (sempre AI_Prezzo_Ridistribuito ora)
    target_col = choose_target(df, config)
    logger.info(f"Target selezionato: {target_col}")

    # Create combined for split key if needed
    Xy_full = df.copy()

    # Temporal split FIRST to avoid leakage (contextual features AFTER split!)
    split_cfg_dict = config.get("temporal_split", {})
    mode = split_cfg_dict.get("mode", "date")
    frac = split_cfg_dict.get("fraction", {})
    date = split_cfg_dict.get("date", {})
    split_cfg = TemporalSplitConfig(
        year_col=split_cfg_dict.get("year_col", "A_AnnoStipula"),
        month_col=split_cfg_dict.get("month_col", "A_MeseStipula"),
        mode=mode,
        test_start_year=int(date.get("test_start_year", split_cfg_dict.get("test_start_year", 2023))),
        test_start_month=int(date.get("test_start_month", split_cfg_dict.get("test_start_month", 1))),
        train_fraction=float(frac.get("train", split_cfg_dict.get("train_fraction", DEFAULT_TEMPORAL_TRAIN_FRACTION))),
        valid_fraction=float(frac.get("valid", split_cfg_dict.get("valid_fraction", DEFAULT_TEMPORAL_VALID_FRACTION))),
    )
    train_df, val_df, test_df = temporal_split_3way(Xy_full, split_cfg)
    logger.info(
        f"Split temporale ({split_cfg.mode}) -> train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    
    # CRITICAL: Verify temporal order is preserved after split
    if "TemporalKey" in train_df.columns:
        if not train_df["TemporalKey"].is_monotonic_increasing:
            raise ValueError(
                "❌ TEMPORAL LEAKAGE RISK: Train set lost temporal order after split! "
                "TemporalKey must be monotonically increasing to prevent future data leaking into training."
            )
        logger.info(
            f"✅ Temporal order verified: Train [{train_df['TemporalKey'].min()} → {train_df['TemporalKey'].max()}], "
            f"Test [{test_df['TemporalKey'].min()} → {test_df['TemporalKey'].max()}]"
        )
        # Verify no overlap between train and test temporal ranges
        if train_df["TemporalKey"].max() >= test_df["TemporalKey"].min():
            logger.warning(
                f"⚠️  Temporal overlap detected: Train max ({train_df['TemporalKey'].max()}) >= "
                f"Test min ({test_df['TemporalKey'].min()}). This may indicate temporal leakage!"
            )

    # ==========================================
    # ADD CONTEXTUAL FEATURES (LEAK-FREE)
    # ==========================================
    # Fit ONLY on train, transform all splits
    # This captures market context: zone statistics, typology patterns, etc.
    # WITHOUT leaking test information into training!
    logger.info("🎯 Aggiunta feature contestuali (zona, tipologia, interazioni) [LEAK-FREE]...")
    train_df, val_df, test_df, contextual_stats = fit_transform_contextual_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_col=target_col
    )
    logger.info(f"✅ Feature contestuali aggiunte. Train cols: {len(train_df.columns)}")

    # Outlier detection ONLY on train target (optionally per category)
    out_cfg_dict = config.get("outliers", {})
    # Use global seed for reproducibility
    global_seed = config.get("training", {}).get("seed", 42)
    out_cfg = OutlierConfig(
        method=out_cfg_dict.get("method", "iqr"),
        z_thresh=float(out_cfg_dict.get("z_thresh", 4.0)),
        iqr_factor=float(out_cfg_dict.get("iqr_factor", 1.5)),
        iso_forest_contamination=float(out_cfg_dict.get("iso_forest_contamination", 0.02)),
        group_by_col=out_cfg_dict.get("group_by_col", "AI_IdCategoriaCatastale"),
        min_group_size=int(out_cfg_dict.get("min_group_size", 30)),
        fallback_strategy=str(out_cfg_dict.get("fallback_strategy", "skip")),
        random_state=int(out_cfg_dict.get("random_state", global_seed)),
    )
    before = len(train_df)
    inliers_mask = detect_outliers(train_df, target_col, out_cfg)
    train_df = train_df.loc[inliers_mask].copy()
    after = len(train_df)
    logger.info(f"Outlier detection: rimossi {before - after} record dal train ({(before-after)/max(1,before)*100:.2f}%)")

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

    # Preserve original-scale targets for evaluation
    y_test_orig = y_test.copy()
    y_val_orig = y_val.copy() if y_val is not None else None

    # Apply target transformation (fit on train, transform test/val with same params)
    y_train, transform_metadata = apply_target_transform_from_config(config, y_train)
    logger.info(f"Target transformation: {get_transform_name(transform_metadata)}")
    
    # For Box-Cox/Yeo-Johnson: use lambda fitted on train for test/val
    # For other transforms: apply same transformation directly
    transform_type = transform_metadata.get("transform")
    if transform_type == "boxcox":
        lambda_val = float(transform_metadata.get("lambda", transform_metadata.get("boxcox_lambda")))
        shift = float(transform_metadata.get("shift", transform_metadata.get("boxcox_shift", 0.0)))
        y_test = pd.Series(
            boxcox_transform(y_test.to_numpy(), lambda_val, shift),
            index=y_test.index,
            name=y_test.name,
        )
        if y_val is not None:
            y_val = pd.Series(
                boxcox_transform(y_val.to_numpy(), lambda_val, shift),
                index=y_val.index,
                name=y_val.name,
            )
    elif transform_type == "yeojohnson":
        lambda_val = float(transform_metadata.get("lambda", transform_metadata.get("yeojohnson_lambda")))
        y_test = pd.Series(
            yeojohnson_transform(y_test.to_numpy(), lambda_val),
            index=y_test.index,
            name=y_test.name,
        )
        if y_val is not None:
            y_val = pd.Series(
                yeojohnson_transform(y_val.to_numpy(), lambda_val),
                index=y_val.index,
                name=y_val.name,
            )
    elif transform_type != "none":
        # For log, log10, sqrt: apply directly (no fitting needed)
        y_test, _ = apply_target_transform(
            y_test, 
            transform_type=transform_type,
            **{k: v for k, v in transform_metadata.items() if k != "transform"}
        )
        if y_val is not None:
            y_val, _ = apply_target_transform(
                y_val,
                transform_type=transform_type,
                **{k: v for k, v in transform_metadata.items() if k != "transform"}
            )

    # Fit imputers on train, transform train/test
    imp_cfg_dict = config.get("imputation", {})
    imp_cfg = ImputationConfig(
        numeric_strategy=imp_cfg_dict.get("numeric_strategy", "median"),
        categorical_strategy=imp_cfg_dict.get("categorical_strategy", "most_frequent"),
        group_by_col=imp_cfg_dict.get("group_by_col", "AI_IdCategoriaCatastale"),
    )
    logger.info(
        f"Imputation: numeric={imp_cfg.numeric_strategy}, categorical={imp_cfg.categorical_strategy}, group_by={imp_cfg.group_by_col}"
    )
    fitted_imputers = fit_imputers(X_train, imp_cfg)
    X_train = transform_with_imputers(X_train, fitted_imputers)
    X_test = transform_with_imputers(X_test, fitted_imputers)
    if X_val is not None:
        X_val = transform_with_imputers(X_val, fitted_imputers)
    # Save imputers for inference reuse
    artifacts_dir = pre_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    dump(fitted_imputers, artifacts_dir / "imputers.joblib")

    # Prepare base copies before per-profile transformations
    base_train = X_train.copy(deep=False)
    base_test = X_test.copy(deep=False)
    base_val = X_val.copy(deep=False) if X_val is not None else None

    # Profile configuration
    profiles_cfg = config.get("profiles", {})
    na_thr = float(config.get("drop_non_descriptive", {}).get("na_threshold", 0.98))
    logger.info(f"Profili attivi: {[k for k,v in profiles_cfg.items() if v.get('enabled', False)]}")

    first_profile_saved = None
    saved_profiles: List[str] = []
    feature_columns_per_profile: Dict[str, List[str]] = {}

    def save_profile(X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr: pd.Series, y_te: pd.Series, X_va: pd.DataFrame | None, y_va: pd.Series | None, prefix: str):
        X_tr.to_parquet(pre_dir / f"X_train_{prefix}.parquet", index=False)
        X_te.to_parquet(pre_dir / f"X_test_{prefix}.parquet", index=False)
        pd.DataFrame({target_col: y_tr}).to_parquet(pre_dir / f"y_train_{prefix}.parquet", index=False)
        pd.DataFrame({target_col: y_te}).to_parquet(pre_dir / f"y_test_{prefix}.parquet", index=False)
        if X_va is not None and y_va is not None:
            X_va.to_parquet(pre_dir / f"X_val_{prefix}.parquet", index=False)
            pd.DataFrame({target_col: y_va}).to_parquet(pre_dir / f"y_val_{prefix}.parquet", index=False)
        # Save original-scale targets for evaluation
        if y_test_orig is not None:
            pd.DataFrame({target_col: y_test_orig}).to_parquet(pre_dir / f"y_test_orig_{prefix}.parquet", index=False)
        if y_val_orig is not None:
            pd.DataFrame({target_col: y_val_orig}).to_parquet(pre_dir / f"y_val_orig_{prefix}.parquet", index=False)
        logger.info(f"Profilo '{prefix}': salvati file train/val/test")
        saved_profiles.append(prefix)
        feature_columns_per_profile[prefix] = list(X_tr.columns)

        # Persist group-by columns sidecar for evaluation (from base_* before encoding)
        try:
            gm_cfg = (config.get("evaluation", {}) or {}).get("group_metrics", {}) or {}
            gb_cols = [c for c in (gm_cfg.get("group_by_columns", []) or []) if c in base_train.columns]
            if gb_cols:
                # Train
                _df = base_train.loc[X_tr.index, gb_cols] if set(X_tr.index) == set(base_train.index) else base_train[gb_cols]
                _df.to_parquet(pre_dir / f"group_cols_train_{prefix}.parquet", index=False)
                # Test
                _df = base_test.loc[X_te.index, gb_cols] if set(X_te.index) == set(base_test.index) else base_test[gb_cols]
                _df.to_parquet(pre_dir / f"group_cols_test_{prefix}.parquet", index=False)
                # Val (if exists)
                if X_va is not None and base_val is not None:
                    _df = base_val.loc[X_va.index, gb_cols] if set(X_va.index) == set(base_val.index) else base_val[gb_cols]
                    _df.to_parquet(pre_dir / f"group_cols_val_{prefix}.parquet", index=False)
                logger.info(f"Profilo '{prefix}': salvate colonne di gruppo per evaluation: {len(gb_cols)}")
        except Exception as _e:
            logger.warning(f"Impossibile salvare sidecar group-by columns per profilo '{prefix}': {_e}")

    # Helper: numeric coercion based on train
    def coerce_numeric_like(train_df: pd.DataFrame, other_dfs: List[pd.DataFrame | None]) -> Tuple[pd.DataFrame, List[pd.DataFrame | None]]:
        numc_cfg = config.get("numeric_coercion", {})
        enabled = bool(numc_cfg.get("enabled", True))
        threshold = float(numc_cfg.get("threshold", 0.95))
        patterns = [str(p) for p in (numc_cfg.get("blacklist_globs") or [
            "II_*",
            "AI_Id*",
            "Foglio",
            "Particella*",
            "Subalterno",
            "SezioneAmministrativa",
            "ZonaOmi",
            "*COD*",
        ])]
        patterns_upper = [p.upper() for p in patterns]
        train = train_df.copy()
        
        # CRITICAL: Replace NaT (Not a Time) values with None before any coercion
        # NaT cannot be converted to float and causes CatBoost errors
        for col in train.columns:
            if train[col].dtype == 'object':
                # Check if column contains NaT values
                try:
                    train[col] = train[col].replace({pd.NaT: None})
                except (TypeError, ValueError):
                    pass  # Column doesn't contain datetime-like values
        
        cols_to_coerce: List[str] = []
        if enabled:
            for col in train.select_dtypes(include=["object"]).columns:
                name_upper = (col or "").upper()
                if any(fnmatch.fnmatchcase(name_upper, pat) for pat in patterns_upper):
                    continue
                coerced = pd.to_numeric(train[col].astype(str).str.replace(",", "."), errors="coerce")
                if coerced.notna().mean() >= threshold:
                    cols_to_coerce.append(col)
                    train[col] = coerced
        if cols_to_coerce:
            logger.info(f"Coercizione numerica da stringhe: {len(cols_to_coerce)} colonne (soglia {threshold}, blacklist da config)")
        outs: List[pd.DataFrame | None] = []
        for df_ in other_dfs:
            if df_ is None:
                outs.append(None)
            else:
                tmp = df_.copy()
                # Replace NaT in other dataframes too
                for col in tmp.columns:
                    if tmp[col].dtype == 'object':
                        try:
                            tmp[col] = tmp[col].replace({pd.NaT: None})
                        except (TypeError, ValueError):
                            pass
                for col in cols_to_coerce:
                    if col in tmp.columns:
                        tmp[col] = pd.to_numeric(tmp[col].astype(str).str.replace(",", "."), errors="coerce")
                outs.append(tmp)
        return train, outs

    def _convert_optional(df_opt: pd.DataFrame | None) -> pd.DataFrame | None:
        return convert_datetime_columns_to_strings(df_opt) if df_opt is not None else None

    # Profile: scaled
    if profiles_cfg.get("scaled", {}).get("enabled", False):
        logger.info(f"[scaled] Starting multi-strategy encoding")
        X_tr = base_train.copy(); X_te = base_test.copy(); X_va = base_val.copy() if base_val is not None else None

        X_tr = convert_datetime_columns_to_strings(X_tr)
        X_te = _convert_optional(X_te)
        X_va = _convert_optional(X_va)

        # Coerce numeric-like strings before planning encoders
        X_tr, [X_te, X_va] = coerce_numeric_like(X_tr, [X_te, X_va])

        # Fit encoders ONLY on training data to prevent data leakage
        # Create profile-specific config with encoding overrides
        profile_config = config.copy()
        profile_encoding = profiles_cfg.get("scaled", {}).get("encoding", {})
        if profile_encoding:
            profile_config["encoding"] = {**config.get("encoding", {}), **profile_encoding}
        
        plan = plan_encodings(X_tr, profile_config)
        X_tr, encoders = fit_apply_encoders(X_tr, y_train, plan, profile_config)
        # Persist encoders
        _prof_dir = artifacts_dir / profiles_cfg.get("scaled", {}).get("output_prefix", "scaled")
        _prof_dir.mkdir(parents=True, exist_ok=True)
        dump(encoders, _prof_dir / "encoders.joblib")
        # Transform test and validation using fitted encoders (no leakage)
        X_te = transform_with_encoders(X_te, encoders)
        if X_va is not None:
            X_va = transform_with_encoders(X_va, encoders)
        prev_cols = len(X_tr.columns)
        X_tr, removed_nd = drop_non_descriptive(X_tr, na_threshold=na_thr)
        logger.info(f"[scaled] Drop non descrittive: {len(removed_nd)}")
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
        if wins_cfg.enabled:
            logger.info(f"[scaled] Winsorization: q=({wins_cfg.lower_quantile}, {wins_cfg.upper_quantile})")
        winsorizer = fit_winsorizer(X_tr_num, wins_cfg)
        # Persist winsorizer
        dump(winsorizer, _prof_dir / "winsorizer.joblib")
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
        logger.info(f"[scaled] Scaling: {scaling_cfg.scaler_type}, PCA: {pca_cfg.enabled}")
        X_tr_f, X_te_f, fitted = scale_and_pca(X_tr_num, X_te_num, scaling_cfg, pca_cfg)
        # Persist transforms (scaler + pca)
        dump(fitted, _prof_dir / "transforms.joblib")
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
        logger.info(f"[scaled] Pruning correlazioni: {len(dropped_corr)} colonne rimosse")
        X_te_f = X_te_f.drop(columns=[c for c in dropped_corr if c in X_te_f.columns], errors="ignore")
        if X_va_f is not None:
            X_va_f = X_va_f.drop(columns=[c for c in dropped_corr if c in X_va_f.columns], errors="ignore")
        prefix = profiles_cfg.get("scaled", {}).get("output_prefix", "scaled")
        save_profile(X_tr_f, X_te_f, y_train, y_test, X_va_f, y_val, prefix)
        if first_profile_saved is None:
            first_profile_saved = prefix

    # Profile: tree (no scaling/PCA)
    if profiles_cfg.get("tree", {}).get("enabled", False):
        logger.info(f"[tree] Starting multi-strategy encoding")
        X_tr = base_train.copy(); X_te = base_test.copy(); X_va = base_val.copy() if base_val is not None else None

        X_tr = convert_datetime_columns_to_strings(X_tr)
        X_te = _convert_optional(X_te)
        X_va = _convert_optional(X_va)

        # Coerce numeric-like strings before planning encoders
        X_tr, [X_te, X_va] = coerce_numeric_like(X_tr, [X_te, X_va])

        # Fit encoders ONLY on training data to prevent data leakage
        # Create profile-specific config with encoding overrides
        profile_config = config.copy()
        profile_encoding = profiles_cfg.get("tree", {}).get("encoding", {})
        if profile_encoding:
            profile_config["encoding"] = {**config.get("encoding", {}), **profile_encoding}
        
        plan = plan_encodings(X_tr, profile_config)
        X_tr, encoders = fit_apply_encoders(X_tr, y_train, plan, profile_config)
        
        # sklearn OneHotEncoder names columns as: originalname_value (e.g., AI_ZonaOmi_B1)
        # We need to identify them to preserve them from correlation pruning
        ohe_generated_cols = []
        if encoders.one_hot is not None and encoders.one_hot_input_cols:
            # Get all OHE feature names from sklearn encoder
            ohe_generated_cols = list(encoders.one_hot.get_feature_names_out(encoders.one_hot_input_cols))
        
        # Persist encoders
        _prof_dir = artifacts_dir / profiles_cfg.get("tree", {}).get("output_prefix", "tree")
        _prof_dir.mkdir(parents=True, exist_ok=True)
        dump(encoders, _prof_dir / "encoders.joblib")
        # Transform test and validation using fitted encoders (no leakage)
        X_te = transform_with_encoders(X_te, encoders)
        if X_va is not None:
            X_va = transform_with_encoders(X_va, encoders)
        # Riempie eventuali NaN introdotti dall'encoding ordinale (categorie sconosciute) con sentinel -1
        for _df in (X_tr, X_te, X_va):
            if _df is not None:
                _ord_cols = [c for c in _df.columns if c.endswith("__ord")]
                if _ord_cols:
                    _df[_ord_cols] = _df[_ord_cols].fillna(-1).astype(float)
        X_tr, removed_nd = drop_non_descriptive(X_tr, na_threshold=na_thr)
        logger.info(f"[tree] Drop non descrittive: {len(removed_nd)}")
        X_te = X_te.drop(columns=[c for c in removed_nd if c in X_te.columns], errors="ignore")
        if X_va is not None:
            X_va = X_va.drop(columns=[c for c in removed_nd if c in X_va.columns], errors="ignore")
        
        # Fill remaining NaN values
        for _df in (X_tr, X_te, X_va):
            if _df is not None:
                # Fill numeric NaN values with 0
                numeric_cols = _df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    _df[numeric_cols] = _df[numeric_cols].fillna(0)
        # Fill categorical NaN values with sentinel
                cat_cols = _df.select_dtypes(include=["object", "category"]).columns
                if len(cat_cols) > 0:
                    _df[cat_cols] = _df[cat_cols].fillna(MISSING_CATEGORY_SENTINEL)
        # Audit: log if any NaN remain
        for name, _df in [("X_tr", X_tr), ("X_te", X_te), ("X_va", X_va)]:
            if _df is not None and _df.isnull().any().any():
                logger.warning(f"[tree] Residual NaN detected after fills in {name}")
        
        # OHE features are binary (0/1) and should NOT be pruned based on correlation
        # They represent categorical information and high correlation is expected
        corr_thr = float(profiles_cfg.get("tree", {}).get("correlation", {}).get("numeric_threshold", config.get("correlation", {}).get("numeric_threshold", 0.98)))
        
        # Use OHE columns tracked during encoding
        ohe_cols = [c for c in ohe_generated_cols if c in X_tr.columns]
        logger.info(f"[tree] Identified {len(ohe_cols)} OHE features to preserve from correlation pruning")
        
        # Correlation pruning ONLY on numeric features that are NOT OHE
        X_tr_num = X_tr.select_dtypes(include=[np.number]).drop(columns=ohe_cols, errors="ignore")
        X_tr_num_pruned, dropped_corr = remove_highly_correlated(X_tr_num, threshold=corr_thr)
        logger.info(f"[tree] Pruning correlazioni numeriche (excluding OHE): {len(dropped_corr)} dropped")
        
        # Recombine: pruned numeric + ALL OHE + non-numeric (target/freq/ordinal encoded)
        kept_num_cols = X_tr_num_pruned.columns.tolist()
        X_tr_ohe = X_tr[ohe_cols] if ohe_cols else pd.DataFrame(index=X_tr.index)
        X_tr_other = X_tr.drop(columns=kept_num_cols + ohe_cols, errors="ignore")
        X_tr_final = pd.concat([X_tr_num_pruned, X_tr_ohe, X_tr_other], axis=1)
        
        X_te_ohe = X_te[ohe_cols] if ohe_cols else pd.DataFrame(index=X_te.index)
        X_te_other = X_te.drop(columns=kept_num_cols + ohe_cols, errors="ignore")
        X_te_final = pd.concat([X_te[kept_num_cols], X_te_ohe, X_te_other], axis=1)
        
        if X_va is not None:
            X_va_ohe = X_va[ohe_cols] if ohe_cols else pd.DataFrame(index=X_va.index)
            X_va_other = X_va.drop(columns=kept_num_cols + ohe_cols, errors="ignore")
            X_va_final = pd.concat([X_va[kept_num_cols], X_va_ohe, X_va_other], axis=1)
        else:
            X_va_final = None
        prefix = profiles_cfg.get("tree", {}).get("output_prefix", "tree")
        save_profile(X_tr_final, X_te_final, y_train, y_test, X_va_final, y_val, prefix)
        if first_profile_saved is None:
            first_profile_saved = prefix

    # Profile: catboost (preserve categoricals)
    if profiles_cfg.get("catboost", {}).get("enabled", False):
        logger.info("[catboost] Preservazione categoriche; niente OHE")
        X_tr = base_train.copy(); X_te = base_test.copy(); X_va = base_val.copy() if base_val is not None else None

        X_tr = convert_datetime_columns_to_strings(X_tr)
        X_te = _convert_optional(X_te)
        X_va = _convert_optional(X_va)

        # Replace NaT values in all columns (both object and numeric)
        # NaT cannot be converted to float and causes CatBoost errors
        for _df in (X_tr, X_te, X_va):
            if _df is not None:
                for col in _df.columns:
                    try:
                        if _df[col].dtype == 'object':
                            _df[col] = _df[col].replace({pd.NaT: None})
                        else:
                            _df[col] = _df[col].replace({pd.NaT: np.nan})
                    except (TypeError, ValueError, AttributeError):
                        pass

        # Coerce numeric-like strings to numeric
        X_tr, [X_te, X_va] = coerce_numeric_like(X_tr, [X_te, X_va])
        
        # Drop extreme high-cardinality categorical features
        extreme_card_threshold = int(config.get("encoding", {}).get("catboost_max_cardinality", 500))
        cat_cols = X_tr.select_dtypes(include=['object', 'category']).columns
        extreme_high_card = []
        for col in cat_cols:
            nunique = X_tr[col].nunique(dropna=True)
            if nunique > extreme_card_threshold:
                extreme_high_card.append(col)
                logger.warning(
                    f"[catboost] Dropping extreme high-cardinality column: {col} "
                    f"({nunique} unique values > {extreme_card_threshold} threshold)"
                )
        
        if extreme_high_card:
            X_tr = X_tr.drop(columns=extreme_high_card)
            X_te = X_te.drop(columns=extreme_high_card, errors="ignore")
            if X_va is not None:
                X_va = X_va.drop(columns=extreme_high_card, errors="ignore")
            logger.info(f"[catboost] Dropped {len(extreme_high_card)} extreme high-cardinality columns")
        
        # Drop non-descriptive
        X_tr, removed_nd = drop_non_descriptive(X_tr, na_threshold=na_thr)
        logger.info(f"[catboost] Drop non descrittive: {len(removed_nd)}")
        X_te = X_te.drop(columns=[c for c in removed_nd if c in X_te.columns], errors="ignore")
        if X_va is not None:
            X_va = X_va.drop(columns=[c for c in removed_nd if c in X_va.columns], errors="ignore")
        
        # Fill remaining NaN values
        for _df in (X_tr, X_te, X_va):
            if _df is not None:
                # Fill numeric NaN values with 0
                numeric_cols = _df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    _df[numeric_cols] = _df[numeric_cols].fillna(0)
        # Fill categorical NaN values with sentinel
                cat_cols = _df.select_dtypes(include=["object", "category"]).columns
                if len(cat_cols) > 0:
                    _df[cat_cols] = _df[cat_cols].fillna(MISSING_CATEGORY_SENTINEL)
        
        # Numeric-only correlation prune
        corr_thr = float(profiles_cfg.get("catboost", {}).get("correlation", {}).get("numeric_threshold", config.get("correlation", {}).get("numeric_threshold", 0.98)))
        X_tr_num = X_tr.select_dtypes(include=[np.number])
        X_tr_num_pruned, dropped_corr = remove_highly_correlated(X_tr_num, threshold=corr_thr)
        logger.info(f"[catboost] Pruning correlazioni numeriche: {len(dropped_corr)}")
        kept_num_cols = X_tr_num_pruned.columns
        X_tr_final = pd.concat([X_tr[kept_num_cols], X_tr.drop(columns=kept_num_cols, errors="ignore").select_dtypes(exclude=[np.number])], axis=1)
        X_te_final = pd.concat([X_te[kept_num_cols], X_te.drop(columns=kept_num_cols, errors="ignore").select_dtypes(exclude=[np.number])], axis=1)
        if X_va is not None:
            X_va_final = pd.concat([X_va[kept_num_cols], X_va.drop(columns=kept_num_cols, errors="ignore").select_dtypes(exclude=[np.number])], axis=1)
        else:
            X_va_final = None
        # Audit: ensure no NaN in numeric parts post-prune
        for name, _df in [("X_tr_final", X_tr_final), ("X_te_final", X_te_final), ("X_va_final", X_va_final)]:
            if _df is not None and _df.select_dtypes(include=[np.number]).isnull().any().any():
                logger.warning(f"[catboost] Residual NaN detected in numeric columns of {name}, filling with 0")
                num_cols = _df.select_dtypes(include=[np.number]).columns
                _df[num_cols] = _df[num_cols].fillna(0)
        
        # CRITICAL: Final check for NaT values before saving
        # Convert any remaining NaT to appropriate missing values
        for name, _df in [("X_tr_final", X_tr_final), ("X_te_final", X_te_final), ("X_va_final", X_va_final)]:
            if _df is not None:
                for col in _df.columns:
                    # Check each cell for NaT (pandas NaT is a singleton)
                    if _df[col].dtype == 'object':
                        # For object columns, check if any values are NaT
                        mask = _df[col].apply(lambda x: pd.isna(x) if not isinstance(x, str) else False)
                        if mask.any():
                            _df.loc[mask, col] = None
                            logger.info(f"[catboost] Replaced {mask.sum()} NaT/NA values with None in {name}.{col}")
                    elif pd.api.types.is_numeric_dtype(_df[col]):
                        # For numeric columns, ensure no object-type NaT leaked through
                        try:
                            _df[col] = pd.to_numeric(_df[col], errors='coerce')
                        except (TypeError, ValueError):
                            pass
        
        prefix = profiles_cfg.get("catboost", {}).get("output_prefix", "catboost")
        save_profile(X_tr_final, X_te_final, y_train, y_test, X_va_final, y_val, prefix)
        # Save list of categorical columns for catboost
        cat_cols = X_tr_final.select_dtypes(include=["object", "category"]).columns.tolist()
        (pre_dir / f"categorical_columns_{prefix}.txt").write_text("\n".join(cat_cols), encoding="utf-8")
        logger.info(f"[catboost] Salvate {len(cat_cols)} colonne categoriche")
        if first_profile_saved is None:
            first_profile_saved = prefix


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

    # Save preprocessing info for evaluation
    prep_info: Dict[str, Any] = {
        "target_column": target_col,
        "target_transformation": transform_metadata,  # Contains transform type + parameters (lambda, offset, etc.)
        "profiles_saved": saved_profiles,
        "feature_columns_per_profile": feature_columns_per_profile,
    }
    try:
        save_json(prep_info, str(pre_dir / "preprocessing_info.json"))
    except Exception as _e:
        logger.warning(f"Impossibile salvare preprocessing_info.json: {_e}")

    logger.info(f"Preprocessing completato. Output in {pre_dir}")
    return pre_dir