from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import re
import fnmatch

from utils.logger import get_logger
from utils.detailed_logging import DetailedLogger
from utils.robust_operations import RobustDataOperations, RobustColumnAnalyzer
from utils.temporal_advanced import AdvancedTemporalUtils, TemporalSplitter
from utils.smart_config import SmartConfigurationManager
from validation.quality_checks import QualityChecker
from preprocessing.pipeline_tracker import PipelineTracker
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
from joblib import dump

logger = get_logger(__name__)


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


def apply_log_target_if(config: Dict[str, Any], y: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    use_log = bool(config.get("target", {}).get("log_transform", False))
    if not use_log:
        return y, {"log": False}
    y_pos = y.clip(lower=1e-6)
    return np.log1p(y_pos), {"log": True}


def run_preprocessing(config: Dict[str, Any]) -> Path:
    """
    Esegue la pipeline completa di preprocessing con tracking avanzato e quality checks.
    
    La pipeline include:
    - Caricamento e validazione dati iniziali
    - Feature extraction da geometrie WKT e dati JSON
    - Pulizia robusta con gestione errori
    - Split temporale con validazione integrità
    - Quality checks automatici per data leakage
    - Tracking completo evoluzione dataset
    
    Args:
        config: Configurazione pipeline con sezioni per paths, feature_extraction,
                quality_checks, tracking, temporal_split, etc.
                
    Returns:
        Path al file preprocessed finale
    """
    # Inizializza smart configuration manager per risoluzione automatica
    smart_config = SmartConfigurationManager()
    smart_config.config = config
    
    # Inizializza pipeline tracker per monitoraggio evoluzione
    tracker = PipelineTracker(config)
    tracker.track_step_start('initialization')
    
    # Setup directories con creazione automatica
    paths = config.get("paths", {})
    raw_dir = Path(paths.get("raw_data", "data/raw"))
    pre_dir = Path(paths.get("preprocessed_data", "data/preprocessed"))
    reports_dir = pre_dir / "reports"
    pre_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Caricamento dati con validazione robusta
    raw_files = list(raw_dir.glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"Nessun file parquet trovato in {raw_dir}")

    df_initial = pd.read_parquet(raw_files[0])
    logger.info(f"Dataset caricato: {len(df_initial):,} righe × {len(df_initial.columns)} colonne")
    
    # Risoluzione intelligente colonne target e temporali
    target_resolution = smart_config.resolve_target_columns(df_initial)
    temporal_resolution = smart_config.resolve_temporal_columns(df_initial)
    
    logger.info(f"Target risolto: {target_resolution['target_column']} "
               f"(metodo: {target_resolution['resolution_method']})")
    logger.info(f"Temporale disponibile: {temporal_resolution['temporal_available']}")
    
    df = df_initial.copy()

    # Pulizia iniziale con tracking evoluzione
    tracker.track_step_start('initial_cleanup')
    df_before_cleanup = df.copy()
    
    # Rimuovi colonne completamente vuote usando operazioni robuste
    empty_columns = [col for col in df.columns if df[col].isna().all()]
    df, cleanup_info = RobustDataOperations.remove_columns_safe(
        df, empty_columns, "PULIZIA COLONNE VUOTE"
    )
    
    tracker.track_step_completion(
        'initial_cleanup', df_before_cleanup, df, cleanup_info
    )

    # Feature extraction con gestione robusta degli errori
    tracker.track_step_start('feature_extraction')
    df_before_fe = df.copy()
    
    fe_cfg = config.get("feature_extraction", {})
    extraction_results = {}
    
    # Estrazione da geometrie WKT se abilitata
    if fe_cfg.get('geometry', True):
        try:
            df, drop_geom = extract_geometry_features(df)
            extraction_results['geometry'] = {
                'enabled': True,
                'columns_dropped': drop_geom,
                'success': True
            }
        except Exception as e:
            logger.warning(f"Feature extraction geometrie fallita: {e}")
            drop_geom = []
            extraction_results['geometry'] = {
                'enabled': True,
                'success': False,
                'error': str(e)
            }
    else:
        drop_geom = []
        extraction_results['geometry'] = {'enabled': False}
    
    # Estrazione da dati JSON se abilitata
    if fe_cfg.get('json', True):
        try:
            df, drop_json = maybe_extract_json_features(df)
            extraction_results['json'] = {
                'enabled': True,
                'columns_dropped': drop_json,
                'success': True
            }
        except Exception as e:
            logger.warning(f"Feature extraction JSON fallita: {e}")
            drop_json = []
            extraction_results['json'] = {
                'enabled': True,
                'success': False,
                'error': str(e)
            }
    else:
        drop_json = []
        extraction_results['json'] = {'enabled': False}
    
    # Estrazione GeoJSON con fallback robusto
    try:
        from preprocessing.feature_extractors import extract_geojson_polygon_features
        df, drop_geojson = extract_geojson_polygon_features(df)
        extraction_results['geojson'] = {
            'enabled': True,
            'columns_dropped': drop_geojson,
            'success': True
        }
    except Exception as e:
        logger.warning(f"Feature extraction GeoJSON fallita: {e}")
        drop_geojson = []
        extraction_results['geojson'] = {
            'enabled': True,
            'success': False,
            'error': str(e)
        }
    
    # Rimuovi colonne raw processate con operazioni sicure
    cols_to_drop = list(set(drop_geom + drop_json + drop_geojson))
    if cols_to_drop:
        df, drop_info = RobustDataOperations.remove_columns_safe(
            df, cols_to_drop, "RIMOZIONE COLONNE RAW POST-EXTRACTION"
        )
    
    # Traccia risultati feature extraction
    tracker.track_feature_engineering(
        list(df_before_fe.columns), list(df.columns), extraction_results
    )
    
    tracker.track_step_completion(
        'feature_extraction', df_before_fe, df, extraction_results
    )

    # Pulizia colonne superficie ridondanti con tracking
    tracker.track_step_start('surface_cleanup')
    df_before_surface = df.copy()
    
    surface_cfg = config.get("surface", {})
    surface_cols_to_drop = surface_cfg.get(
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
    
    # Rimuovi colonne superficie ridondanti mantenendo solo AI_Superficie canonica
    df, surface_removal_info = RobustDataOperations.remove_columns_safe(
        df, surface_cols_to_drop, "PULIZIA COLONNE SUPERFICIE RIDONDANTI"
    )
    
    # Rimuovi colonne costanti geografiche (SRID sempre uguale)
    geo_constants = ["PC_PoligonoMetricoSrid"]
    df, geo_removal_info = RobustDataOperations.remove_columns_safe(
        df, geo_constants, "RIMOZIONE COSTANTI GEOGRAFICHE"
    )
    
    surface_cleanup_info = {
        'surface_columns_removed': surface_removal_info.get('existing_columns', []),
        'geo_constants_removed': geo_removal_info.get('existing_columns', [])
    }
    
    tracker.track_step_completion(
        'surface_cleanup', df_before_surface, df, surface_cleanup_info
    )

    # Processing colonna AI_Piano con feature engineering avanzato
    tracker.track_step_start('piano_processing')
    df_before_piano = df.copy()
    
    piano_features_added = []
    if "AI_Piano" in df.columns:
        try:
            from preprocessing.floor_parser import extract_floor_features_series
            floor_feats = extract_floor_features_series(df["AI_Piano"])
            df = pd.concat([df, floor_feats], axis=1)
            piano_features_added = list(floor_feats.columns)
            logger.info(f"Feature estratte da AI_Piano: {piano_features_added}")
        except ImportError as e:
            logger.warning(f"Floor parser non disponibile: {e}")
        except Exception as e:
            logger.warning(f"Estrazione feature piano fallita: {e}")
    
    # Rimuovi colonna raw AI_Piano dopo estrazione
    df, piano_removal_info = RobustDataOperations.remove_columns_safe(
        df, ["AI_Piano"], "RIMOZIONE AI_PIANO RAW POST-EXTRACTION"
    )
    
    piano_processing_info = {
        'features_added': piano_features_added,
        'raw_column_removed': piano_removal_info.get('existing_columns', [])
    }
    
    tracker.track_step_completion(
        'piano_processing', df_before_piano, df, piano_processing_info
    )

    # Processing AI_Civico: estrai solo parte numerica
    tracker.track_step_start('civico_processing')
    df_before_civico = df.copy()
    
    civico_processing_info = {'processed': False}
    if "AI_Civico" in df.columns:
        try:
            # Estrai numero civico con regex robusto
            civico_num = df["AI_Civico"].astype(str).str.extract(r"(\d+)", expand=False)
            df["AI_Civico_num"] = pd.to_numeric(civico_num, errors="coerce")
            
            # Rimuovi colonna originale
            df = df.drop(columns=["AI_Civico"], errors="ignore")
            
            civico_processing_info = {
                'processed': True,
                'feature_created': 'AI_Civico_num',
                'original_removed': 'AI_Civico'
            }
            logger.info("AI_Civico processato: estratta parte numerica")
        except Exception as e:
            logger.warning(f"Processing AI_Civico fallito: {e}")
            civico_processing_info = {'processed': False, 'error': str(e)}
    
    tracker.track_step_completion(
        'civico_processing', df_before_civico, df, civico_processing_info
    )

    # Crea feature temporali avanzate se disponibili colonne temporali
    if temporal_resolution['temporal_available']:
        tracker.track_step_start('temporal_features')
        df_before_temporal = df.copy()
        
        year_col = temporal_resolution['year_column']
        month_col = temporal_resolution['month_column']
        
        # Crea feature temporali usando utility avanzate
        df = AdvancedTemporalUtils.create_temporal_features(
            df, year_col, month_col, prefix="temporal_"
        )
        
        # Mantieni anche TemporalKey originale per compatibilità
        df["TemporalKey"] = df[year_col].astype(int) * 100 + df[month_col].astype(int)
        
        temporal_features_info = {
            'temporal_key_created': True,
            'advanced_features_added': [col for col in df.columns if col.startswith('temporal_')],
            'year_column': year_col,
            'month_column': month_col
        }
        
        tracker.track_step_completion(
            'temporal_features', df_before_temporal, df, temporal_features_info
        )
    else:
        logger.warning("Colonne temporali non disponibili - skip creazione feature temporali")

    # Risoluzione target usando smart config o fallback
    if target_resolution['target_column']:
        target_col = target_resolution['target_column']
    else:
        target_col = choose_target(df, config)
    
    logger.info(f"Target finale: {target_col}")

    # Split temporale avanzato con validazione integrità
    tracker.track_step_start('temporal_split')
    
    if temporal_resolution['temporal_available']:
        # Usa splitter temporale avanzato con validazione
        year_col = temporal_resolution['year_column']
        month_col = temporal_resolution['month_column']
        
        split_cfg_dict = config.get("temporal_split", {})
        train_fraction = float(split_cfg_dict.get("train_fraction", 0.7))
        valid_fraction = float(split_cfg_dict.get("valid_fraction", 0.15))
        test_fraction = 1.0 - train_fraction - valid_fraction
        
        # Split con validazione automatica
        train_df, val_df, test_df, split_info = TemporalSplitter.split_temporal_with_validation(
            df, year_col, month_col, train_fraction, valid_fraction, test_fraction
        )
        
        # Log dettagliato range temporali
        DetailedLogger.log_split_temporal_ranges(df, year_col, month_col, len(train_df), len(train_df) + len(val_df))
        
        split_method = 'advanced_temporal'
        
    else:
        # Fallback a split originale se temporale non disponibile
        logger.warning("Split temporale non disponibile - usando split originale")
        split_cfg_dict = config.get("temporal_split", {})
        split_cfg = TemporalSplitConfig(
            year_col=split_cfg_dict.get("year_col", "A_AnnoStipula"),
            month_col=split_cfg_dict.get("month_col", "A_MeseStipula"),
            mode=split_cfg_dict.get("mode", "fraction"),
            test_start_year=int(split_cfg_dict.get("test_start_year", 2023)),
            test_start_month=int(split_cfg_dict.get("test_start_month", 1)),
            train_fraction=float(split_cfg_dict.get("train_fraction", 0.8)),
            valid_fraction=float(split_cfg_dict.get("valid_fraction", 0.0)),
        )
        train_df, val_df, test_df = temporal_split_3way(df, split_cfg)
        split_info = {'fallback_used': True}
        split_method = 'legacy_temporal'
    
    logger.info(f"Split completato ({split_method}): train={len(train_df):,}, "
               f"val={len(val_df):,}, test={len(test_df):,}")
    
    # Traccia informazioni split
    split_tracking_info = {
        'method': split_method,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'split_info': split_info,
        'temporal_available': temporal_resolution['temporal_available']
    }
    
    tracker.track_step_completion(
        'temporal_split', df, df, split_tracking_info  # df rimane uguale, cambia solo split
    )

    # Outlier detection robusto solo su training set per evitare leakage
    tracker.track_step_start('outlier_detection')
    train_df_before_outliers = train_df.copy()
    
    out_cfg_dict = config.get("outliers", {})
    global_seed = config.get("training", {}).get("seed", 42)
    
    # Risolvi colonna per stratificazione usando smart config
    categorical_resolution = smart_config.resolve_categorical_columns(train_df)
    group_by_col = categorical_resolution.get('category_column_for_outliers')
    
    # Se non trovata, usa fallback dalla configurazione
    if not group_by_col:
        group_by_col = out_cfg_dict.get("group_by_col", "AI_IdCategoriaCatastale")
        logger.warning(f"Colonna stratificazione non trovata automaticamente, uso: {group_by_col}")
    
    out_cfg = OutlierConfig(
        method=out_cfg_dict.get("method", "iqr"),
        z_thresh=float(out_cfg_dict.get("z_thresh", 4.0)),
        iqr_factor=float(out_cfg_dict.get("iqr_factor", 1.5)),
        iso_forest_contamination=float(out_cfg_dict.get("iso_forest_contamination", 0.02)),
        group_by_col=group_by_col,
        min_group_size=int(out_cfg_dict.get("min_group_size", 30)),
        fallback_strategy=str(out_cfg_dict.get("fallback_strategy", "global")),
        random_state=int(out_cfg_dict.get("random_state", global_seed)),
    )
    
    # Esegui detection con tracking dettagliato
    before_outliers = len(train_df)
    inliers_mask = detect_outliers(train_df, target_col, out_cfg)
    train_df = train_df.loc[inliers_mask].copy()
    after_outliers = len(train_df)
    
    outliers_removed = before_outliers - after_outliers
    outlier_percentage = (outliers_removed / before_outliers) * 100 if before_outliers > 0 else 0
    
    # Crea info dettagliate per tracking
    outliers_info = {
        'total_outliers': outliers_removed,
        'outlier_percentage': outlier_percentage,
        'method_used': out_cfg.method,
        'group_by_column': group_by_col,
        'samples_before': before_outliers,
        'samples_after': after_outliers,
        'by_method': {out_cfg.method: outliers_removed}
    }
    
    # Traccia risultati outlier detection
    tracker.track_outlier_detection(outliers_info, before_outliers, outliers_removed)
    
    tracker.track_step_completion(
        'outlier_detection', train_df_before_outliers, train_df, outliers_info
    )
    
    logger.info(f"Outlier detection completato: rimossi {outliers_removed:,} record "
               f"({outlier_percentage:.2f}%) dal training set")

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
        y_test = np.log1p(y_test.clip(lower=1e-6))
        if y_val is not None:
            y_val = np.log1p(y_val.clip(lower=1e-6))
        logger.info("Applicata trasformazione log1p al target")

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
    base_train = X_train.copy()
    base_test = X_test.copy()
    base_val = X_val.copy() if X_val is not None else None

    # Optional profile-level thresholds
    profiles_cfg = config.get("profiles", {})
    # Provide fallback defaults only if profiles is completely missing from config
    if not profiles_cfg:
        profiles_cfg = {
            "scaled": {"enabled": True, "output_prefix": "scaled"},
            "tree": {"enabled": False, "output_prefix": "tree"},
            "catboost": {"enabled": False, "output_prefix": "catboost"},
        }
    # NA threshold for dropping non-descriptive columns
    na_thr = float(config.get("drop_non_descriptive", {}).get("na_threshold", 0.98))
    logger.info(f"Profili attivi: {[k for k,v in profiles_cfg.items() if v.get('enabled', False)]}")

    first_profile_saved = None

    def save_profile(X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr: pd.Series, y_te: pd.Series, X_va: pd.DataFrame | None, y_va: pd.Series | None, prefix: str):
        X_tr.to_parquet(pre_dir / f"X_train_{prefix}.parquet", index=False)
        X_te.to_parquet(pre_dir / f"X_test_{prefix}.parquet", index=False)
        pd.DataFrame({target_col: y_tr}).to_parquet(pre_dir / f"y_train_{prefix}.parquet", index=False)
        pd.DataFrame({target_col: y_te}).to_parquet(pre_dir / f"y_test_{prefix}.parquet", index=False)
        if X_va is not None and y_va is not None:
            X_va.to_parquet(pre_dir / f"X_val_{prefix}.parquet", index=False)
            pd.DataFrame({target_col: y_va}).to_parquet(pre_dir / f"y_val_{prefix}.parquet", index=False)
        logger.info(f"Profilo '{prefix}': salvati file train/val/test")

    # Helper: numeric coercion based on train
    def coerce_numeric_like(train_df: pd.DataFrame, other_dfs: List[pd.DataFrame | None]) -> Tuple[pd.DataFrame, List[pd.DataFrame | None]]:
        numc_cfg = config.get("numeric_coercion", {})
        enabled = bool(numc_cfg.get("enabled", True))
        threshold = float(numc_cfg.get("threshold", 0.95))
        patterns = [str(p) for p in numc_cfg.get("blacklist_patterns", [
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
                for col in cols_to_coerce:
                    if col in tmp.columns:
                        tmp[col] = pd.to_numeric(tmp[col].astype(str).str.replace(",", "."), errors="coerce")
                outs.append(tmp)
        return train, outs

    # Profile: scaled
    if profiles_cfg.get("scaled", {}).get("enabled", False):
        enc_max = int(profiles_cfg.get("scaled", {}).get("encoding", {}).get("max_ohe_cardinality", config.get("encoding", {}).get("max_ohe_cardinality", 12)))
        logger.info(f"[scaled] Encoding plan con max_ohe_cardinality={enc_max}")
        X_tr = base_train.copy(); X_te = base_test.copy(); X_va = base_val.copy() if base_val is not None else None
        # CRITICAL: Fit encoders ONLY on training data to prevent data leakage
        plan = plan_encodings(X_tr, max_ohe_cardinality=enc_max)
        X_tr, encoders, _ = fit_apply_encoders(X_tr, plan)
        # Persist encoders
        _prof_dir = artifacts_dir / profiles_cfg.get("scaled", {}).get("output_prefix", "scaled")
        _prof_dir.mkdir(parents=True, exist_ok=True)
        dump(encoders, _prof_dir / "encoders.joblib")
        # Transform test and validation using fitted encoders (no leakage)
        X_te = transform_with_encoders(X_te, encoders)
        if X_va is not None:
            X_va = transform_with_encoders(X_va, encoders)
        X_tr, [X_te, X_va] = coerce_numeric_like(X_tr, [X_te, X_va])
        prev_cols = len(X_tr.columns)
        # Rimozione colonne non descrittive con operazioni robuste
        non_descriptive_cols, nd_stats = RobustColumnAnalyzer.analyze_missing_values(
            X_tr, threshold=na_thr
        )
        X_tr, nd_removal_info = RobustDataOperations.remove_columns_safe(
            X_tr, non_descriptive_cols, f"[scaled] RIMOZIONE COLONNE NON DESCRITTIVE"
        )
        
        # Applica rimozione anche a test e validation
        if non_descriptive_cols:
            X_te, _ = RobustDataOperations.remove_columns_safe(
                X_te, non_descriptive_cols, f"[scaled] SYNC RIMOZIONE TEST"
            )
            if X_va is not None:
                X_va, _ = RobustDataOperations.remove_columns_safe(
                    X_va, non_descriptive_cols, f"[scaled] SYNC RIMOZIONE VAL"
                )
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
        # Pruning correlazioni con analisi robusta
        corr_thr = float(profiles_cfg.get("scaled", {}).get("correlation", {}).get("numeric_threshold", 
                                                           config.get("correlation", {}).get("numeric_threshold", 0.98)))
        
        # Usa analyzer robusto per trovare correlazioni
        corr_cols_to_remove, corr_stats = RobustColumnAnalyzer.find_highly_correlated_columns(
            X_tr_f, threshold=corr_thr
        )
        
        # Rimuovi con operazioni robuste
        X_tr_f, corr_removal_info = RobustDataOperations.remove_columns_safe(
            X_tr_f, corr_cols_to_remove, f"[scaled] PRUNING CORRELAZIONI (soglia={corr_thr})"
        )
        
        # Sincronizza rimozione su test e validation
        if corr_cols_to_remove:
            X_te_f, _ = RobustDataOperations.remove_columns_safe(
                X_te_f, corr_cols_to_remove, "[scaled] SYNC CORRELAZIONI TEST"
            )
            if X_va_f is not None:
                X_va_f, _ = RobustDataOperations.remove_columns_safe(
                    X_va_f, corr_cols_to_remove, "[scaled] SYNC CORRELAZIONI VAL"
                )
        prefix = profiles_cfg.get("scaled", {}).get("output_prefix", "scaled")
        save_profile(X_tr_f, X_te_f, y_train, y_test, X_va_f, y_val, prefix)
        if first_profile_saved is None:
            first_profile_saved = prefix

    # Profile: tree (no scaling/PCA)
    if profiles_cfg.get("tree", {}).get("enabled", False):
        enc_max = int(profiles_cfg.get("tree", {}).get("encoding", {}).get("max_ohe_cardinality", config.get("encoding", {}).get("max_ohe_cardinality", 12)))
        logger.info(f"[tree] Encoding plan con max_ohe_cardinality={enc_max}")
        X_tr = base_train.copy(); X_te = base_test.copy(); X_va = base_val.copy() if base_val is not None else None
        # CRITICAL: Fit encoders ONLY on training data to prevent data leakage
        plan = plan_encodings(X_tr, max_ohe_cardinality=enc_max)
        X_tr, encoders, _ = fit_apply_encoders(X_tr, plan)
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
        X_tr, [X_te, X_va] = coerce_numeric_like(X_tr, [X_te, X_va])
        # Rimozione colonne non descrittive con operazioni robuste
        non_descriptive_cols, nd_stats = RobustColumnAnalyzer.analyze_missing_values(
            X_tr, threshold=na_thr
        )
        X_tr, nd_removal_info = RobustDataOperations.remove_columns_safe(
            X_tr, non_descriptive_cols, f"[tree] RIMOZIONE COLONNE NON DESCRITTIVE"
        )
        
        # Sincronizza rimozione su test e validation
        if non_descriptive_cols:
            X_te, _ = RobustDataOperations.remove_columns_safe(
                X_te, non_descriptive_cols, "[tree] SYNC RIMOZIONE TEST"
            )
            if X_va is not None:
                X_va, _ = RobustDataOperations.remove_columns_safe(
                    X_va, non_descriptive_cols, "[tree] SYNC RIMOZIONE VAL"
                )
        
        # Fill any remaining NaN values to ensure compatibility with all sklearn models
        for _df in (X_tr, X_te, X_va):
            if _df is not None:
                # Fill numeric NaN values with 0
                numeric_cols = _df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    _df[numeric_cols] = _df[numeric_cols].fillna(0)
                # Fill categorical NaN values with "UNKNOWN"
                cat_cols = _df.select_dtypes(include=["object", "category"]).columns
                if len(cat_cols) > 0:
                    _df[cat_cols] = _df[cat_cols].fillna("UNKNOWN")
        # Audit: log if any NaN remain
        for name, _df in [("X_tr", X_tr), ("X_te", X_te), ("X_va", X_va)]:
            if _df is not None and _df.isnull().any().any():
                logger.warning(f"[tree] Residual NaN detected after fills in {name}")
        
        # Pruning correlazioni numeriche con analisi robusta
        corr_thr = float(profiles_cfg.get("tree", {}).get("correlation", {}).get("numeric_threshold", 
                                                         config.get("correlation", {}).get("numeric_threshold", 0.98)))
        
        # Analizza correlazioni solo su colonne numeriche
        X_tr_numeric = X_tr.select_dtypes(include=[np.number])
        corr_cols_to_remove, corr_stats = RobustColumnAnalyzer.find_highly_correlated_columns(
            X_tr_numeric, threshold=corr_thr
        )
        
        # Rimuovi colonne correlate mantenendo struttura categoriche + numeriche
        if corr_cols_to_remove:
            # Rimuovi dalle numeriche
            X_tr_num_clean, _ = RobustDataOperations.remove_columns_safe(
                X_tr_numeric, corr_cols_to_remove, f"[tree] PRUNING CORRELAZIONI NUMERICHE (soglia={corr_thr})"
            )
            
            # Ricostruisci dataset completo
            X_tr_categorical = X_tr.select_dtypes(exclude=[np.number])
            X_tr_final = pd.concat([X_tr_num_clean, X_tr_categorical], axis=1)
            
            # Sincronizza su test e validation
            X_te_num_clean, _ = RobustDataOperations.remove_columns_safe(
                X_te.select_dtypes(include=[np.number]), corr_cols_to_remove, "[tree] SYNC CORRELAZIONI TEST"
            )
            X_te_categorical = X_te.select_dtypes(exclude=[np.number])
            X_te_final = pd.concat([X_te_num_clean, X_te_categorical], axis=1)
            
            if X_va is not None:
                X_va_num_clean, _ = RobustDataOperations.remove_columns_safe(
                    X_va.select_dtypes(include=[np.number]), corr_cols_to_remove, "[tree] SYNC CORRELAZIONI VAL"
                )
                X_va_categorical = X_va.select_dtypes(exclude=[np.number])
                X_va_final = pd.concat([X_va_num_clean, X_va_categorical], axis=1)
            else:
                X_va_final = None
        else:
            # Nessuna correlazione da rimuovere
            X_tr_final = X_tr
            X_te_final = X_te
            X_va_final = X_va
        prefix = profiles_cfg.get("tree", {}).get("output_prefix", "tree")
        save_profile(X_tr_final, X_te_final, y_train, y_test, X_va_final, y_val, prefix)
        if first_profile_saved is None:
            first_profile_saved = prefix

    # Profile: catboost (preserve categoricals)
    if profiles_cfg.get("catboost", {}).get("enabled", False):
        logger.info("[catboost] Preservazione categoriche; niente OHE")
        X_tr = base_train.copy(); X_te = base_test.copy(); X_va = base_val.copy() if base_val is not None else None
        # Coerce numeric-like strings to numeric
        X_tr, [X_te, X_va] = coerce_numeric_like(X_tr, [X_te, X_va])
        # Rimozione colonne non descrittive con operazioni robuste
        non_descriptive_cols, nd_stats = RobustColumnAnalyzer.analyze_missing_values(
            X_tr, threshold=na_thr
        )
        X_tr, nd_removal_info = RobustDataOperations.remove_columns_safe(
            X_tr, non_descriptive_cols, f"[catboost] RIMOZIONE COLONNE NON DESCRITTIVE"
        )
        
        # Sincronizza rimozione preservando categoriche per CatBoost
        if non_descriptive_cols:
            X_te, _ = RobustDataOperations.remove_columns_safe(
                X_te, non_descriptive_cols, "[catboost] SYNC RIMOZIONE TEST"
            )
            if X_va is not None:
                X_va, _ = RobustDataOperations.remove_columns_safe(
                    X_va, non_descriptive_cols, "[catboost] SYNC RIMOZIONE VAL"
                )
        
        # Fill any remaining NaN values to ensure compatibility with all sklearn models
        for _df in (X_tr, X_te, X_va):
            if _df is not None:
                # Fill numeric NaN values with 0
                numeric_cols = _df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    _df[numeric_cols] = _df[numeric_cols].fillna(0)
                # Fill categorical NaN values with "UNKNOWN"
                cat_cols = _df.select_dtypes(include=["object", "category"]).columns
                if len(cat_cols) > 0:
                    _df[cat_cols] = _df[cat_cols].fillna("UNKNOWN")
        
        # Pruning correlazioni numeriche preservando categoriche per CatBoost
        corr_thr = float(profiles_cfg.get("catboost", {}).get("correlation", {}).get("numeric_threshold", 
                                                             config.get("correlation", {}).get("numeric_threshold", 0.98)))
        
        # Analizza correlazioni solo su colonne numeriche (preserva categoriche)
        X_tr_numeric = X_tr.select_dtypes(include=[np.number])
        corr_cols_to_remove, corr_stats = RobustColumnAnalyzer.find_highly_correlated_columns(
            X_tr_numeric, threshold=corr_thr
        )
        
        # Rimuovi correlazioni mantenendo tutte le categoriche per CatBoost
        if corr_cols_to_remove:
            # Rimuovi solo dalle numeriche
            X_tr_num_clean, _ = RobustDataOperations.remove_columns_safe(
                X_tr_numeric, corr_cols_to_remove, f"[catboost] PRUNING CORRELAZIONI NUMERICHE (soglia={corr_thr})"
            )
            
            # Ricostruisci dataset completo mantenendo TUTTE le categoriche
            X_tr_categorical = X_tr.select_dtypes(exclude=[np.number])
            X_tr_final = pd.concat([X_tr_num_clean, X_tr_categorical], axis=1)
            
            # Sincronizza su test e validation
            X_te_num_clean, _ = RobustDataOperations.remove_columns_safe(
                X_te.select_dtypes(include=[np.number]), corr_cols_to_remove, "[catboost] SYNC CORRELAZIONI TEST"
            )
            X_te_categorical = X_te.select_dtypes(exclude=[np.number])
            X_te_final = pd.concat([X_te_num_clean, X_te_categorical], axis=1)
            
            if X_va is not None:
                X_va_num_clean, _ = RobustDataOperations.remove_columns_safe(
                    X_va.select_dtypes(include=[np.number]), corr_cols_to_remove, "[catboost] SYNC CORRELAZIONI VAL"
                )
                X_va_categorical = X_va.select_dtypes(exclude=[np.number])
                X_va_final = pd.concat([X_va_num_clean, X_va_categorical], axis=1)
            else:
                X_va_final = None
        else:
            # Nessuna correlazione da rimuovere
            X_tr_final = X_tr
            X_te_final = X_te  
            X_va_final = X_va
        # Audit: ensure no NaN in numeric parts post-prune
        for name, _df in [("X_tr_final", X_tr_final), ("X_te_final", X_te_final), ("X_va_final", X_va_final)]:
            if _df is not None and _df.select_dtypes(include=[np.number]).isnull().any().any():
                logger.warning(f"[catboost] Residual NaN detected in numeric columns of {name}, filling with 0")
                num_cols = _df.select_dtypes(include=[np.number]).columns
                _df[num_cols] = _df[num_cols].fillna(0)
        prefix = profiles_cfg.get("catboost", {}).get("output_prefix", "catboost")
        save_profile(X_tr_final, X_te_final, y_train, y_test, X_va_final, y_val, prefix)
        # Save list of categorical columns for catboost
        cat_cols = X_tr_final.select_dtypes(include=["object", "category"]).columns.tolist()
        (pre_dir / f"categorical_columns_{prefix}.txt").write_text("\n".join(cat_cols), encoding="utf-8")
        logger.info(f"[catboost] Salvate {len(cat_cols)} colonne categoriche")
        if first_profile_saved is None:
            first_profile_saved = prefix

    # Backward-compatible symlinks (copy) to default filenames using first enabled profile
    if first_profile_saved is not None:
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
        logger.info(f"Back-compat: copiati file del profilo '{first_profile_saved}' nei nomi default e combinati in {out_path}")

    # Quality Checks finali se abilitati
    quality_checks_enabled = config.get("quality_checks", {}).get("check_temporal_leakage", False)
    if quality_checks_enabled:
        logger.info("Esecuzione quality checks finali...")
        tracker.track_step_start('quality_checks')
        
        try:
            # Inizializza quality checker
            quality_checker = QualityChecker(config)
            
            # Carica dati processati per quality checks
            if first_profile_saved:
                X_train_qc = pd.read_parquet(pre_dir / f"X_train_{first_profile_saved}.parquet")
                X_test_qc = pd.read_parquet(pre_dir / f"X_test_{first_profile_saved}.parquet")
                y_train_qc = pd.read_parquet(pre_dir / f"y_train_{first_profile_saved}.parquet")[target_col]
                y_test_qc = pd.read_parquet(pre_dir / f"y_test_{first_profile_saved}.parquet")[target_col]
                
                X_val_qc = None
                y_val_qc = None
                if (pre_dir / f"X_val_{first_profile_saved}.parquet").exists():
                    X_val_qc = pd.read_parquet(pre_dir / f"X_val_{first_profile_saved}.parquet")
                    y_val_qc = pd.read_parquet(pre_dir / f"y_val_{first_profile_saved}.parquet")[target_col]
                
                # Esegui tutti i quality checks
                quality_results = quality_checker.run_all_checks(
                    X_train_qc, X_val_qc, X_test_qc, y_train_qc, y_val_qc, y_test_qc,
                    preprocessing_info=tracker.generate_comprehensive_report()
                )
                
                # Salva risultati quality checks
                quality_report_path = reports_dir / "quality_checks_report.json"
                import json
                with open(quality_report_path, 'w', encoding='utf-8') as f:
                    json.dump(quality_results, f, indent=2, default=str)
                
                logger.info(f"Quality checks completati - Status: {quality_results['overall_status']}")
                
                # Log warnings/errori critici
                if quality_results['critical_errors']:
                    logger.error(f"Quality checks - Errori critici: {quality_results['critical_errors']}")
                if quality_results['warnings']:
                    logger.warning(f"Quality checks - Warnings: {len(quality_results['warnings'])} rilevati")
                
                tracker.track_step_completion('quality_checks', df, df, quality_results)
            else:
                logger.warning("Quality checks saltati - nessun profilo processato")
                
        except Exception as e:
            logger.error(f"Errore durante quality checks: {e}")
    
    # Genera report finale completo con tracking
    final_report = tracker.generate_comprehensive_report()
    
    # Salva report tradizionale
    save_report(
        reports_dir / "preprocessing.md",
        sections={
            "Raw profile": dataframe_profile(df_initial),
            "After feature extraction": dataframe_profile(base_train) if 'base_train' in locals() else {},
            "Train features": dataframe_profile(base_train) if 'base_train' in locals() else {},
            "Test features": dataframe_profile(base_test) if 'base_test' in locals() else {},
            "Pipeline Summary": {
                "Total Steps": final_report['pipeline_summary']['steps_completed'],
                "Total Duration": final_report['pipeline_summary']['total_duration_formatted'],
                "Final Status": "Completed Successfully"
            }
        },
    )
    
    # Log summary finale
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETATO CON SUCCESSO")
    logger.info("=" * 60)
    logger.info(f"📊 Steps completati: {final_report['pipeline_summary']['steps_completed']}")
    logger.info(f"⏱️  Durata totale: {final_report['pipeline_summary']['total_duration_formatted']}")
    logger.info(f"📁 Output directory: {pre_dir}")
    logger.info(f"📋 Report tracking: {tracker.reports_dir}")
    
    if quality_checks_enabled and 'quality_results' in locals():
        logger.info(f"🔍 Quality checks: {quality_results['overall_status']}")
    
    logger.info("=" * 60)

    return out_path