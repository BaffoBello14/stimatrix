from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json

import numpy as np
import pandas as pd
from joblib import dump

from utils.logger import get_logger
from preprocessing.target_transforms import inverse_target_transform
from .tuner import tune_model
from .model_zoo import build_estimator
from .metrics import regression_metrics, overfit_diagnostics, grouped_regression_metrics, _build_price_bands
from .ensembles import build_voting, build_stacking
from .shap_utils import compute_shap, save_shap_plots
from utils.wandb_utils import WandbTracker

logger = get_logger(__name__)


def _inverse_transform_predictions(
    y_pred: np.ndarray,
    transform_metadata: Dict[str, Any],
    smearing_factor: Optional[float] = None
) -> np.ndarray:
    """
    Apply inverse transformation to predictions.
    
    For log transform: applies Duan smearing if smearing_factor provided.
    For other transforms: applies inverse directly (no smearing).
    """
    # Apply inverse transformation
    y_pred_orig = inverse_target_transform(y_pred, transform_metadata)
    
    # Apply Duan smearing for log transform if provided
    if transform_metadata.get("transform") == "log" and smearing_factor is not None:
        y_pred_orig = y_pred_orig * smearing_factor
    
    return y_pred_orig


def _load_xy(pre_dir: Path, prefix: Optional[str]) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series], pd.DataFrame, pd.Series]:
    def name(base: str) -> Path:
        return pre_dir / (f"{base}_{prefix}.parquet" if prefix else f"{base}.parquet")

    X_train = pd.read_parquet(name("X_train"))
    y_train_df = pd.read_parquet(name("y_train"))
    target_col = y_train_df.columns[0]
    y_train = y_train_df[target_col]

    X_test = pd.read_parquet(name("X_test"))
    y_test = pd.read_parquet(name("y_test"))[target_col]

    X_val_path = name("X_val")
    y_val_path = name("y_val")
    X_val = pd.read_parquet(X_val_path) if X_val_path.exists() else None
    y_val = pd.read_parquet(y_val_path)[target_col] if y_val_path.exists() else None
    return X_train, y_train, X_val, y_val, X_test, y_test


def _profile_for(model_key: str, cfg: Dict[str, Any]) -> Optional[str]:
    m = cfg.get("training", {}).get("profile_map", {})
    pf = m.get(model_key, None)
    return pf


def _catboost_cat_features(pre_dir: Path, prefix: str, X: pd.DataFrame) -> List[int]:
    lst = (pre_dir / f"categorical_columns_{prefix}.txt")
    if lst.exists():
        cols = lst.read_text(encoding="utf-8").splitlines()
        indices = [i for i, c in enumerate(X.columns) if c in cols]
        return indices
    # fallback: infer
    return [i for i, dt in enumerate(X.dtypes) if str(dt) in ("object", "category")]


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    paths = config.get("paths", {})
    pre_dir = Path(paths.get("preprocessed_data", "data/preprocessed"))
    models_dir = Path(paths.get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Paths: preprocessed={pre_dir} | models_dir={models_dir}")

    # Initialize W&B tracking (no-op if disabled or not installed)
    wb = WandbTracker(config)
    wb.start_run(job_type="training")

    tr_cfg = config.get("training", {})
    primary_metric: str = tr_cfg.get("primary_metric", "r2")
    report_metrics: List[str] = tr_cfg.get("report_metrics", ["r2", "rmse", "mse", "mae", "mape"])
    sampler_name = tr_cfg.get("sampler", "auto")
    seed = int(tr_cfg.get("seed", 42))
    
    # Get tuning split fraction from temporal_split config (supports new nested schema)
    temporal_cfg = config.get("temporal_split", {})
    frac_cfg = temporal_cfg.get("fraction", {})
    tuning_split_fraction = float(frac_cfg.get("train", temporal_cfg.get("train_fraction", 0.8)))

    shap_cfg = tr_cfg.get("shap", {"enabled": True})

    # Group-metrics configuration
    eval_cfg: Dict[str, Any] = config.get("evaluation", {}) or {}
    gm_cfg: Dict[str, Any] = eval_cfg.get("group_metrics", {}) or {}
    gm_enabled: bool = bool(gm_cfg.get("enabled", True))
    gm_report_metrics: List[str] = list(gm_cfg.get("report_metrics", report_metrics))
    gm_min_group_size: int = int(gm_cfg.get("min_group_size", 30))
    gm_original_scale: bool = bool(gm_cfg.get("original_scale", True))
    gm_log_wandb: bool = bool(gm_cfg.get("log_wandb", True))
    price_cfg: Dict[str, Any] = gm_cfg.get("price_band", {}) or {}

    # Read preprocessing info for target transformation metadata
    prep_info_path = pre_dir / "preprocessing_info.json"
    transform_metadata: Dict[str, Any] = {"transform": "none"}
    if prep_info_path.exists():
        try:
            prep_info = json.loads(prep_info_path.read_text(encoding="utf-8"))
            transform_metadata = prep_info.get("target_transformation", {"transform": "none"})
            # Backward compatibility with old log_transformation format
            if transform_metadata.get("transform") == "none":
                old_log_flag = ((prep_info.get("log_transformation", {}) or {}).get("applied", False))
                if old_log_flag:
                    transform_metadata = {"transform": "log"}
                    logger.warning("⚠️  Using legacy log_transformation format from preprocessing_info.json")
        except Exception as e:
            logger.warning(f"Could not load preprocessing_info.json: {e}")
            transform_metadata = {"transform": "none"}
    
    # Helper flag for backward compatibility
    log_applied_global = transform_metadata.get("transform") != "none"

    # Raccogli definizioni per-modello
    models_cfg: Dict[str, Any] = tr_cfg.get("models", {})
    selected_models: List[str] = [k for k, v in models_cfg.items() if bool(v.get("enabled", False))]
    logger.info(f"Modelli selezionati: {selected_models}")
    logger.info(f"SHAP: enabled={bool(shap_cfg.get('enabled', True))} sample_size={int(shap_cfg.get('sample_size', 2000))}")
    # Log basic run configuration
    wb.log({
        "config/primary_metric": primary_metric,
        "config/seed": seed,
        "config/sampler": sampler_name,
        "config/models_count": len(selected_models),
    })

    results: Dict[str, Any] = {"models": {}, "ensembles": {}, "baselines": {}}
    table_rows: List[Dict[str, Any]] = []

    # Per-model loop
    for model_key in selected_models:
        model_entry = models_cfg.get(model_key, {})
        prefix = model_entry.get("profile", None)
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = _load_xy(pre_dir, prefix)
        except Exception as e:
            logger.error(f"Caricamento dataset fallito per modello {model_key} (prefix={prefix}): {e}")
            continue

        # Determina se richiede solo numeriche
        default_numeric_only = model_key.lower() not in ["catboost"]
        requires_numeric_only = bool(model_entry.get("requires_numeric_only", default_numeric_only))

        cat_features: Optional[List[int]] = None
        if model_key.lower() == "catboost":
            if prefix is None:
                logger.warning("CatBoost richiede il profilo 'catboost'. Provo inferenza colonne categoriche.")
                cat_features = [i for i, dt in enumerate(X_train.dtypes) if str(dt) in ("object", "category")]
            else:
                cat_features = _catboost_cat_features(pre_dir, prefix, X_train)

        # Per modelli che richiedono solo numeriche, rimuovi eventuali colonne non numeriche rimaste
        if requires_numeric_only:
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) != X_train.shape[1]:
                X_train = X_train[numeric_cols]
                X_test = X_test.reindex(columns=numeric_cols, fill_value=0)
                if X_val is not None:
                    X_val = X_val.reindex(columns=numeric_cols, fill_value=0)

        # Safety check NaN
        if model_key.lower() in ['gbr', 'hgbt', 'svr', 'linear', 'ridge', 'lasso', 'elasticnet', 'knn', 'dt', 'rf', 'lightgbm', 'xgboost']:
            if X_train.isnull().any().any():
                logger.warning(f"Found NaN values in training data for {model_key}, filling with 0")
                X_train = X_train.fillna(0)
            if X_test.isnull().any().any():
                logger.warning(f"Found NaN values in test data for {model_key}, filling with 0")
                X_test = X_test.fillna(0)
            if X_val is not None and X_val.isnull().any().any():
                logger.warning(f"Found NaN values in validation data for {model_key}, filling with 0")
                X_val = X_val.fillna(0)

        # Prepare base params (also used as baseline)
        space = model_entry.get("search_space", {})
        base = {}
        base.update(model_entry.get("base_params", {}) or {})
        n_jobs_default = int(tr_cfg.get("n_jobs_default", -1))
        mk_lower = model_key.lower()
        if mk_lower in {"rf", "knn", "xgboost", "lightgbm"} and "n_jobs" not in base:
            base["n_jobs"] = n_jobs_default
        if mk_lower == "catboost" and "thread_count" not in base:
            base["thread_count"] = n_jobs_default
        # seeds
        if mk_lower in {"dt", "rf", "gbr", "hgbt"} and "random_state" not in base:
            base["random_state"] = seed
        if mk_lower in {"xgboost", "lightgbm"} and "random_state" not in base:
            base["random_state"] = seed
        if mk_lower == "catboost" and "random_seed" not in base:
            base["random_seed"] = seed
        # Improve convergence for coordinate-descent models
        if mk_lower in {"lasso", "elasticnet"} and "max_iter" not in base:
            base["max_iter"] = 10000
        n_trials = int(model_entry.get("trials", 50))
        timeout = tr_cfg.get("timeout", None)
        use_values_for_tuning = requires_numeric_only or mk_lower == "lightgbm"

        # BASELINE evaluation (no tuning)
        try:
            baseline_est = build_estimator(model_key, base)
            if mk_lower == "catboost":
                # Ensure CatBoost receives categorical feature indices to avoid coercion errors
                if cat_features is None:
                    cat_features = _catboost_cat_features(pre_dir, prefix or "catboost", X_train)
                if X_val is None or y_val is None:
                    baseline_est.fit(X_train, y_train, cat_features=cat_features)
                else:
                    baseline_est.fit(X_train, y_train, cat_features=cat_features)
                y_pred_test_base = baseline_est.predict(X_test)
            elif requires_numeric_only or mk_lower == "lightgbm":
                baseline_est.fit(X_train.values if X_val is None else pd.concat([X_train, X_val]).values,
                                 y_train.values if y_val is None else pd.concat([y_train, y_val]).values)
                y_pred_test_base = baseline_est.predict(X_test.values)
            else:
                baseline_est.fit(X_train if X_val is None else pd.concat([X_train, X_val]),
                                 y_train if y_val is None else pd.concat([y_train, y_val]))
                y_pred_test_base = baseline_est.predict(X_test)
            m_test_base = regression_metrics(y_test.values, y_pred_test_base)
            results["baselines"][model_key] = {
                "metrics_test": m_test_base,
            }
            wb.log_prefixed_metrics(f"baseline/{model_key}", {f"test_{k}": v for k, v in m_test_base.items()})
            table_rows.append({
                "Model": f"Baseline_{model_key}",
                "Category": "Baseline",
                "Test_RMSE": m_test_base.get("rmse"),
                "Test_R2": m_test_base.get("r2"),
            })
            logger.info(f"[baseline:{model_key}] test r2={m_test_base['r2']:.4f} rmse={m_test_base['rmse']:.4f}")
        except Exception as e:
            logger.warning(f"Baseline fallita per {model_key}: {e}")

        try:
            tuning = tune_model(
                model_key=model_key,
                X_train=X_train.values if use_values_for_tuning else X_train,
                y_train=y_train.values,
                X_val=None if X_val is None else (X_val.values if use_values_for_tuning else X_val),
                y_val=None if y_val is None else y_val.values,
                primary_metric=primary_metric,
                n_trials=n_trials,
                timeout=timeout,
                sampler_name=sampler_name,
                seed=seed,
                base_params=base,
                search_space=space,
                cat_features=cat_features,
                cv_config=(tr_cfg.get("cv_when_no_val", {}) if X_val is None else None),
                tuning_split_fraction=tuning_split_fraction,
            )
        except ImportError as e:
            logger.warning(f"Dipendenza mancante per modello {model_key}: {e}. Skip del modello.")
            continue

        best_params_merged = {**base, **tuning.best_params}

        if X_val is not None and y_val is not None:
            X_tr_final = pd.concat([X_train, X_val], axis=0)
            y_tr_final = pd.concat([y_train, y_val], axis=0)
        else:
            X_tr_final, y_tr_final = X_train, y_train

        try:
            estimator = build_estimator(model_key, best_params_merged)
        except ImportError as e:
            logger.warning(f"Dipendenza mancante per modello {model_key}: {e}. Skip del modello.")
            continue

        fit_params = model_entry.get("fit_params", {}) or {}
        if "__categorical_indices__" in str(fit_params):
            if cat_features is None:
                cat_features = _catboost_cat_features(pre_dir, prefix or "catboost", X_tr_final)
            fit_params = json.loads(json.dumps(fit_params).replace("__categorical_indices__", json.dumps(cat_features)))
        if isinstance(fit_params, dict):
            for k, v in list(fit_params.items()):
                if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
                    try:
                        fit_params[k] = json.loads(v)
                    except Exception:
                        pass

        use_values_for_final = requires_numeric_only or mk_lower == "lightgbm"
        if use_values_for_final:
            estimator.fit(X_tr_final.values, y_tr_final.values, **fit_params)
        else:
            estimator.fit(X_tr_final, y_tr_final, **fit_params)

        y_pred_test = estimator.predict(X_test.values if use_values_for_final else X_test)
        y_pred_train = estimator.predict(X_tr_final.values if use_values_for_final else X_tr_final)

        m_test = regression_metrics(y_test.values, y_pred_test)
        m_train = regression_metrics(y_tr_final.values, y_pred_train)
        # Also compute metrics on original scale (euros) if log-transform was applied
        m_test_orig = None
        m_train_orig = None
        smearing_factor: Optional[float] = None
        try:
            if log_applied_global:
                # Test original scale
                try:
                    y_test_orig_path = pre_dir / (f"y_test_orig_{prefix}.parquet" if prefix else "y_test_orig.parquet")
                    if y_test_orig_path.exists():
                        y_test_true_orig = pd.read_parquet(y_test_orig_path).iloc[:, 0].values
                    else:
                        y_test_true_orig = np.expm1(y_test.values)
                except Exception:
                    y_test_true_orig = np.expm1(y_test.values)
                # Duan's smearing factor from training residuals in log-space
                try:
                    residuals_log = (y_tr_final.values - np.asarray(y_pred_train))
                    smearing_factor = float(np.mean(np.exp(residuals_log)))
                except Exception:
                    smearing_factor = 1.0
                y_pred_test_orig = np.expm1(np.asarray(y_pred_test)) * (smearing_factor if smearing_factor is not None else 1.0)
                m_test_orig = regression_metrics(y_test_true_orig, y_pred_test_orig)

                # Train original scale
                y_train_true_orig = np.expm1(y_tr_final.values)
                y_pred_train_orig = np.expm1(np.asarray(y_pred_train)) * (smearing_factor if smearing_factor is not None else 1.0)
                m_train_orig = regression_metrics(y_train_true_orig, y_pred_train_orig)
            else:
                # If no log transform, original-scale == current metrics
                m_test_orig = dict(m_test)
                m_train_orig = dict(m_train)
        except Exception as _e:
            # Fail-safe: keep None if any issue arises
            m_test_orig = m_test_orig or None
            m_train_orig = m_train_orig or None
        diag = overfit_diagnostics(m_train, m_test)
        wb.log_prefixed_metrics(f"model/{model_key}", {**{f"train_{k}": v for k, v in m_train.items()}, **{f"test_{k}": v for k, v in m_test.items()}})
        wb.log_prefixed_metrics(f"model/{model_key}", {f"overfit_{k}": v for k, v in diag.items()})

        model_id = f"{model_key}"
        model_dir = models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        dump(estimator, model_dir / "model.pkl")
        # Add MAPE with floor on original scale if available
        try:
            price_cfg: Dict[str, Any] = gm_cfg.get("price_band", {}) if 'gm_cfg' in locals() else {}
            mape_floor = float(price_cfg.get("mape_floor", 1e-8))
            if m_test_orig is not None:
                denom = np.where(np.abs(y_test_true_orig) < max(mape_floor, 1e-8), max(mape_floor, 1e-8), np.abs(y_test_true_orig))
                mape_orig_floor = float(np.mean(np.abs((y_test_true_orig - y_pred_test_orig) / denom)))
                m_test_orig["mape_floor"] = mape_orig_floor
            if m_train_orig is not None:
                denom_tr = np.where(np.abs(y_train_true_orig) < max(mape_floor, 1e-8), max(mape_floor, 1e-8), np.abs(y_train_true_orig))
                mape_orig_floor_tr = float(np.mean(np.abs((y_train_true_orig - y_pred_train_orig) / denom_tr)))
                m_train_orig["mape_floor"] = mape_orig_floor_tr
        except Exception:
            pass

        meta = {
            "model_key": model_key,
            "prefix": prefix,
            "primary_metric": primary_metric,
            "best_primary_value": tuning.best_value,
            "best_params": tuning.best_params,
            "metrics_train": m_train,
            "metrics_test": m_test,
            "metrics_train_original": m_train_orig,
            "metrics_test_original": m_test_orig,
            "smearing_factor": smearing_factor if smearing_factor is not None else (1.0 if not log_applied_global else None),
            "overfit": diag,
        }
        (model_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Per-group metrics on test set (option 1)
        if gm_enabled:
            try:
                # Determine ground truth and predictions for grouping (original scale optional)
                y_true_series: pd.Series
                y_pred_series: pd.Series
                if gm_original_scale:
                    # Load original-scale y_test if available
                    y_test_orig_path = pre_dir / (f"y_test_orig_{prefix}.parquet" if prefix else "y_test_orig.parquet")
                    if y_test_orig_path.exists():
                        y_true_series = pd.read_parquet(y_test_orig_path).iloc[:, 0]
                    else:
                        # Fallback: invert from log if applied, else use y_test
                        if log_applied_global:
                            y_true_series = pd.Series(np.expm1(y_test.values))
                        else:
                            y_true_series = y_test
                    # Predictions to original scale
                    if log_applied_global:
                        y_pred_series = pd.Series(np.expm1(y_pred_test))
                    else:
                        y_pred_series = pd.Series(y_pred_test)
                else:
                    y_true_series = y_test
                    y_pred_series = pd.Series(y_pred_test)

                # Load configured group-by columns for TEST split
                group_cols_path = pre_dir / (f"group_cols_test_{prefix}.parquet" if prefix else "group_cols_test.parquet")
                if group_cols_path.exists():
                    grp_df = pd.read_parquet(group_cols_path)
                    gb_cols_cfg: List[str] = [c for c in (gm_cfg.get("group_by_columns", []) or []) if c in grp_df.columns]
                else:
                    grp_df = pd.DataFrame()
                    gb_cols_cfg = []

                # Compute and save grouped metrics for each configured column
                for gb_col in gb_cols_cfg:
                    groups = grp_df[gb_col].fillna("MISSING")
                    gm_df = grouped_regression_metrics(
                        y_true=y_true_series,
                        y_pred=y_pred_series,
                        groups=groups,
                        report_metrics=gm_report_metrics,
                        min_group_size=gm_min_group_size,
                        mape_floor=float(price_cfg.get("mape_floor", 1e-8)),
                    )
                    if not gm_df.empty:
                        out_csv = model_dir / f"group_metrics_{gb_col}.csv"
                        gm_df.to_csv(out_csv, index=False)
                        if gm_log_wandb:
                            try:
                                wb.log_artifact(out_csv, name=f"{model_key}_{gb_col}_group_metrics.csv", type="metrics")
                            except Exception:
                                pass

                # Price-band grouped metrics
                if price_cfg:
                    bands = _build_price_bands(
                        y_true_orig=y_true_series,
                        method=str(price_cfg.get("method", "quantile")),
                        quantiles=price_cfg.get("quantiles"),
                        fixed_edges=price_cfg.get("fixed_edges"),
                        label_prefix=str(price_cfg.get("label_prefix", "PREZZO_")),
                    )
                    gm_price = grouped_regression_metrics(
                        y_true=y_true_series,
                        y_pred=y_pred_series,
                        groups=bands,
                        report_metrics=gm_report_metrics,
                        min_group_size=gm_min_group_size,
                        mape_floor=float(price_cfg.get("mape_floor", 1e-8)),
                    )
                    if not gm_price.empty:
                        out_csv = model_dir / "group_metrics_price_band.csv"
                        gm_price.to_csv(out_csv, index=False)
                        if gm_log_wandb:
                            try:
                                wb.log_artifact(out_csv, name=f"{model_key}_price_band_metrics.csv", type="metrics")
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"Per-group metrics failed for {model_key}: {e}")

        # Log Optuna trials table and model directory as artifact
        try:
            df_trials = tuning.study.trials_dataframe()
            df_trials.to_csv(model_dir / "optuna_trials.csv", index=False)
            try:
                wb.log({f"trials/{model_key}": df_trials})
            except Exception:
                pass
        except Exception:
            pass

        if bool(shap_cfg.get("enabled", True)):
            try:
                shap_bundle = compute_shap(
                    model=estimator,
                    X=X_tr_final.values if use_values_for_final else X_tr_final,
                    sample_size=int(shap_cfg.get("sample_size", 2000)),
                    max_display=int(shap_cfg.get("max_display", 30)),
                    keep_as_numpy=use_values_for_final,
                    random_state=seed,
                    feature_names=list(X_tr_final.columns) if use_values_for_final else None,
                )
                if bool(shap_cfg.get("save_plots", True)):
                    save_shap_plots(str(model_dir / "shap"), shap_bundle, model_id)
                    try:
                        wb.log_image(key=f"shap/{model_key}_beeswarm", image_path=model_dir / "shap" / f"shap_{model_id}_beeswarm.png")
                        wb.log_image(key=f"shap/{model_key}_bar", image_path=model_dir / "shap" / f"shap_{model_id}_bar.png")
                    except Exception:
                        pass
                if bool(shap_cfg.get("save_values", False)):
                    try:
                        shap_values_obj = shap_bundle.get("values")
                        values_array = shap_values_obj.values if hasattr(shap_values_obj, "values") else np.asarray(shap_values_obj)
                        np.save(model_dir / "shap_values.npy", values_array, allow_pickle=False)
                    except Exception:
                        # As a last resort, skip saving raw values
                        pass
                    try:
                        # Save the sampled data used for SHAP
                        shap_bundle["data_sample"].to_parquet(model_dir / "shap_sample.parquet", index=False)
                    except Exception:
                        pass
                try:
                    wb.log_prefixed_metrics(f"shap/{model_key}", {"sample_size": int(shap_bundle.get("sample_size", 0))})
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"SHAP fallito per {model_key}: {e}")

        results["models"][model_id] = {
            "best_params": tuning.best_params,
            "best_primary_value": tuning.best_value,
            "metrics_test": m_test,
            "metrics_train": m_train,
            "metrics_test_original": m_test_orig,
            "metrics_train_original": m_train_orig,
            "smearing_factor": meta.get("smearing_factor"),
            "overfit": diag,
        }
        table_rows.append({
            "Model": f"Optimized_{model_key}",
            "Category": "Optimized",
            "Test_RMSE": m_test.get("rmse"),
            "Test_R2": m_test.get("r2"),
        })
        logger.info(f"[{model_key}] best {primary_metric}={tuning.best_value:.6f} | test r2={m_test['r2']:.4f} rmse={m_test['rmse']:.4f}")
        wb.log({f"model/{model_key}/best_primary_value": tuning.best_value})

    # ENSEMBLES
    ens_cfg = tr_cfg.get("ensembles", {})
    ranked = sorted(
        [
            (k, v["best_params"], v["best_primary_value"]) for k, v in results.get("models", {}).items()
        ],
        key=lambda x: x[2],
        reverse=True,
    )

    if ens_cfg.get("voting", {}).get("enabled", False) and len(ranked) >= 2:
        top_n = int(ens_cfg.get("voting", {}).get("top_n", 3))
        selected = [(k, p) for (k, p, _) in ranked[:top_n]]
        vote = build_voting(selected, tune_weights=bool(ens_cfg.get("voting", {}).get("tune_weights", True)), n_jobs=int(tr_cfg.get("n_jobs_default", -1)))
        first_key = selected[0][0]
        prefix = _profile_for(first_key, config)
        X_train, y_train, X_val, y_val, X_test, y_test = _load_xy(pre_dir, prefix)
        X_tr_final = pd.concat([X_train, X_val], axis=0) if X_val is not None else X_train
        y_tr_final = pd.concat([y_train, y_val], axis=0) if y_val is not None else y_train
        vote.fit(X_tr_final.values, y_tr_final.values)
        y_pred_test = vote.predict(X_test.values)
        y_pred_train = vote.predict(X_tr_final.values)
        m_test = regression_metrics(y_test.values, y_pred_test)
        m_train = regression_metrics(y_tr_final.values, y_pred_train)
        # Original-scale metrics with Duan smearing (if log applied)
        m_test_orig = None
        m_train_orig = None
        try:
            if log_applied_global:
                y_test_true_orig = (pd.read_parquet(pre_dir / (f"y_test_orig_{prefix}.parquet" if prefix else "y_test_orig.parquet")).iloc[:, 0].values
                                    if (pre_dir / (f"y_test_orig_{prefix}.parquet" if prefix else "y_test_orig.parquet")).exists()
                                    else np.expm1(y_test.values))
                residuals_log = y_tr_final.values - y_pred_train
                smear_v = float(np.mean(np.exp(residuals_log)))
                y_pred_test_orig = np.expm1(y_pred_test) * smear_v
                y_train_true_orig = np.expm1(y_tr_final.values)
                y_pred_train_orig = np.expm1(y_pred_train) * smear_v
                m_test_orig = regression_metrics(y_test_true_orig, y_pred_test_orig)
                m_train_orig = regression_metrics(y_train_true_orig, y_pred_train_orig)
                # mape floor
                price_cfg: Dict[str, Any] = gm_cfg.get("price_band", {}) if 'gm_cfg' in locals() else {}
                mape_floor = float(price_cfg.get("mape_floor", 1e-8))
                denom = np.where(np.abs(y_test_true_orig) < max(mape_floor, 1e-8), max(mape_floor, 1e-8), np.abs(y_test_true_orig))
                m_test_orig["mape_floor"] = float(np.mean(np.abs((y_test_true_orig - y_pred_test_orig) / denom)))
            else:
                m_test_orig = dict(m_test)
                m_train_orig = dict(m_train)
        except Exception:
            pass
        diag = overfit_diagnostics(m_train, m_test)
        ens_id = "voting"
        ens_dir = models_dir / ens_id
        ens_dir.mkdir(parents=True, exist_ok=True)
        dump(vote, ens_dir / "model.pkl")
        meta = {
            "type": "voting",
            "members": [k for k, _ in selected],
            "metrics_train": m_train,
            "metrics_test": m_test,
            "metrics_train_original": m_train_orig,
            "metrics_test_original": m_test_orig,
            "overfit": diag,
        }
        (ens_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        results["ensembles"][ens_id] = meta
        table_rows.append({
            "Model": f"Ensemble_{ens_id}",
            "Category": "Ensemble",
            "Test_RMSE": m_test.get("rmse"),
            "Test_R2": m_test.get("r2"),
        })
        logger.info(f"[voting] test r2={m_test['r2']:.4f} rmse={m_test['rmse']:.4f}")
        wb.log_prefixed_metrics("ensemble/voting", {**{f"train_{k}": v for k, v in m_train.items()}, **{f"test_{k}": v for k, v in m_test.items()}})

        # Group metrics for ensemble voting
        if gm_enabled:
            try:
                y_true_series = pd.Series(y_test_true_orig) if (log_applied_global and 'y_test_true_orig' in locals()) else y_test
                y_pred_series = pd.Series(y_pred_test_orig) if (log_applied_global and 'y_pred_test_orig' in locals()) else pd.Series(y_pred_test)
                group_cols_path = pre_dir / (f"group_cols_test_{prefix}.parquet" if prefix else "group_cols_test.parquet")
                if group_cols_path.exists():
                    grp_df = pd.read_parquet(group_cols_path)
                    gb_cols_cfg: List[str] = [c for c in (gm_cfg.get("group_by_columns", []) or []) if c in grp_df.columns]
                else:
                    grp_df = pd.DataFrame()
                    gb_cols_cfg = []
                for gb_col in gb_cols_cfg:
                    groups = grp_df[gb_col].fillna("MISSING")
                    gm_df = grouped_regression_metrics(
                        y_true=y_true_series,
                        y_pred=y_pred_series,
                        groups=groups,
                        report_metrics=gm_report_metrics,
                        min_group_size=gm_min_group_size,
                        mape_floor=float(price_cfg.get("mape_floor", 1e-8)),
                    )
                    if not gm_df.empty:
                        out_csv = ens_dir / f"group_metrics_{gb_col}.csv"
                        gm_df.to_csv(out_csv, index=False)
                        if gm_log_wandb:
                            try:
                                wb.log_artifact(out_csv, name=f"{ens_id}_{gb_col}_group_metrics.csv", type="metrics")
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"Per-group metrics failed for ensemble {ens_id}: {e}")

    if ens_cfg.get("stacking", {}).get("enabled", False) and len(ranked) >= 2:
        top_n = int(ens_cfg.get("stacking", {}).get("top_n", 5))
        final_est_key = str(ens_cfg.get("stacking", {}).get("final_estimator", "ridge"))
        cv_folds = int(ens_cfg.get("stacking", {}).get("cv_folds", 5))
        selected = [(k, p) for (k, p, _) in ranked[:top_n]]
        stack = build_stacking(selected, final_estimator_key=final_est_key, cv_folds=cv_folds, n_jobs=int(tr_cfg.get("n_jobs_default", -1)))
        first_key = selected[0][0]
        prefix = _profile_for(first_key, config)
        X_train, y_train, X_val, y_val, X_test, y_test = _load_xy(pre_dir, prefix)
        X_tr_final = pd.concat([X_train, X_val], axis=0) if X_val is not None else X_train
        y_tr_final = pd.concat([y_train, y_val], axis=0) if y_val is not None else y_train
        stack.fit(X_tr_final.values, y_tr_final.values)
        y_pred_test = stack.predict(X_test.values)
        y_pred_train = stack.predict(X_tr_final.values)
        m_test = regression_metrics(y_test.values, y_pred_test)
        m_train = regression_metrics(y_tr_final.values, y_pred_train)
        # Original-scale metrics + smearing
        m_test_orig = None
        m_train_orig = None
        try:
            if log_applied_global:
                y_test_true_orig = (pd.read_parquet(pre_dir / (f"y_test_orig_{prefix}.parquet" if prefix else "y_test_orig.parquet")).iloc[:, 0].values
                                    if (pre_dir / (f"y_test_orig_{prefix}.parquet" if prefix else "y_test_orig.parquet")).exists()
                                    else np.expm1(y_test.values))
                residuals_log = y_tr_final.values - y_pred_train
                smear_s = float(np.mean(np.exp(residuals_log)))
                y_pred_test_orig = np.expm1(y_pred_test) * smear_s
                y_train_true_orig = np.expm1(y_tr_final.values)
                y_pred_train_orig = np.expm1(y_pred_train) * smear_s
                m_test_orig = regression_metrics(y_test_true_orig, y_pred_test_orig)
                m_train_orig = regression_metrics(y_train_true_orig, y_pred_train_orig)
                # mape floor
                price_cfg: Dict[str, Any] = gm_cfg.get("price_band", {}) if 'gm_cfg' in locals() else {}
                mape_floor = float(price_cfg.get("mape_floor", 1e-8))
                denom = np.where(np.abs(y_test_true_orig) < max(mape_floor, 1e-8), max(mape_floor, 1e-8), np.abs(y_test_true_orig))
                m_test_orig["mape_floor"] = float(np.mean(np.abs((y_test_true_orig - y_pred_test_orig) / denom)))
            else:
                m_test_orig = dict(m_test)
                m_train_orig = dict(m_train)
        except Exception:
            pass
        diag = overfit_diagnostics(m_train, m_test)
        ens_id = "stacking"
        ens_dir = models_dir / ens_id
        ens_dir.mkdir(parents=True, exist_ok=True)
        dump(stack, ens_dir / "model.pkl")
        meta = {
            "type": "stacking",
            "members": [k for k, _ in selected],
            "final_estimator": final_est_key,
            "metrics_train": m_train,
            "metrics_test": m_test,
            "metrics_train_original": m_train_orig,
            "metrics_test_original": m_test_orig,
            "overfit": diag,
        }
        (ens_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        results["ensembles"][ens_id] = meta
        table_rows.append({
            "Model": f"Ensemble_{ens_id}",
            "Category": "Ensemble",
            "Test_RMSE": m_test.get("rmse"),
            "Test_R2": m_test.get("r2"),
        })
        logger.info(f"[stacking] test r2={m_test['r2']:.4f} rmse={m_test['rmse']:.4f}")
        wb.log_prefixed_metrics("ensemble/stacking", {**{f"train_{k}": v for k, v in m_train.items()}, **{f"test_{k}": v for k, v in m_test.items()}})

        # Group metrics for stacking ensemble
        if gm_enabled:
            try:
                y_true_series = pd.Series(y_test_true_orig) if (log_applied_global and 'y_test_true_orig' in locals()) else y_test
                y_pred_series = pd.Series(y_pred_test_orig) if (log_applied_global and 'y_pred_test_orig' in locals()) else pd.Series(y_pred_test)
                group_cols_path = pre_dir / (f"group_cols_test_{prefix}.parquet" if prefix else "group_cols_test.parquet")
                if group_cols_path.exists():
                    grp_df = pd.read_parquet(group_cols_path)
                    gb_cols_cfg: List[str] = [c for c in (gm_cfg.get("group_by_columns", []) or []) if c in grp_df.columns]
                else:
                    grp_df = pd.DataFrame()
                    gb_cols_cfg = []
                for gb_col in gb_cols_cfg:
                    groups = grp_df[gb_col].fillna("MISSING")
                    gm_df = grouped_regression_metrics(
                        y_true=y_true_series,
                        y_pred=y_pred_series,
                        groups=groups,
                        report_metrics=gm_report_metrics,
                        min_group_size=gm_min_group_size,
                        mape_floor=float(price_cfg.get("mape_floor", 1e-8)),
                    )
                    if not gm_df.empty:
                        out_csv = ens_dir / f"group_metrics_{gb_col}.csv"
                        gm_df.to_csv(out_csv, index=False)
                        if gm_log_wandb:
                            try:
                                wb.log_artifact(out_csv, name=f"{ens_id}_{gb_col}_group_metrics.csv", type="metrics")
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"Per-group metrics failed for ensemble {ens_id}: {e}")

        # SHAP for stacking meta-model (explain base learner contributions)
        try:
            # Build meta features as predictions of base estimators
            try:
                base_ests = [est for est in stack.estimators_]
                base_names = [name for name, _ in selected]
            except Exception:
                base_ests = []
                base_names = []
            if base_ests:
                X_meta_train = np.column_stack([est.predict(X_tr_final.values) for est in base_ests])
                meta_est = getattr(stack, 'final_estimator_', None)
                if meta_est is not None:
                    shap_bundle = compute_shap(
                        model=meta_est,
                        X=X_meta_train,
                        sample_size=int(shap_cfg.get("sample_size", 2000)),
                        max_display=int(shap_cfg.get("max_display", 30)),
                        keep_as_numpy=True,
                        random_state=seed,
                        feature_names=base_names if len(base_names) == X_meta_train.shape[1] else None,
                    )
                    save_shap_plots(str(ens_dir / "shap_meta"), shap_bundle, f"{ens_id}_meta")
                    try:
                        wb.log_image(key=f"shap/{ens_id}_meta_beeswarm", image_path=ens_dir / "shap_meta" / f"shap_{ens_id}_meta_beeswarm.png")
                        wb.log_image(key=f"shap/{ens_id}_meta_bar", image_path=ens_dir / "shap_meta" / f"shap_{ens_id}_meta_bar.png")
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"SHAP meta-model for stacking failed: {e}")

    # Build and persist ranking table
    try:
        df_results = pd.DataFrame(table_rows)
        if not df_results.empty:
            df_results = df_results.sort_values(["Test_RMSE", "Test_R2"], ascending=[True, False]).reset_index(drop=True)
            df_results.to_csv(models_dir / "validation_results.csv", index=False)
            results["df_validation_results_path"] = str(models_dir / "validation_results.csv")
    except Exception as e:
        logger.warning(f"Impossibile generare ranking risultati: {e}")

    (models_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    try:
        wb.log_artifact(models_dir, name="models_dir", type="models", description="All trained models and metrics")
    except Exception:
        pass

    logger.info("Training/Tuning/Evaluation completati.")
    wb.finish()
    return results