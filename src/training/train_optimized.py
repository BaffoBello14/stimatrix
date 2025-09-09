from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json

import numpy as np
import pandas as pd
from joblib import dump

from utils.logger import get_logger
from .tuner_optimized import tune_model
from .model_zoo import build_estimator
from .metrics import regression_metrics, overfit_diagnostics, grouped_regression_metrics, _build_price_bands
from .metrics_optimized import (
    calculate_all_metrics, 
    calculate_overfit_metrics,
    calculate_weighted_score,
    calculate_robust_score,
    evaluate_model_stability
)
from .ensembles import build_voting, build_stacking
from .shap_utils import compute_shap, save_shap_plots
from utils.wandb_utils import WandbTracker

logger = get_logger(__name__)


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
    secondary_metrics: List[str] = tr_cfg.get("secondary_metrics", [])
    report_metrics: List[str] = tr_cfg.get("report_metrics", ["r2", "rmse", "mse", "mae", "mape"])
    sampler_name = tr_cfg.get("sampler", "auto")
    seed = int(tr_cfg.get("seed", 42))
    
    # Get tuning split fraction from temporal_split config (supports new nested schema)
    temporal_cfg = config.get("temporal_split", {})
    frac_cfg = temporal_cfg.get("fraction", {})
    tuning_split_fraction = float(frac_cfg.get("train", temporal_cfg.get("train_fraction", 0.8)))

    # Ottimizzazione configurations
    early_stopping_cfg = tr_cfg.get("early_stopping", {})
    overfitting_penalty_cfg = tr_cfg.get("overfitting_penalty", {})
    cv_during_tuning_cfg = tr_cfg.get("cv_during_tuning", {})

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

    # Read preprocessing info to know if log-transform was applied
    prep_info_path = pre_dir / "preprocessing_info.json"
    log_applied_global: bool = False
    if prep_info_path.exists():
        try:
            prep_info = json.loads(prep_info_path.read_text(encoding="utf-8"))
            log_applied_global = bool(((prep_info or {}).get("log_transformation", {}) or {}).get("applied", False))
        except Exception:
            log_applied_global = False

    cv_when_no_val = tr_cfg.get("cv_when_no_val", {})
    timeout = tr_cfg.get("timeout", None)
    n_jobs_default = int(tr_cfg.get("n_jobs_default", -1))

    all_results: Dict[str, Any] = {}
    profiles_seen = set()
    baseline_profiles_data: Dict[str, Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series], pd.DataFrame, pd.Series]] = {}

    ## Modelli singoli (main loop rimane compatibile)
    for mdl_key, mdl_cfg in tr_cfg.get("models", {}).items():
        if not mdl_cfg.get("enabled", True):
            logger.info(f"Model {mdl_key} disabled, skipping.")
            continue
        # Select profile (prioritize per-model profile over profile_map)
        prof_override = mdl_cfg.get("profile", None)
        if prof_override is not None:
            pf = prof_override
        else:
            pf = _profile_for(mdl_key, config)
        if pf and not config.get("profiles", {}).get(pf, {}).get("enabled", True):
            logger.info(f"Profile '{pf}' disabled for model {mdl_key}, skipping.")
            continue
        logger.info(f"=== Training model '{mdl_key}' (profile: {pf or 'none'}) ===")

        # Load data
        if pf is None:
            X_train, y_train, X_val, y_val, X_test, y_test = _load_xy(pre_dir, None)
            cat_features = None
        else:
            profiles_seen.add(pf)
            if pf in baseline_profiles_data:
                X_train, y_train, X_val, y_val, X_test, y_test = baseline_profiles_data[pf]
            else:
                X_train, y_train, X_val, y_val, X_test, y_test = _load_xy(pre_dir, pf)
                baseline_profiles_data[pf] = (X_train, y_train, X_val, y_val, X_test, y_test)
            if mdl_key.lower() == "catboost" and pf == "catboost":
                cat_features = _catboost_cat_features(pre_dir, pf, X_train)
            else:
                cat_features = None

        wb.log({"model": mdl_key, "profile": pf or "none", "n_train": len(X_train), "n_test": len(X_test), "n_features": X_train.shape[1]})

        # Parametri e ottimizzazione
        base_params = mdl_cfg.get("base_params", {}) or {}
        if "n_jobs" not in base_params and mdl_key.lower() in {"rf", "gbr", "hgbt", "xgboost", "lightgbm"}:
            base_params["n_jobs"] = n_jobs_default
        if "random_state" not in base_params:
            base_params["random_state"] = seed

        search_space = mdl_cfg.get("search_space", {})
        n_trials = int(mdl_cfg.get("trials", 100))

        result_dict: Dict[str, Any] = {"model": mdl_key, "profile": pf}
        if n_trials > 0 and search_space:
            # Optuna tuning con configurazioni ottimizzate
            logger.info(f"Running Optuna for {mdl_key} with {n_trials} trials...")
            tuning_result = tune_model(
                model_key=mdl_key,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                primary_metric=primary_metric,
                n_trials=n_trials,
                timeout=timeout,
                sampler_name=sampler_name,
                seed=seed,
                base_params=base_params,
                search_space=search_space,
                cat_features=cat_features,
                cv_config=cv_when_no_val if (X_val is None or y_val is None) else None,
                tuning_split_fraction=tuning_split_fraction,
                overfitting_penalty_config=overfitting_penalty_cfg,
                cv_during_tuning_config=cv_during_tuning_cfg,
            )
            best_params = {**base_params, **tuning_result.best_params}
            logger.info(f"Best params for {mdl_key}: {tuning_result.best_params}")
            logger.info(f"Best {primary_metric}: {tuning_result.best_value}")
            
            # Log overfitting scores se disponibili
            if tuning_result.overfit_scores:
                logger.info(f"Overfitting metrics: {tuning_result.overfit_scores}")
                wb.log({f"{mdl_key}_overfit": tuning_result.overfit_scores})
            
            result_dict["best_params"] = tuning_result.best_params
            result_dict["best_primary_value"] = tuning_result.best_value
        else:
            best_params = base_params
            result_dict["best_params"] = {}
            result_dict["best_primary_value"] = None

        # Train finale
        logger.info(f"Training final model {mdl_key}...")
        model = build_estimator(mdl_key, best_params)
        
        # Applica early stopping configuration se disponibile
        if mdl_key.lower() in ["xgboost", "lightgbm", "catboost", "hgbt"] and X_val is not None:
            fit_params = mdl_cfg.get("fit_params", {}).copy()
            
            # Early stopping dinamico basato sulla configurazione
            es_rounds = early_stopping_cfg.get("patience", 100)
            
            if mdl_key.lower() == "xgboost":
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                         verbose=False, early_stopping_rounds=es_rounds)
            elif mdl_key.lower() == "lightgbm":
                import lightgbm as lgb
                model.fit(X_train.values, y_train.values, 
                         eval_set=[(X_val.values, y_val.values)],
                         callbacks=[lgb.early_stopping(es_rounds, verbose=False)])
            elif mdl_key.lower() == "catboost":
                if cat_features:
                    model.fit(X_train, y_train, cat_features=cat_features, 
                             eval_set=(X_val, y_val), verbose=False, 
                             early_stopping_rounds=es_rounds)
                else:
                    model.fit(X_train, y_train, eval_set=(X_val, y_val), 
                             verbose=False, early_stopping_rounds=es_rounds)
            elif mdl_key.lower() == "hgbt":
                # HGBT ha early stopping built-in se configurato nei parametri
                model.fit(X_train, y_train)
        else:
            # Standard fit
            fit_params = mdl_cfg.get("fit_params", {}).copy()
            if mdl_key.lower() == "catboost" and cat_features is not None:
                indices_ph = "__categorical_indices__"
                if fit_params.get("cat_features") == indices_ph:
                    fit_params["cat_features"] = cat_features
            model.fit(X_train, y_train, **fit_params)

        # Salva modello
        model_path = models_dir / f"{mdl_key}_model.joblib"
        dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        result_dict["model_path"] = str(model_path)

        # Valutazione con metriche ottimizzate
        logger.info(f"Evaluating {mdl_key} on test set...")
        y_pred_test = model.predict(X_test)
        
        # Calcola tutte le metriche
        metrics_test = calculate_all_metrics(y_test, y_pred_test)
        result_dict["metrics_test"] = metrics_test
        
        # Log metriche principali
        logger.info(f"Test R2: {metrics_test['r2']:.4f}")
        logger.info(f"Test RMSE: {metrics_test['rmse']:.2f}")
        logger.info(f"Test MAE: {metrics_test['mae']:.2f}")
        
        wb.log({f"{mdl_key}_test_{k}": v for k, v in metrics_test.items()})

        # Calcola metriche su training per overfitting analysis
        y_pred_train = model.predict(X_train)
        metrics_train = calculate_all_metrics(y_train, y_pred_train)
        result_dict["metrics_train"] = metrics_train
        
        # Calcola metriche di overfitting
        overfit_metrics = calculate_overfit_metrics(metrics_train, metrics_test)
        result_dict["overfit"] = overfit_metrics
        
        logger.info(f"Overfitting - R2 gap: {overfit_metrics['gap_r2']:.4f}")
        logger.info(f"Overfitting - RMSE ratio: {overfit_metrics['ratio_rmse']:.2f}")
        
        wb.log({f"{mdl_key}_overfit_{k}": v for k, v in overfit_metrics.items()})

        # Valuta stabilità del modello (opzionale, solo per modelli selezionati)
        if mdl_key.lower() in ["rf", "xgboost", "lightgbm", "catboost"]:
            logger.info(f"Evaluating model stability for {mdl_key}...")
            stability_metrics = evaluate_model_stability(model, X_test, y_test, n_bootstrap=5, seed=seed)
            result_dict["stability"] = stability_metrics
            logger.info(f"Prediction stability: {stability_metrics.get('prediction_stability', 0):.4f}")
            wb.log({f"{mdl_key}_stability": stability_metrics.get('prediction_stability', 0)})

        # SHAP analysis
        if shap_cfg.get("enabled", True):
            logger.info(f"Computing SHAP values for {mdl_key}...")
            shap_values = compute_shap(model, X_test, 
                                     sample_size=shap_cfg.get("sample_size", 500),
                                     max_display=shap_cfg.get("max_display", 30))
            if shap_values is not None and shap_cfg.get("save_plots", True):
                save_shap_plots(shap_values, X_test, models_dir / f"{mdl_key}_shap.png")
                logger.info(f"SHAP plots saved for {mdl_key}")
                wb.log({f"{mdl_key}_shap": wb.Image(str(models_dir / f"{mdl_key}_shap.png"))})
            if shap_values is not None and shap_cfg.get("save_values", True):
                shap_path = models_dir / f"{mdl_key}_shap_values.npy"
                np.save(shap_path, shap_values)
                result_dict["shap_values_path"] = str(shap_path)

        # Group metrics
        if gm_enabled:
            for gcol in gm_cfg.get("group_by_columns", []):
                if gcol not in X_test.columns:
                    logger.warning(f"Group column '{gcol}' not found in X_test, skipping group metrics.")
                    continue
                logger.info(f"Computing group metrics by '{gcol}' for {mdl_key}...")
                gm_res = grouped_regression_metrics(
                    y_true=y_test,
                    y_pred=y_pred_test,
                    groups=X_test[gcol],
                    metrics=gm_report_metrics,
                    min_group_size=gm_min_group_size,
                    y_pred_log=log_applied_global,
                    y_true_log=log_applied_global,
                    original_scale=gm_original_scale
                )
                if gm_res and gm_log_wandb:
                    df_gm = pd.DataFrame(gm_res)
                    wb.log({f"{mdl_key}_group_metrics_{gcol}": wb.Table(dataframe=df_gm)})
                key_gm = f"group_metrics_{gcol}"
                if key_gm not in result_dict:
                    result_dict[key_gm] = {}
                result_dict[key_gm][mdl_key] = gm_res

            # Price band analysis
            if price_cfg:
                logger.info(f"Computing price band metrics for {mdl_key}...")
                price_edges = _build_price_bands(y_test, price_cfg)
                if price_edges is not None:
                    gm_price = grouped_regression_metrics(
                        y_true=y_test,
                        y_pred=y_pred_test,
                        groups=pd.cut(y_test, bins=price_edges, labels=False, include_lowest=True),
                        metrics=gm_report_metrics,
                        min_group_size=gm_min_group_size,
                        y_pred_log=log_applied_global,
                        y_true_log=log_applied_global,
                        original_scale=gm_original_scale
                    )
                    if gm_price and gm_log_wandb:
                        df_gm_price = pd.DataFrame(gm_price)
                        wb.log({f"{mdl_key}_price_band_metrics": wb.Table(dataframe=df_gm_price)})
                    result_dict["price_band_metrics"] = gm_price

        all_results[mdl_key] = result_dict

    ## Baseline comparison
    logger.info("=== Training baseline models for comparison ===")
    baselines = {}
    for mdl_key, mdl_cfg in tr_cfg.get("models", {}).items():
        if not mdl_cfg.get("enabled", True):
            continue
        prof_override = mdl_cfg.get("profile", None)
        if prof_override is not None:
            pf = prof_override
        else:
            pf = _profile_for(mdl_key, config)
        if pf and not config.get("profiles", {}).get(pf, {}).get("enabled", True):
            continue
            
        logger.info(f"Training baseline {mdl_key}...")
        
        # Load data
        if pf is None:
            X_train, y_train, X_val, y_val, X_test, y_test = _load_xy(pre_dir, None)
            cat_features = None
        else:
            if pf in baseline_profiles_data:
                X_train, y_train, X_val, y_val, X_test, y_test = baseline_profiles_data[pf]
            else:
                X_train, y_train, X_val, y_val, X_test, y_test = _load_xy(pre_dir, pf)
                baseline_profiles_data[pf] = (X_train, y_train, X_val, y_val, X_test, y_test)
            if mdl_key.lower() == "catboost" and pf == "catboost":
                cat_features = _catboost_cat_features(pre_dir, pf, X_train)
            else:
                cat_features = None

        # Train con parametri di default
        base_params = mdl_cfg.get("base_params", {}) or {}
        if "n_jobs" not in base_params and mdl_key.lower() in {"rf", "gbr", "hgbt", "xgboost", "lightgbm"}:
            base_params["n_jobs"] = n_jobs_default
        if "random_state" not in base_params:
            base_params["random_state"] = seed
            
        model = build_estimator(mdl_key, base_params)
        
        fit_params = mdl_cfg.get("fit_params", {}).copy()
        if mdl_key.lower() == "catboost" and cat_features is not None:
            indices_ph = "__categorical_indices__"
            if fit_params.get("cat_features") == indices_ph:
                fit_params["cat_features"] = cat_features
        
        model.fit(X_train, y_train, **fit_params)
        y_pred_test = model.predict(X_test)
        
        metrics_test = calculate_all_metrics(y_test, y_pred_test)
        baselines[mdl_key] = {"metrics_test": metrics_test}
        
        logger.info(f"Baseline {mdl_key} - R2: {metrics_test['r2']:.4f}, RMSE: {metrics_test['rmse']:.2f}")
        wb.log({f"baseline_{mdl_key}_test_{k}": v for k, v in metrics_test.items()})

    ## Ensembles
    ens_cfg = tr_cfg.get("ensembles", {})
    ens_results = {}

    # Carica tutti i modelli salvati per gli ensemble
    saved_models = {}
    for mdl_key in all_results:
        model_path = Path(all_results[mdl_key].get("model_path", ""))
        if model_path.exists():
            from joblib import load
            saved_models[mdl_key] = load(model_path)

    if ens_cfg.get("voting", {}).get("enabled", True) and saved_models:
        logger.info("=== Building Voting Ensemble ===")
        # Seleziona top N modelli basati sulla metrica primaria
        top_n = ens_cfg["voting"].get("top_n", 3)
        model_scores = [(k, all_results[k]["metrics_test"][primary_metric.replace("neg_", "").replace("_", "")]) 
                       for k in saved_models if "metrics_test" in all_results[k]]
        
        # Ordina per score (invertendo se necessario per metriche negative)
        if "neg_" in primary_metric:
            model_scores.sort(key=lambda x: x[1])  # Lower is better for errors
        else:
            model_scores.sort(key=lambda x: x[1], reverse=True)  # Higher is better for R2
        
        top_models = [k for k, _ in model_scores[:top_n]]
        logger.info(f"Top {top_n} models for voting: {top_models}")
        
        # Costruisci ensemble (assumendo stesso profilo per semplicità)
        if top_models:
            # Usa il profilo del primo modello
            pf = all_results[top_models[0]].get("profile")
            if pf in baseline_profiles_data:
                X_train, y_train, X_val, y_val, X_test, y_test = baseline_profiles_data[pf]
            else:
                X_train, y_train, X_val, y_val, X_test, y_test = _load_xy(pre_dir, pf)
            
            voting_models = [(k, saved_models[k]) for k in top_models]
            voting_ens = build_voting(voting_models)
            
            logger.info("Training voting ensemble...")
            voting_ens.fit(X_train, y_train)
            
            y_pred_voting = voting_ens.predict(X_test)
            metrics_voting = calculate_all_metrics(y_test, y_pred_voting)
            
            logger.info(f"Voting Ensemble - R2: {metrics_voting['r2']:.4f}, RMSE: {metrics_voting['rmse']:.2f}")
            wb.log({f"voting_ensemble_test_{k}": v for k, v in metrics_voting.items()})
            
            # Calcola anche train metrics per overfitting
            y_pred_voting_train = voting_ens.predict(X_train)
            metrics_voting_train = calculate_all_metrics(y_train, y_pred_voting_train)
            overfit_voting = calculate_overfit_metrics(metrics_voting_train, metrics_voting)
            
            ens_results["voting"] = {
                "type": "voting",
                "members": top_models,
                "metrics_train": metrics_voting_train,
                "metrics_test": metrics_voting,
                "overfit": overfit_voting
            }
            
            # Salva ensemble
            voting_path = models_dir / "voting_ensemble.joblib"
            dump(voting_ens, voting_path)
            logger.info(f"Voting ensemble saved to {voting_path}")

    if ens_cfg.get("stacking", {}).get("enabled", True) and saved_models:
        logger.info("=== Building Stacking Ensemble ===")
        top_n = ens_cfg["stacking"].get("top_n", 5)
        final_est = ens_cfg["stacking"].get("final_estimator", "ridge")
        cv_folds = ens_cfg["stacking"].get("cv_folds", 5)
        
        # Seleziona top N modelli
        model_scores = [(k, all_results[k]["metrics_test"][primary_metric.replace("neg_", "").replace("_", "")]) 
                       for k in saved_models if "metrics_test" in all_results[k]]
        
        if "neg_" in primary_metric:
            model_scores.sort(key=lambda x: x[1])
        else:
            model_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_models = [k for k, _ in model_scores[:top_n]]
        logger.info(f"Top {top_n} models for stacking: {top_models}")
        
        if top_models:
            pf = all_results[top_models[0]].get("profile")
            if pf in baseline_profiles_data:
                X_train, y_train, X_val, y_val, X_test, y_test = baseline_profiles_data[pf]
            else:
                X_train, y_train, X_val, y_val, X_test, y_test = _load_xy(pre_dir, pf)
            
            stacking_models = [(k, saved_models[k]) for k in top_models]
            stacking_ens = build_stacking(stacking_models, final_estimator=final_est, cv_folds=cv_folds)
            
            logger.info("Training stacking ensemble...")
            stacking_ens.fit(X_train, y_train)
            
            y_pred_stacking = stacking_ens.predict(X_test)
            metrics_stacking = calculate_all_metrics(y_test, y_pred_stacking)
            
            logger.info(f"Stacking Ensemble - R2: {metrics_stacking['r2']:.4f}, RMSE: {metrics_stacking['rmse']:.2f}")
            wb.log({f"stacking_ensemble_test_{k}": v for k, v in metrics_stacking.items()})
            
            # Train metrics
            y_pred_stacking_train = stacking_ens.predict(X_train)
            metrics_stacking_train = calculate_all_metrics(y_train, y_pred_stacking_train)
            overfit_stacking = calculate_overfit_metrics(metrics_stacking_train, metrics_stacking)
            
            ens_results["stacking"] = {
                "type": "stacking",
                "members": top_models,
                "final_estimator": final_est,
                "metrics_train": metrics_stacking_train,
                "metrics_test": metrics_stacking,
                "overfit": overfit_stacking
            }
            
            # Salva ensemble
            stacking_path = models_dir / "stacking_ensemble.joblib"
            dump(stacking_ens, stacking_path)
            logger.info(f"Stacking ensemble saved to {stacking_path}")

    # Crea report finale con confronto
    logger.info("=== Creating final comparison report ===")
    
    # Crea DataFrame per confronto
    comparison_data = []
    
    # Aggiungi risultati modelli ottimizzati
    for mdl_key, res in all_results.items():
        if "metrics_test" in res:
            row = {
                "Model": f"Optimized_{mdl_key}",
                "Category": "Optimized",
                "Test_RMSE": res["metrics_test"]["rmse"],
                "Test_R2": res["metrics_test"]["r2"],
                "Test_MAE": res["metrics_test"]["mae"],
                "Overfit_Gap_R2": res.get("overfit", {}).get("gap_r2", None),
                "Overfit_Ratio_RMSE": res.get("overfit", {}).get("ratio_rmse", None)
            }
            comparison_data.append(row)
    
    # Aggiungi baseline
    for mdl_key, res in baselines.items():
        row = {
            "Model": f"Baseline_{mdl_key}",
            "Category": "Baseline",
            "Test_RMSE": res["metrics_test"]["rmse"],
            "Test_R2": res["metrics_test"]["r2"],
            "Test_MAE": res["metrics_test"]["mae"],
            "Overfit_Gap_R2": None,
            "Overfit_Ratio_RMSE": None
        }
        comparison_data.append(row)
    
    # Aggiungi ensemble
    for ens_key, res in ens_results.items():
        row = {
            "Model": f"Ensemble_{ens_key}",
            "Category": "Ensemble",
            "Test_RMSE": res["metrics_test"]["rmse"],
            "Test_R2": res["metrics_test"]["r2"],
            "Test_MAE": res["metrics_test"]["mae"],
            "Overfit_Gap_R2": res.get("overfit", {}).get("gap_r2", None),
            "Overfit_Ratio_RMSE": res.get("overfit", {}).get("ratio_rmse", None)
        }
        comparison_data.append(row)
    
    # Crea e salva DataFrame di confronto
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values("Test_RMSE")  # Ordina per RMSE (lower is better)
    
    comparison_path = models_dir / "model_comparison_optimized.csv"
    df_comparison.to_csv(comparison_path, index=False)
    logger.info(f"Model comparison saved to {comparison_path}")
    
    # Log to W&B
    wb.log({"model_comparison": wb.Table(dataframe=df_comparison)})
    
    # Stampa top 5 modelli
    logger.info("\n=== TOP 5 MODELS ===")
    for idx, row in df_comparison.head(5).iterrows():
        logger.info(f"{row['Model']} ({row['Category']}): R2={row['Test_R2']:.4f}, RMSE={row['Test_RMSE']:.2f}")

    wb.finish()
    
    return {
        "models": all_results,
        "baselines": baselines,
        "ensembles": ens_results,
        "comparison": df_comparison.to_dict('records'),
        "comparison_path": str(comparison_path)
    }