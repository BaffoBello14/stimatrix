from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json

import numpy as np
import pandas as pd
from joblib import dump

from utils.logger import get_logger
from utils.detailed_logging import DetailedLogger
from .tuner import tune_model
from .model_zoo import build_estimator
from .metrics import regression_metrics, overfit_diagnostics
from .ensembles import build_voting, build_stacking
from .shap_utils import compute_shap, save_shap_plots
from .feature_importance_advanced import AdvancedFeatureImportance
from .evaluation_advanced import MultiScaleEvaluation, ResidualAnalyzer

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
    """
    Esegue training completo di modelli multipli con evaluation avanzata.
    
    La funzione implementa:
    - Training modelli multipli con profili specifici
    - Ottimizzazione iperparametri con Optuna
    - Feature importance multi-metodo (Built-in + Permutation + SHAP)
    - Evaluation dual-scale (trasformata + originale)
    - Ensemble models con voting e stacking
    - Tracking performance e salvataggio artifacts
    
    Args:
        config: Configurazione completa con sezioni training, paths, target, etc.
                Include configurazioni per modelli, profili, SHAP, ensemble
                
    Returns:
        Dict con risultati training, metriche, modelli addestrati e artifacts
    """
    # Setup paths e directories
    paths = config.get("paths", {})
    pre_dir = Path(paths.get("preprocessed_data", "data/preprocessed"))
    models_dir = Path(paths.get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Inizializza sistemi avanzati
    feature_importance_system = AdvancedFeatureImportance(config)
    multi_scale_evaluator = MultiScaleEvaluation(config)
    
    logger.info(f"🚀 Training inizializzato - Output: {models_dir}")

    # Configurazione training
    tr_cfg = config.get("training", {})
    primary_metric: str = tr_cfg.get("primary_metric", "r2")
    report_metrics: List[str] = tr_cfg.get("report_metrics", ["r2", "rmse", "mse", "mae", "mape"])
    sampler_name = tr_cfg.get("sampler", "auto")
    seed = int(tr_cfg.get("seed", 42))
    
    # Frazione split per tuning (consistenza con preprocessing)
    temporal_cfg = config.get("temporal_split", {})
    tuning_split_fraction = float(temporal_cfg.get("train_fraction", 0.8))

    # Configurazione SHAP ottimizzata
    shap_cfg = tr_cfg.get("shap", {"enabled": True})
    shap_enabled = bool(shap_cfg.get('enabled', True))
    shap_sample_size = int(shap_cfg.get('sample_size', 500))  # Ridotto per performance

    # Raccolta modelli abilitati con validazione
    models_cfg: Dict[str, Any] = tr_cfg.get("models", {})
    selected_models: List[str] = [k for k, v in models_cfg.items() if bool(v.get("enabled", False))]
    
    if not selected_models:
        logger.warning("⚠️ Nessun modello abilitato - abilito linear come fallback")
        selected_models = ["linear"]
    
    logger.info(f"🤖 Modelli da addestrare: {selected_models}")
    logger.info(f"📊 Metrica primaria: {primary_metric}")
    logger.info(f"🔍 SHAP abilitato: {shap_enabled} (sample_size={shap_sample_size})")

    # Inizializza risultati con tracking avanzato
    results: Dict[str, Any] = {
        "models": {}, 
        "ensembles": {},
        "training_summary": {
            "models_requested": selected_models,
            "models_completed": [],
            "models_failed": [],
            "start_time": pd.Timestamp.now().isoformat()
        },
        "feature_importance": {},
        "evaluation_results": {}
    }

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

        # Tuning
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
        diag = overfit_diagnostics(m_train, m_test)

        model_id = f"{model_key}"
        model_dir = models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        dump(estimator, model_dir / "model.pkl")
        meta = {
            "model_key": model_key,
            "prefix": prefix,
            "primary_metric": primary_metric,
            "best_primary_value": tuning.best_value,
            "best_params": tuning.best_params,
            "metrics_train": m_train,
            "metrics_test": m_test,
            "overfit": diag,
        }
        (model_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        try:
            df_trials = tuning.study.trials_dataframe()
            df_trials.to_csv(model_dir / "optuna_trials.csv", index=False)
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
                )
                if bool(shap_cfg.get("save_plots", True)):
                    save_shap_plots(str(model_dir / "shap"), shap_bundle, model_id)
                if bool(shap_cfg.get("save_values", False)):
                    np.save(model_dir / "shap_values.npy", shap_bundle["values"].values, allow_pickle=False)
                    shap_bundle["data_sample"].to_parquet(model_dir / "shap_sample.parquet", index=False)
            except Exception as e:
                logger.warning(f"SHAP fallito per {model_key}: {e}")

        results["models"][model_id] = {
            "best_params": tuning.best_params,
            "best_primary_value": tuning.best_value,
            "metrics_test": m_test,
            "metrics_train": m_train,
            "overfit": diag,
        }
        logger.info(f"[{model_key}] best {primary_metric}={tuning.best_value:.6f} | test r2={m_test['r2']:.4f} rmse={m_test['rmse']:.4f}")

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
            "overfit": diag,
        }
        (ens_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        results["ensembles"][ens_id] = meta
        logger.info(f"[voting] test r2={m_test['r2']:.4f} rmse={m_test['rmse']:.4f}")

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
            "overfit": diag,
        }
        (ens_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        results["ensembles"][ens_id] = meta
        logger.info(f"[stacking] test r2={m_test['r2']:.4f} rmse={m_test['rmse']:.4f}")

    # Feature Importance Analysis Avanzata
    logger.info("🧠 Calcolo feature importance avanzata...")
    
    try:
        # Carica dati per feature importance (usa primo profilo disponibile)
        first_profile = None
        for model_key in selected_models:
            model_entry = models_cfg.get(model_key, {})
            profile = model_entry.get("profile", None)
            if profile:
                first_profile = profile
                break
        
        if first_profile and results["models"]:
            X_train_fi, y_train_fi, X_val_fi, y_val_fi, X_test_fi, y_test_fi = _load_xy(pre_dir, first_profile)
            
            # Prepara modelli per feature importance
            trained_models = {}
            for model_id, model_results in results["models"].items():
                if "model" in model_results:
                    trained_models[model_id] = {
                        'model': model_results["model"],
                        'name': model_id,
                        'type': model_results.get('type', 'unknown')
                    }
            
            # Calcola feature importance comprehensive
            if trained_models:
                comprehensive_importance = feature_importance_system.calculate_comprehensive_importance(
                    trained_models, X_train_fi, X_test_fi, y_test_fi, list(X_train_fi.columns)
                )
                
                results["feature_importance"] = comprehensive_importance
                
                # Salva plot feature importance
                fi_output_dir = models_dir / "feature_importance"
                fi_output_dir.mkdir(exist_ok=True)
                feature_importance_system.save_importance_plots(
                    comprehensive_importance, str(fi_output_dir), top_n=20
                )
                
                logger.info(f"✅ Feature importance calcolata per {len(trained_models)} modelli")
            else:
                logger.warning("⚠️ Nessun modello valido per feature importance")
                
    except Exception as e:
        logger.error(f"❌ Errore feature importance: {e}")
        results["feature_importance"] = {"error": str(e)}

    # Evaluation Multi-Scala Avanzata
    logger.info("📈 Evaluation multi-scala...")
    
    try:
        if results["models"] and first_profile:
            # Determina se è stata applicata trasformazione log
            target_config = config.get("target", {})
            log_transform_applied = target_config.get("log_transform", False)
            
            transform_info = {"log": log_transform_applied}
            
            # Prepara target originale (se log applicato, y_test è già in scala log)
            if log_transform_applied:
                y_test_original = np.expm1(y_test_fi)  # Inverse log1p
                y_test_transformed = y_test_fi
            else:
                y_test_original = y_test_fi
                y_test_transformed = y_test_fi
            
            # Evaluation multi-scala per tutti i modelli
            multi_scale_results = multi_scale_evaluator.evaluate_multiple_models_dual_scale(
                trained_models, X_test_fi, y_test_transformed, y_test_original, transform_info
            )
            
            results["evaluation_results"] = multi_scale_results
            
            # Salva visualizzazioni performance
            eval_output_dir = models_dir / "evaluation"
            eval_output_dir.mkdir(exist_ok=True)
            
            # Crea report comparativo
            from .evaluation_advanced import ModelComparator
            comparison_report = ModelComparator.create_comparison_report(
                multi_scale_results["models"], 
                output_file=str(eval_output_dir / "models_comparison.csv")
            )
            
            # Crea visualizzazioni
            ModelComparator.create_performance_visualization(
                multi_scale_results["models"], str(eval_output_dir)
            )
            
            logger.info(f"✅ Evaluation multi-scala completata - Report: {eval_output_dir}")
            
    except Exception as e:
        logger.error(f"❌ Errore evaluation multi-scala: {e}")
        results["evaluation_results"] = {"error": str(e)}

    # Finalizza summary training
    results["training_summary"]["end_time"] = pd.Timestamp.now().isoformat()
    results["training_summary"]["models_completed"] = [k for k in results["models"].keys()]
    results["training_summary"]["total_models_trained"] = len(results["models"])
    results["training_summary"]["total_ensembles_created"] = len(results["ensembles"])
    
    # Salva summary completo
    (models_dir / "training_summary.json").write_text(
        json.dumps(results["training_summary"], indent=2, default=str), encoding="utf-8"
    )
    
    # Salva risultati completi (backward compatibility)
    (models_dir / "summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Log finale con statistiche
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETATO CON SUCCESSO")
    logger.info("=" * 60)
    logger.info(f"🤖 Modelli addestrati: {len(results['models'])}")
    logger.info(f"🔗 Ensemble creati: {len(results['ensembles'])}")
    logger.info(f"🧠 Feature importance: {'✅' if 'error' not in results['feature_importance'] else '❌'}")
    logger.info(f"📊 Evaluation multi-scala: {'✅' if 'error' not in results['evaluation_results'] else '❌'}")
    logger.info(f"📁 Artifacts salvati in: {models_dir}")
    logger.info("=" * 60)
    
    return results