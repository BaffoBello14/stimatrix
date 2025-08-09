from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json

import numpy as np
import pandas as pd
from joblib import dump

from utils.logger import get_logger
from .tuner import tune_model
from .model_zoo import build_estimator
from .metrics import regression_metrics, overfit_diagnostics
from .ensembles import build_voting, build_stacking
from .shap_utils import compute_shap, save_shap_plots

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

    tr_cfg = config.get("training", {})
    primary_metric: str = tr_cfg.get("primary_metric", "r2")
    report_metrics: List[str] = tr_cfg.get("report_metrics", ["r2", "rmse", "mse", "mae", "mape"])

    opt_cfg = tr_cfg.get("optuna", {})
    n_trials = int(opt_cfg.get("n_trials", 50))
    timeout = opt_cfg.get("timeout", None)
    timeout = None if timeout in (None, "null") else int(timeout)
    sampler_name = opt_cfg.get("sampler", "auto")
    seed = int(opt_cfg.get("seed", 42))

    shap_cfg = tr_cfg.get("shap", {"enabled": True})

    selected_models: List[str] = tr_cfg.get("models", ["ridge", "rf", "lightgbm"])  # default

    results: Dict[str, Any] = {"models": {}, "ensembles": {}}

    # Per-model loop
    for model_key in selected_models:
        prefix = _profile_for(model_key, config)
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = _load_xy(pre_dir, prefix)
        except Exception as e:
            logger.error(f"Caricamento dataset fallito per modello {model_key} (prefix={prefix}): {e}")
            continue

        cat_features: Optional[List[int]] = None
        if model_key.lower() == "catboost":
            if prefix is None:
                logger.warning("CatBoost richiede il profilo 'catboost'. Provo inferenza colonne categoriche.")
                cat_features = [i for i, dt in enumerate(X_train.dtypes) if str(dt) in ("object", "category")]
            else:
                cat_features = _catboost_cat_features(pre_dir, prefix, X_train)

        # Tuning
        tuning = tune_model(
            model_key=model_key,
            X_train=X_train.values if model_key != "catboost" else X_train,
            y_train=y_train.values,
            X_val=None if X_val is None else (X_val.values if model_key != "catboost" else X_val),
            y_val=None if y_val is None else y_val.values,
            primary_metric=primary_metric,
            n_trials=n_trials,
            timeout=timeout,
            sampler_name=sampler_name,
            seed=seed,
            cat_features=cat_features,
        )

        # Retrain on train+val with best params
        if X_val is not None and y_val is not None:
            X_tr_final = pd.concat([X_train, X_val], axis=0)
            y_tr_final = pd.concat([y_train, y_val], axis=0)
        else:
            X_tr_final, y_tr_final = X_train, y_train

        estimator = build_estimator(model_key, tuning.best_params)
        if model_key == "catboost" and cat_features is not None:
            estimator.fit(X_tr_final, y_tr_final, cat_features=cat_features, verbose=False)
        else:
            estimator.fit(X_tr_final.values if model_key != "catboost" else X_tr_final, y_tr_final.values)

        # Evaluate
        y_pred_test = estimator.predict(X_test.values if model_key != "catboost" else X_test)
        y_pred_train = estimator.predict(X_tr_final.values if model_key != "catboost" else X_tr_final)

        m_test = regression_metrics(y_test.values, y_pred_test)
        m_train = regression_metrics(y_tr_final.values, y_pred_train)
        diag = overfit_diagnostics(m_train, m_test)

        model_id = f"{model_key}"
        model_dir = models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        # Save model and metadata
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
        # Save study
        try:
            df_trials = tuning.study.trials_dataframe()
            df_trials.to_csv(model_dir / "optuna_trials.csv", index=False)
        except Exception:
            pass

        # SHAP
        if bool(shap_cfg.get("enabled", True)):
            try:
                shap_bundle = compute_shap(
                    model=estimator,
                    X=X_tr_final if model_key == "catboost" else X_tr_final,
                    sample_size=int(shap_cfg.get("sample_size", 2000)),
                    max_display=int(shap_cfg.get("max_display", 30)),
                )
                if bool(shap_cfg.get("save_plots", True)):
                    save_shap_plots(str(model_dir / "shap"), shap_bundle, model_id)
                if bool(shap_cfg.get("save_values", False)):
                    # Beware: large files
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
    # Collect top models by their validation best score (we used primary metric on val)
    ranked = sorted(
        [
            (k, v["best_params"], v["best_primary_value"]) for k, v in results.get("models", {}).items()
        ],
        key=lambda x: x[2],
        reverse=True,
    )

    def load_for_inference(key: str):
        from joblib import load
        return load(models_dir / key / "model.pkl")

    # Voting
    if ens_cfg.get("voting", {}).get("enabled", False) and len(ranked) >= 2:
        top_n = int(ens_cfg.get("voting", {}).get("top_n", 3))
        selected = [(k, p) for (k, p, _) in ranked[:top_n]]
        vote = build_voting(selected, tune_weights=bool(ens_cfg.get("voting", {}).get("tune_weights", True)))
        # Use first model's dataset as reference
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

    # Stacking
    if ens_cfg.get("stacking", {}).get("enabled", False) and len(ranked) >= 2:
        top_n = int(ens_cfg.get("stacking", {}).get("top_n", 5))
        final_est_key = str(ens_cfg.get("stacking", {}).get("final_estimator", "ridge"))
        cv_folds = int(ens_cfg.get("stacking", {}).get("cv_folds", 5))
        selected = [(k, p) for (k, p, _) in ranked[:top_n]]
        stack = build_stacking(selected, final_estimator_key=final_est_key, cv_folds=cv_folds)
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

    # Save global summary
    (models_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    logger.info("Training/Tuning/Evaluation completati.")
    return results