from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, TYPE_CHECKING

import numpy as np
import optuna
import optunahub
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold, TimeSeriesSplit

from .model_zoo import build_estimator
from .metrics import select_primary_value, regression_metrics
from .constants import CATBOOST_KEY, XGBOOST_KEY, LIGHTGBM_KEY, DEFAULT_TUNING_SPLIT_FRACTION

if TYPE_CHECKING:
    from utils.wandb_utils import WandbTracker


@dataclass
class TuningResult:
    best_params: Dict[str, Any]
    best_value: float
    study: optuna.Study


def _apply_suggestions(trial: optuna.Trial, space: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    params = base.copy()
    for name, spec in (space or {}).items():
        t = str(spec.get("type", "")).lower()
        if t == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
        elif t == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec.get("choices", []))
        else:
            if "value" in spec:
                params[name] = spec["value"]
    return params


def tune_model(
    model_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    primary_metric: str,
    n_trials: int,
    timeout: Optional[int],
    sampler_name: str,
    seed: int,
    base_params: Dict[str, Any],
    search_space: Optional[Dict[str, Any]] = None,
    cat_features: Optional[List[int]] = None,
    cv_config: Optional[Dict[str, Any]] = None,
    tuning_split_fraction: float = DEFAULT_TUNING_SPLIT_FRACTION,  # fraction for temporal split (consistent with preprocessing)
    wandb_manager: Optional['WandbTracker'] = None,  # for per-trial logging
) -> TuningResult:
    direction = "maximize"

    if sampler_name == "auto":
        try:
            module = optunahub.load_module(package="samplers/auto_sampler")
            sampler = module.AutoSampler(seed=seed)
        except Exception:
            sampler = optuna.samplers.TPESampler(seed=seed)
    elif sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)

    def _log_trial_to_wandb(trial_number: int, y_true: np.ndarray, y_pred: np.ndarray, trial_params: Dict[str, Any]) -> None:
        """Helper to log trial metrics to W&B"""
        if wandb_manager is None:
            return
        try:
            # Compute full metrics for this trial
            metrics = regression_metrics(y_true, y_pred)
            # Log trial metrics (without step parameter - let W&B auto-increment)
            log_dict = {
                f"tuning/{model_key}/trial_number": trial_number,
                f"tuning/{model_key}/{primary_metric}": metrics.get(primary_metric.replace("neg_", ""), 0.0),
                f"tuning/{model_key}/val_r2": metrics.get("r2", 0.0),
                f"tuning/{model_key}/val_rmse": metrics.get("rmse", 0.0),
                f"tuning/{model_key}/val_mae": metrics.get("mae", 0.0),
                f"tuning/{model_key}/val_mape": metrics.get("mape", 0.0),
            }
            # Optionally log hyperparameters (can be noisy, disabled by default)
            # for k, v in trial_params.items():
            #     if isinstance(v, (int, float, bool)):
            #         log_dict[f"tuning/{model_key}/hp_{k}"] = v
            wandb_manager.log(log_dict)  # No step parameter - auto-increment by W&B
        except Exception:
            pass  # Fail silently to not break tuning

    def objective(trial: optuna.Trial) -> float:
        params = _apply_suggestions(trial, search_space or {}, base_params or {})
        # Guardia SVR: degree ha senso solo con kernel 'poly'; coef0 solo per 'poly' o 'sigmoid'
        if model_key.lower() == "svr":
            k = params.get("kernel")
            if k != "poly":
                params.pop("degree", None)
            if k not in {"poly", "sigmoid"}:
                params.pop("coef0", None)
        est: RegressorMixin = build_estimator(model_key, params)
        # Nessuna validation esterna -> opzionale cross-validation
        if X_val is None or y_val is None:
            use_cv = bool((cv_config or {}).get("enabled", False))
            if use_cv:
                kind = str((cv_config or {}).get("kind", "timeseries")).lower()
                n_splits = int((cv_config or {}).get("n_splits", 5))
                shuffle = bool((cv_config or {}).get("shuffle", False))
                if kind == "kfold":
                    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
                else:
                    splitter = TimeSeriesSplit(n_splits=n_splits)
                scores: List[float] = []
                for tr_idx, va_idx in splitter.split(X_train):
                    # Use row-based indexing for pandas objects to avoid interpreting indices as column labels
                    if hasattr(X_train, 'iloc'):
                        X_tr = X_train.iloc[tr_idx]
                        X_va = X_train.iloc[va_idx]
                        y_tr = y_train.iloc[tr_idx] if hasattr(y_train, 'iloc') else y_train[tr_idx]
                        y_va = y_train.iloc[va_idx] if hasattr(y_train, 'iloc') else y_train[va_idx]
                    else:
                        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                    mk = model_key.lower()
                    try:
                        if mk == XGBOOST_KEY:
                            est.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, early_stopping_rounds=50)
                        elif mk == LIGHTGBM_KEY:
                            metric_map = {
                                "r2": None,  # not a native built-in; let it default
                                "neg_mean_squared_error": "l2",
                                "neg_root_mean_squared_error": "rmse",
                                "neg_mean_absolute_error": "mae",
                                "neg_mean_absolute_percentage_error": "mape",
                            }
                            eval_metric = metric_map.get(primary_metric.lower(), None)
                            fit_kwargs: Dict[str, Any] = {"eval_set": [(X_va.values if hasattr(X_va, 'values') else X_va, y_va.values if hasattr(y_va, 'values') else y_va)]}
                            if eval_metric is not None:
                                fit_kwargs["eval_metric"] = eval_metric
                            try:
                                import lightgbm as lgb  # type: ignore
                                fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
                            except Exception:
                                fit_kwargs["early_stopping_rounds"] = 50
                            est.fit(X_tr.values if hasattr(X_tr, 'values') else X_tr, y_tr.values if hasattr(y_tr, 'values') else y_tr, **fit_kwargs)
                        elif mk == CATBOOST_KEY:
                            if cat_features is not None:
                                est.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_va, y_va), verbose=False, early_stopping_rounds=50)
                            else:
                                est.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False, early_stopping_rounds=50)
                        else:
                            if mk == CATBOOST_KEY and cat_features is not None:
                                est.fit(X_tr, y_tr, cat_features=cat_features, verbose=False)
                            else:
                                est.fit(X_tr, y_tr)
                    except Exception:
                        if mk == CATBOOST_KEY and cat_features is not None:
                            est.fit(X_tr, y_tr, cat_features=cat_features, verbose=False)
                        else:
                            est.fit(X_tr, y_tr)
                    y_pred = est.predict(X_va)
                    scores.append(select_primary_value(primary_metric, y_va, y_pred))
                final_score = float(np.mean(scores)) if scores else -np.inf
                # Log CV trial (simplified: only aggregate score)
                if wandb_manager is not None:
                    try:
                        wandb_manager.log({
                            f"tuning/{model_key}/trial_number": trial.number,
                            f"tuning/{model_key}/{primary_metric}_cv": final_score,
                        })  # No step parameter - auto-increment by W&B
                    except Exception:
                        pass
                return final_score
            # Use temporal split instead of random split to avoid data leakage
            # Maintain chronological order for time-series data
            split_point = int(len(X_train) * tuning_split_fraction)
            if hasattr(X_train, 'iloc'):  # DataFrame
                X_tr = X_train.iloc[:split_point]
                X_va = X_train.iloc[split_point:]
                y_tr = y_train.iloc[:split_point] if hasattr(y_train, 'iloc') else y_train[:split_point]
                y_va = y_train.iloc[split_point:] if hasattr(y_train, 'iloc') else y_train[split_point:]
            else:  # numpy array
                X_tr, X_va = X_train[:split_point], X_train[split_point:]
                y_tr, y_va = y_train[:split_point], y_train[split_point:]
            # Fit con eventuale early stopping non applicabile senza validation esterna coerente; esegue fit semplice
            if model_key.lower() == CATBOOST_KEY and cat_features is not None:
                est.fit(X_tr, y_tr, cat_features=cat_features, verbose=False)
            else:
                est.fit(X_tr, y_tr)
            y_pred = est.predict(X_va)
            # Log trial to W&B
            _log_trial_to_wandb(trial.number, y_va, y_pred, params)
            return select_primary_value(primary_metric, y_va, y_pred)
        else:
            # Con validation esterna: abilita early stopping dove supportato
            mk = model_key.lower()
            try:
                if mk == XGBOOST_KEY:
                    est.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=50)
                elif mk == LIGHTGBM_KEY:
                    # Map primary_metric to LGBM-native metric when possible
                    metric_map = {
                        "r2": None,  # not a native built-in; let it default
                        "neg_mean_squared_error": "l2",
                        "neg_root_mean_squared_error": "rmse",
                        "neg_mean_absolute_error": "mae",
                        "neg_mean_absolute_percentage_error": "mape",
                    }
                    eval_metric = metric_map.get(primary_metric.lower(), None)
                    fit_kwargs = {"eval_set": [(X_val.values if hasattr(X_val, 'values') else X_val, y_val.values if hasattr(y_val, 'values') else y_val)]}
                    if eval_metric is not None:
                        fit_kwargs["eval_metric"] = eval_metric
                    # callbacks per early stopping silenzioso
                    try:
                        import lightgbm as lgb  # type: ignore
                        fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
                    except Exception:
                        fit_kwargs["early_stopping_rounds"] = 50
                    est.fit(X_train.values if hasattr(X_train, 'values') else X_train, y_train.values if hasattr(y_train, 'values') else y_train, **fit_kwargs)
                elif mk == CATBOOST_KEY:
                    if cat_features is not None:
                        est.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=50)
                    else:
                        est.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=50)
                else:
                    if mk == CATBOOST_KEY and cat_features is not None:
                        est.fit(X_train, y_train, cat_features=cat_features, verbose=False)
                    else:
                        est.fit(X_train, y_train)
            except Exception:
                # Fallback robusto
                if mk == CATBOOST_KEY and cat_features is not None:
                    est.fit(X_train, y_train, cat_features=cat_features, verbose=False)
                else:
                    est.fit(X_train, y_train)
            y_pred = est.predict(X_val)
            # Log trial to W&B
            _log_trial_to_wandb(trial.number, y_val, y_pred, params)
            return select_primary_value(primary_metric, y_val, y_pred)

    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return TuningResult(best_params=study.best_params, best_value=study.best_value, study=study)