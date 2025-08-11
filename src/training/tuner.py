from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import optuna
import optunahub
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split

from .model_zoo import build_estimator
from .metrics import select_primary_value


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
) -> TuningResult:
    direction = "maximize"

    if sampler_name == "auto":
        module = optunahub.load_module(package="samplers/auto_sampler")
        sampler = module.AutoSampler(seed=seed)
    elif sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)

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
        if X_val is None or y_val is None:
            X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.2, random_state=seed, shuffle=False)
            # Fit con eventuale early stopping non applicabile senza validation esterna coerente; esegue fit semplice
            if model_key.lower() == "catboost" and cat_features is not None:
                est.fit(X_tr, y_tr, cat_features=cat_features, verbose=False)
            else:
                est.fit(X_tr, y_tr)
            y_pred = est.predict(X_va)
            return select_primary_value(primary_metric, y_va, y_pred)
        else:
            # Con validation esterna: abilita early stopping dove supportato
            mk = model_key.lower()
            try:
                if mk == "xgboost":
                    est.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=50)
                elif mk == "lightgbm":
                    # Mappa metrica primaria a metrica LGBM
                    metric_map = {
                        "r2": "r2",
                        "neg_mean_squared_error": "l2",
                        "neg_root_mean_squared_error": "rmse",
                        "neg_mean_absolute_error": "mae",
                        "neg_mean_absolute_percentage_error": "mape",
                    }
                    eval_metric = metric_map.get(primary_metric.lower(), None)
                    fit_kwargs = {"eval_set": [(X_val, y_val)]}
                    if eval_metric is not None:
                        fit_kwargs["eval_metric"] = eval_metric
                    # callbacks per early stopping silenzioso
                    try:
                        import lightgbm as lgb  # type: ignore
                        fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
                    except Exception:
                        fit_kwargs["early_stopping_rounds"] = 50
                    est.fit(X_train, y_train, **fit_kwargs)
                elif mk == "catboost":
                    if cat_features is not None:
                        est.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=50)
                    else:
                        est.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=50)
                else:
                    if mk == "catboost" and cat_features is not None:
                        est.fit(X_train, y_train, cat_features=cat_features, verbose=False)
                    else:
                        est.fit(X_train, y_train)
            except Exception:
                # Fallback robusto
                if mk == "catboost" and cat_features is not None:
                    est.fit(X_train, y_train, cat_features=cat_features, verbose=False)
                else:
                    est.fit(X_train, y_train)
            y_pred = est.predict(X_val)
            return select_primary_value(primary_metric, y_val, y_pred)

    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return TuningResult(best_params=study.best_params, best_value=study.best_value, study=study)