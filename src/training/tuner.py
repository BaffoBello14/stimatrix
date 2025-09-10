from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import optuna
import optunahub
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold, TimeSeriesSplit

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
    cv_config: Optional[Dict[str, Any]] = None,
    tuning_split_fraction: float = 0.8,  # fraction for temporal split (consistent with preprocessing)
    tuning_options: Optional[Dict[str, Any]] = None,
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

    # --- Build pruner
    pruner_name = str(((tuning_options or {}).get("pruner", "median") or "").lower())
    if pruner_name in {"median", "med": "median"}:
        pruner = optuna.pruners.MedianPruner()
    elif pruner_name in {"halving", "successive_halving", "sha"}:
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    elif pruner_name in {"hyperband"}:
        pruner = optuna.pruners.HyperbandPruner()
    elif pruner_name in {"none", "off", "disable"}:
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = optuna.pruners.MedianPruner()

    # --- Optionally narrow the search space around baseline
    def _anchored_space(space: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
        if not bool((tuning_options or {}).get("anchor_to_baseline", False)):
            return space
        anchored: Dict[str, Any] = {}
        # defaults
        anchor_cfg = (tuning_options or {}).get("anchor", {}) or {}
        float_rel_span = float(anchor_cfg.get("float_rel_span", 0.5))  # ±50% around base for linear floats
        int_rel_span = float(anchor_cfg.get("int_rel_span", 0.3))      # ±30% around base for ints
        for name, spec in (space or {}).items():
            bval = base.get(name, None)
            if bval is None:
                anchored[name] = spec
                continue
            t = str(spec.get("type", "")).lower()
            lo = spec.get("low")
            hi = spec.get("high")
            if t == "float" and lo is not None and hi is not None:
                lo = float(lo)
                hi = float(hi)
                if bool(spec.get("log", False)):
                    # multiplicative window
                    low_new = max(lo, bval * max(1e-8, 1 - float_rel_span))
                    high_new = min(hi, bval * (1 + float_rel_span))
                else:
                    # additive window by relative span
                    delta = max(1e-12, abs(bval) * float_rel_span)
                    low_new = max(lo, float(bval) - delta)
                    high_new = min(hi, float(bval) + delta)
                if low_new >= high_new:  # fallback to original
                    anchored[name] = spec
                else:
                    s = dict(spec)
                    s["low"] = low_new
                    s["high"] = high_new
                    anchored[name] = s
            elif t == "int" and lo is not None and hi is not None:
                lo = int(lo)
                hi = int(hi)
                window = max(2, int(round(abs(int(bval)) * int_rel_span)))
                low_new = max(lo, int(bval) - window)
                high_new = min(hi, int(bval) + window)
                if low_new >= high_new:
                    anchored[name] = spec
                else:
                    s = dict(spec)
                    s["low"] = low_new
                    s["high"] = high_new
                    anchored[name] = s
            else:
                anchored[name] = spec
        return anchored

    eff_space = _anchored_space(search_space or {}, base_params or {})

    def objective(trial: optuna.Trial) -> float:
        params = _apply_suggestions(trial, eff_space or {}, base_params or {})
        # Guardia SVR: degree ha senso solo con kernel 'poly'; coef0 solo per 'poly' o 'sigmoid'
        if model_key.lower() == "svr":
            k = params.get("kernel")
            if k != "poly":
                params.pop("degree", None)
            if k not in {"poly", "sigmoid"}:
                params.pop("coef0", None)
        est: RegressorMixin = build_estimator(model_key, params)
        # Forza CV durante tuning se configurato, anche se esiste una validation esterna
        cv_always_cfg = (tuning_options or {}).get("cv_always", {}) or {}
        force_cv = bool(cv_always_cfg.get("enabled", False))
        # Nessuna validation esterna -> opzionale cross-validation
        if force_cv or X_val is None or y_val is None:
            use_cv = bool((cv_config or {}).get("enabled", False))
            # Se force_cv, usa cv_always_cfg come prioritaria
            if force_cv:
                use_cv = True
            active_cv_cfg = dict(cv_config or {})
            if force_cv:
                active_cv_cfg.update(cv_always_cfg)
            if use_cv:
                kind = str((active_cv_cfg or {}).get("kind", "timeseries")).lower()
                n_splits = int((active_cv_cfg or {}).get("n_splits", 5))
                shuffle = bool((active_cv_cfg or {}).get("shuffle", False))
                if kind == "kfold":
                    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
                else:
                    splitter = TimeSeriesSplit(n_splits=n_splits)
                scores: List[float] = []
                for tr_idx, va_idx in splitter.split(X_train):
                    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                    mk = model_key.lower()
                    try:
                        if mk == "xgboost":
                            # Prefer RMSE as eval metric for regression
                            est.fit(
                                X_tr,
                                y_tr,
                                eval_set=[(X_va, y_va)],
                                eval_metric="rmse",
                                verbose=False,
                                early_stopping_rounds=200,
                            )
                        elif mk == "lightgbm":
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
                                fit_kwargs["callbacks"] = [lgb.early_stopping(200, verbose=False)]
                            except Exception:
                                fit_kwargs["early_stopping_rounds"] = 200
                            est.fit(X_tr.values if hasattr(X_tr, 'values') else X_tr, y_tr.values if hasattr(y_tr, 'values') else y_tr, **fit_kwargs)
                        elif mk == "catboost":
                            if cat_features is not None:
                                est.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_va, y_va), verbose=False, early_stopping_rounds=200)
                            else:
                                est.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False, early_stopping_rounds=200)
                        else:
                            if mk == "catboost" and cat_features is not None:
                                est.fit(X_tr, y_tr, cat_features=cat_features, verbose=False)
                            else:
                                est.fit(X_tr, y_tr)
                    except Exception:
                        if mk == "catboost" and cat_features is not None:
                            est.fit(X_tr, y_tr, cat_features=cat_features, verbose=False)
                        else:
                            est.fit(X_tr, y_tr)
                    y_pred = est.predict(X_va)
                    scores.append(select_primary_value(primary_metric, y_va, y_pred))
                return float(np.mean(scores)) if scores else -np.inf
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
            # Fit con early stopping dove supportato usando lo split temporale come validazione
            mk = model_key.lower()
            try:
                if mk == "xgboost":
                    est.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_va, y_va)],
                        eval_metric="rmse",
                        verbose=False,
                        early_stopping_rounds=200,
                    )
                elif mk == "lightgbm":
                    metric_map = {
                        "r2": None,
                        "neg_mean_squared_error": "l2",
                        "neg_root_mean_squared_error": "rmse",
                        "neg_mean_absolute_error": "mae",
                        "neg_mean_absolute_percentage_error": "mape",
                    }
                    eval_metric = metric_map.get(primary_metric.lower(), None)
                    fit_kwargs = {
                        "eval_set": [(
                            X_va.values if hasattr(X_va, 'values') else X_va,
                            y_va.values if hasattr(y_va, 'values') else y_va,
                        )]
                    }
                    if eval_metric is not None:
                        fit_kwargs["eval_metric"] = eval_metric
                    try:
                        import lightgbm as lgb  # type: ignore
                        fit_kwargs["callbacks"] = [lgb.early_stopping(200, verbose=False)]
                    except Exception:
                        fit_kwargs["early_stopping_rounds"] = 200
                    est.fit(
                        X_tr.values if hasattr(X_tr, 'values') else X_tr,
                        y_tr.values if hasattr(y_tr, 'values') else y_tr,
                        **fit_kwargs,
                    )
                elif mk == "catboost":
                    if cat_features is not None:
                        est.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_va, y_va), verbose=False, early_stopping_rounds=200)
                    else:
                        est.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False, early_stopping_rounds=200)
                else:
                    if mk == "catboost" and cat_features is not None:
                        est.fit(X_tr, y_tr, cat_features=cat_features, verbose=False)
                    else:
                        est.fit(X_tr, y_tr)
            except Exception:
                # Fallback robusto
                if mk == "catboost" and cat_features is not None:
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
                    est.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric="rmse",
                        verbose=False,
                        early_stopping_rounds=200,
                    )
                elif mk == "lightgbm":
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
                        fit_kwargs["callbacks"] = [lgb.early_stopping(200, verbose=False)]
                    except Exception:
                        fit_kwargs["early_stopping_rounds"] = 200
                    est.fit(X_train.values if hasattr(X_train, 'values') else X_train, y_train.values if hasattr(y_train, 'values') else y_train, **fit_kwargs)
                elif mk == "catboost":
                    if cat_features is not None:
                        est.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=200)
                    else:
                        est.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=200)
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

    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    # Enqueue baseline trial to guarantee evaluation of base params
    if bool((tuning_options or {}).get("enqueue_baseline_trial", False)) and (base_params or {}):
        try:
            study.enqueue_trial(base_params)
        except Exception:
            pass
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return TuningResult(best_params=study.best_params, best_value=study.best_value, study=study)