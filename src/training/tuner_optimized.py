from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import logging

import numpy as np
import optuna
import optunahub
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from .model_zoo import build_estimator
from .metrics import select_primary_value

logger = logging.getLogger(__name__)

# Parametri baseline ottimali basati sui risultati
BASELINE_PARAMS = {
    "lightgbm": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": -1,
        "num_leaves": 31,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "min_child_samples": 20,
        "min_split_gain": 0.0
    },
    "xgboost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0
    },
    "catboost": {
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 1.0,
        "border_count": 32,
        "random_strength": 1.0,
        "rsm": 1.0
    },
    "hgbt": {
        "learning_rate": 0.1,
        "max_depth": None,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.0,
        "max_bins": 255,
        "min_samples_leaf": 20
    },
    "gbr": {
        "learning_rate": 0.1,
        "n_estimators": 100,
        "max_depth": 3,
        "subsample": 1.0,
        "min_samples_leaf": 1,
        "max_features": None,
        "min_samples_split": 2,
        "min_impurity_decrease": 0.0
    },
    "rf": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True
    }
}


@dataclass
class TuningResult:
    best_params: Dict[str, Any]
    best_value: float
    study: optuna.Study
    overfit_scores: Dict[str, float]  # Aggiunto per tracciare overfitting


def calculate_overfit_penalty(train_score: float, val_score: float, penalty_weight: float = 0.15) -> float:
    """Calcola la penalità per overfitting basata sulla differenza tra train e validation score."""
    gap = abs(train_score - val_score)
    # Normalizza il gap (assumendo score tra 0 e 1 per R2, o simile range per altre metriche)
    normalized_gap = min(gap, 1.0)  # Cap a 1.0 per evitare penalità eccessive
    return penalty_weight * normalized_gap


def apply_parameter_constraints(model_key: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Applica vincoli logici ai parametri per prevenire combinazioni che portano a overfitting."""
    mk = model_key.lower()
    
    # Vincoli per LightGBM
    if mk == "lightgbm":
        lr = params.get("learning_rate", 0.1)
        if lr > 0.1:
            # Con learning rate alto, limita il numero di estimatori
            params["n_estimators"] = min(params.get("n_estimators", 100), 300)
        # num_leaves deve essere <= 2^max_depth
        if params.get("max_depth", -1) > 0:
            max_leaves = 2 ** params["max_depth"]
            params["num_leaves"] = min(params.get("num_leaves", 31), max_leaves - 1)
    
    # Vincoli per XGBoost
    elif mk == "xgboost":
        lr = params.get("learning_rate", 0.1)
        if lr > 0.1:
            params["n_estimators"] = min(params.get("n_estimators", 100), 300)
        # Se la regolarizzazione è bassa, limita la profondità
        if params.get("reg_alpha", 0) < 0.1 and params.get("reg_lambda", 1.0) < 0.1:
            params["max_depth"] = min(params.get("max_depth", 6), 6)
    
    # Vincoli per CatBoost
    elif mk == "catboost":
        # Con depth alto, aumenta la regolarizzazione
        if params.get("depth", 6) > 6:
            params["l2_leaf_reg"] = max(params.get("l2_leaf_reg", 3.0), 5.0)
    
    # Vincoli per Random Forest
    elif mk == "rf":
        # Se max_depth è None o molto alto, aumenta min_samples_leaf
        if params.get("max_depth") is None or params.get("max_depth", 0) > 15:
            params["min_samples_leaf"] = max(params.get("min_samples_leaf", 1), 5)
    
    # Vincoli per GBR
    elif mk == "gbr":
        # Con molti estimatori, riduci learning rate
        if params.get("n_estimators", 100) > 500:
            params["learning_rate"] = min(params.get("learning_rate", 0.1), 0.05)
    
    return params


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
    tuning_split_fraction: float = 0.8,
    overfitting_penalty_config: Optional[Dict[str, Any]] = None,
    cv_during_tuning_config: Optional[Dict[str, Any]] = None,
) -> TuningResult:
    """
    Ottimizza i parametri del modello con controlli avanzati per prevenire overfitting.
    
    Args:
        model_key: Chiave del modello
        X_train, y_train: Dati di training
        X_val, y_val: Dati di validazione (opzionali)
        primary_metric: Metrica primaria per ottimizzazione
        n_trials: Numero di trial Optuna
        timeout: Timeout in secondi
        sampler_name: Nome del sampler Optuna
        seed: Random seed
        base_params: Parametri base del modello
        search_space: Spazio di ricerca dei parametri
        cat_features: Indici delle feature categoriche
        cv_config: Configurazione cross-validation
        tuning_split_fraction: Frazione per split temporale
        overfitting_penalty_config: Configurazione penalità overfitting
        cv_during_tuning_config: Configurazione CV durante tuning
    """
    direction = "maximize"
    
    # Configurazione overfitting penalty
    overfitting_config = overfitting_penalty_config or {}
    use_overfitting_penalty = overfitting_config.get("enabled", False)
    penalty_weight = overfitting_config.get("weight", 0.15)
    max_acceptable_gap = overfitting_config.get("max_gap", 0.10)
    
    # Configurazione CV durante tuning
    cv_tuning_config = cv_during_tuning_config or {}
    use_cv_during_tuning = cv_tuning_config.get("enabled", False)
    cv_n_splits = cv_tuning_config.get("n_splits", 3)
    cv_strategy = cv_tuning_config.get("strategy", "timeseries")

    # Setup sampler
    if sampler_name == "auto":
        try:
            module = optunahub.load_module(package="samplers/auto_sampler")
            sampler = module.AutoSampler(seed=seed)
        except Exception:
            sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
    elif sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
    
    # Setup pruner per fermare trial non promettenti
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=10
    )

    def objective(trial: optuna.Trial) -> float:
        # Suggerisci parametri
        params = _apply_suggestions(trial, search_space or {}, base_params or {})
        
        # Applica vincoli ai parametri
        params = apply_parameter_constraints(model_key, params)
        
        # Guardia SVR
        if model_key.lower() == "svr":
            k = params.get("kernel")
            if k != "poly":
                params.pop("degree", None)
            if k not in {"poly", "sigmoid"}:
                params.pop("coef0", None)
        
        # Costruisci il modello
        est: RegressorMixin = build_estimator(model_key, params)
        
        # Se abbiamo validation set esterno E CV durante tuning è abilitato
        if X_val is not None and y_val is not None and use_cv_during_tuning:
            # Usa cross-validation sul training set per valutazione più robusta
            if cv_strategy == "kfold":
                splitter = KFold(n_splits=cv_n_splits, shuffle=True, random_state=seed)
            else:
                splitter = TimeSeriesSplit(n_splits=cv_n_splits)
            
            cv_scores = []
            train_scores = []
            
            for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X_train)):
                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                
                # Fit del modello con early stopping dove supportato
                mk = model_key.lower()
                try:
                    fit_model_with_early_stopping(est, mk, X_tr, y_tr, X_va, y_va, cat_features)
                except Exception as e:
                    logger.warning(f"Early stopping fallito per {mk}: {e}")
                    if mk == "catboost" and cat_features is not None:
                        est.fit(X_tr, y_tr, cat_features=cat_features, verbose=False)
                    else:
                        est.fit(X_tr, y_tr)
                
                # Calcola score su validation fold
                y_pred_va = est.predict(X_va)
                val_score = select_primary_value(primary_metric, y_va, y_pred_va)
                cv_scores.append(val_score)
                
                # Calcola score su training per monitorare overfitting
                if use_overfitting_penalty:
                    y_pred_tr = est.predict(X_tr)
                    train_score = select_primary_value(primary_metric, y_tr, y_pred_tr)
                    train_scores.append(train_score)
                
                # Report per pruning
                intermediate_value = np.mean(cv_scores)
                trial.report(intermediate_value, fold_idx)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Score finale
            final_val_score = np.mean(cv_scores)
            
            # Applica penalità overfitting se abilitata
            if use_overfitting_penalty and train_scores:
                avg_train_score = np.mean(train_scores)
                penalty = calculate_overfit_penalty(avg_train_score, final_val_score, penalty_weight)
                
                # Se il gap è troppo grande, penalizza pesantemente
                gap = abs(avg_train_score - final_val_score)
                if gap > max_acceptable_gap:
                    penalty *= 2.0
                
                adjusted_score = final_val_score - penalty
                
                # Salva info overfitting nel trial
                trial.set_user_attr("train_score", avg_train_score)
                trial.set_user_attr("val_score", final_val_score)
                trial.set_user_attr("overfit_gap", gap)
                trial.set_user_attr("overfit_penalty", penalty)
                
                return adjusted_score
            else:
                return final_val_score
                
        # Comportamento originale quando non c'è CV durante tuning
        elif X_val is None or y_val is None:
            # Codice originale per backward compatibility
            use_cv = bool((cv_config or {}).get("enabled", False))
            if use_cv:
                # CV standard come nel codice originale
                kind = str((cv_config or {}).get("kind", "timeseries")).lower()
                n_splits = int((cv_config or {}).get("n_splits", 5))
                shuffle = bool((cv_config or {}).get("shuffle", False))
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
                        fit_model_with_early_stopping(est, mk, X_tr, y_tr, X_va, y_va, cat_features)
                    except Exception:
                        if mk == "catboost" and cat_features is not None:
                            est.fit(X_tr, y_tr, cat_features=cat_features, verbose=False)
                        else:
                            est.fit(X_tr, y_tr)
                    y_pred = est.predict(X_va)
                    scores.append(select_primary_value(primary_metric, y_va, y_pred))
                return float(np.mean(scores)) if scores else -np.inf
            
            # Temporal split
            split_point = int(len(X_train) * tuning_split_fraction)
            if hasattr(X_train, 'iloc'):
                X_tr = X_train.iloc[:split_point]
                X_va = X_train.iloc[split_point:]
                y_tr = y_train.iloc[:split_point] if hasattr(y_train, 'iloc') else y_train[:split_point]
                y_va = y_train.iloc[split_point:] if hasattr(y_train, 'iloc') else y_train[split_point:]
            else:
                X_tr, X_va = X_train[:split_point], X_train[split_point:]
                y_tr, y_va = y_train[:split_point], y_train[split_point:]
            
            if model_key.lower() == "catboost" and cat_features is not None:
                est.fit(X_tr, y_tr, cat_features=cat_features, verbose=False)
            else:
                est.fit(X_tr, y_tr)
            y_pred = est.predict(X_va)
            return select_primary_value(primary_metric, y_va, y_pred)
        else:
            # Con validation set esterno ma senza CV durante tuning
            mk = model_key.lower()
            try:
                fit_model_with_early_stopping(est, mk, X_train, y_train, X_val, y_val, cat_features)
            except Exception:
                if mk == "catboost" and cat_features is not None:
                    est.fit(X_train, y_train, cat_features=cat_features, verbose=False)
                else:
                    est.fit(X_train, y_train)
            y_pred = est.predict(X_val)
            return select_primary_value(primary_metric, y_val, y_pred)

    # Crea lo studio con pruner
    study = optuna.create_study(
        direction=direction, 
        sampler=sampler,
        pruner=pruner
    )
    
    # Aggiungi parametri baseline come primo trial se disponibili
    model_key_lower = model_key.lower()
    if model_key_lower in BASELINE_PARAMS:
        baseline = BASELINE_PARAMS[model_key_lower].copy()
        # Filtra solo i parametri che sono nello spazio di ricerca
        if search_space:
            baseline_filtered = {k: v for k, v in baseline.items() if k in search_space}
            if baseline_filtered:
                study.enqueue_trial(baseline_filtered)
                logger.info(f"Aggiunto trial con parametri baseline per {model_key}")
    
    # Ottimizza
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Estrai informazioni sull'overfitting dal miglior trial
    overfit_scores = {}
    if study.best_trial.user_attrs:
        overfit_scores = {
            "train_score": study.best_trial.user_attrs.get("train_score"),
            "val_score": study.best_trial.user_attrs.get("val_score"),
            "overfit_gap": study.best_trial.user_attrs.get("overfit_gap"),
            "overfit_penalty": study.best_trial.user_attrs.get("overfit_penalty")
        }

    return TuningResult(
        best_params=study.best_params, 
        best_value=study.best_value, 
        study=study,
        overfit_scores=overfit_scores
    )


def fit_model_with_early_stopping(
    est: RegressorMixin,
    model_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cat_features: Optional[List[int]] = None,
    early_stopping_rounds: int = 100
) -> None:
    """Helper per fit del modello con early stopping dove supportato."""
    mk = model_key.lower()
    
    if mk == "xgboost":
        est.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, 
                early_stopping_rounds=early_stopping_rounds)
    elif mk == "lightgbm":
        # LightGBM richiede conversione a .values per DataFrame
        X_tr = X_train.values if hasattr(X_train, 'values') else X_train
        y_tr = y_train.values if hasattr(y_train, 'values') else y_train
        X_va = X_val.values if hasattr(X_val, 'values') else X_val
        y_va = y_val.values if hasattr(y_val, 'values') else y_val
        
        import lightgbm as lgb
        fit_kwargs = {
            "eval_set": [(X_va, y_va)],
            "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        }
        est.fit(X_tr, y_tr, **fit_kwargs)
    elif mk == "catboost":
        if cat_features is not None:
            est.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), 
                   verbose=False, early_stopping_rounds=early_stopping_rounds)
        else:
            est.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, 
                   early_stopping_rounds=early_stopping_rounds)
    else:
        # Modello senza early stopping
        if mk == "catboost" and cat_features is not None:
            est.fit(X_train, y_train, cat_features=cat_features, verbose=False)
        else:
            est.fit(X_train, y_train)