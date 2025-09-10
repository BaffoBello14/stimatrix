"""
Utilità per gestire early stopping correttamente con validation set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


def prepare_early_stopping_params(
    model_key: str, 
    base_params: Dict[str, Any], 
    fit_params: Dict[str, Any],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Prepara parametri per early stopping corretto.
    
    Args:
        model_key: Nome del modello (lightgbm, xgboost, catboost)
        base_params: Parametri base del modello
        fit_params: Parametri di fit
        X_val: Validation features (opzionale)
        y_val: Validation target (opzionale)
    
    Returns:
        Tuple di (base_params_cleaned, fit_params_with_eval)
    """
    
    # Copia per non modificare originali
    clean_base_params = base_params.copy()
    enhanced_fit_params = fit_params.copy()
    
    # Se non abbiamo validation set, rimuovi early stopping
    if X_val is None or y_val is None:
        logger.warning(f"[{model_key}] Nessun validation set disponibile - rimuovo early stopping")
        
        # Rimuovi parametri early stopping da base_params
        early_stopping_keys = [
            'early_stopping_rounds', 'early_stopping_round', 
            'use_best_model', 'eval_metric'
        ]
        for key in early_stopping_keys:
            clean_base_params.pop(key, None)
            enhanced_fit_params.pop(key, None)
        
        # Riduci n_estimators/iterations per compensare
        if 'n_estimators' in clean_base_params:
            original = clean_base_params['n_estimators']
            clean_base_params['n_estimators'] = min(500, original)
            logger.info(f"[{model_key}] Ridotto n_estimators: {original} → {clean_base_params['n_estimators']}")
        
        if 'iterations' in clean_base_params:
            original = clean_base_params['iterations']
            clean_base_params['iterations'] = min(500, original)
            logger.info(f"[{model_key}] Ridotto iterations: {original} → {clean_base_params['iterations']}")
        
        return clean_base_params, enhanced_fit_params
    
    # Abbiamo validation set - configura early stopping corretto
    logger.info(f"[{model_key}] Configurazione early stopping con validation set")
    
    if model_key.lower() == 'lightgbm':
        return _prepare_lightgbm_early_stopping(
            clean_base_params, enhanced_fit_params, X_val, y_val
        )
    elif model_key.lower() == 'xgboost':
        return _prepare_xgboost_early_stopping(
            clean_base_params, enhanced_fit_params, X_val, y_val
        )
    elif model_key.lower() == 'catboost':
        return _prepare_catboost_early_stopping(
            clean_base_params, enhanced_fit_params, X_val, y_val
        )
    else:
        # Altri modelli - rimuovi early stopping
        early_stopping_keys = ['early_stopping_rounds', 'use_best_model', 'eval_metric']
        for key in early_stopping_keys:
            clean_base_params.pop(key, None)
            enhanced_fit_params.pop(key, None)
    
    return clean_base_params, enhanced_fit_params


def _prepare_lightgbm_early_stopping(
    base_params: Dict[str, Any], 
    fit_params: Dict[str, Any],
    X_val: pd.DataFrame, 
    y_val: pd.Series
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Configura early stopping per LightGBM."""
    
    # Rimuovi da base_params (vanno in fit_params)
    early_stopping_rounds = base_params.pop('early_stopping_rounds', 50)
    eval_metric = base_params.pop('eval_metric', 'rmse')
    
    # Aggiungi a fit_params
    fit_params['eval_set'] = [(X_val.values, y_val.values)]
    fit_params['eval_names'] = ['validation']
    fit_params['eval_metric'] = eval_metric
    fit_params['callbacks'] = [
        # LightGBM callback per early stopping
        __import__('lightgbm').early_stopping(
            stopping_rounds=early_stopping_rounds,
            verbose=False
        )
    ]
    
    logger.info(f"[lightgbm] Early stopping configurato: {early_stopping_rounds} rounds, metric={eval_metric}")
    
    return base_params, fit_params


def _prepare_xgboost_early_stopping(
    base_params: Dict[str, Any], 
    fit_params: Dict[str, Any],
    X_val: pd.DataFrame, 
    y_val: pd.Series
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Configura early stopping per XGBoost."""
    
    # Rimuovi da base_params
    early_stopping_rounds = base_params.pop('early_stopping_rounds', 50)
    eval_metric = base_params.pop('eval_metric', 'rmse')
    
    # Aggiungi a fit_params
    fit_params['eval_set'] = [(X_val.values, y_val.values)]
    fit_params['early_stopping_rounds'] = early_stopping_rounds
    fit_params['verbose'] = False
    
    # XGBoost eval_metric va in base_params
    base_params['eval_metric'] = eval_metric
    
    logger.info(f"[xgboost] Early stopping configurato: {early_stopping_rounds} rounds, metric={eval_metric}")
    
    return base_params, fit_params


def _prepare_catboost_early_stopping(
    base_params: Dict[str, Any], 
    fit_params: Dict[str, Any],
    X_val: pd.DataFrame, 
    y_val: pd.Series
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Configura early stopping per CatBoost."""
    
    # Per CatBoost, early stopping va in fit_params
    early_stopping_rounds = base_params.pop('early_stopping_rounds', 50)
    use_best_model = base_params.pop('use_best_model', True)
    
    # Aggiungi a fit_params
    fit_params['eval_set'] = (X_val.values, y_val.values)
    fit_params['early_stopping_rounds'] = early_stopping_rounds
    fit_params['use_best_model'] = use_best_model
    fit_params['verbose'] = False
    
    logger.info(f"[catboost] Early stopping configurato: {early_stopping_rounds} rounds, use_best_model={use_best_model}")
    
    return base_params, fit_params


def should_use_early_stopping(config: Dict[str, Any]) -> bool:
    """Determina se usare early stopping basato sulla configurazione."""
    
    early_stopping_config = config.get("training", {}).get("early_stopping", {})
    return bool(early_stopping_config.get("enabled", False))


def log_early_stopping_results(model_key: str, estimator: Any) -> Dict[str, Any]:
    """Log risultati early stopping se disponibili."""
    
    results = {}
    
    try:
        if hasattr(estimator, 'best_iteration_'):
            results['best_iteration'] = int(estimator.best_iteration_)
            results['stopped_early'] = True
            logger.info(f"[{model_key}] Early stopping attivato alla iterazione {results['best_iteration']}")
        
        elif hasattr(estimator, 'best_iteration'):
            results['best_iteration'] = int(estimator.best_iteration)
            results['stopped_early'] = True
            logger.info(f"[{model_key}] Early stopping attivato alla iterazione {results['best_iteration']}")
        
        elif hasattr(estimator, 'tree_count_'):
            results['final_trees'] = int(estimator.tree_count_)
            results['stopped_early'] = False
            logger.info(f"[{model_key}] Training completato con {results['final_trees']} alberi")
        
        else:
            results['stopped_early'] = False
            logger.info(f"[{model_key}] Training completato senza early stopping")
    
    except Exception as e:
        logger.warning(f"[{model_key}] Errore nel recupero info early stopping: {e}")
        results['stopped_early'] = False
    
    return results