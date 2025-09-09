from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional

from sklearn.ensemble import VotingRegressor, StackingRegressor

from .model_zoo import build_estimator


def build_voting(models_best: List[Tuple[str, Dict[str, Any]]], tune_weights: bool = True, n_jobs: int = -1, weights: Optional[List[float]] = None) -> VotingRegressor:
    estimators = [(f"{k}", build_estimator(k, p)) for k, p in models_best]
    if weights is not None:
        return VotingRegressor(estimators=estimators, weights=weights, n_jobs=n_jobs)
    if tune_weights:
        # simple heuristic weights proportional to rank (can be tuned with optuna externally if desired)
        n = len(estimators)
        weights = list(reversed(range(1, n + 1)))
        return VotingRegressor(estimators=estimators, weights=weights, n_jobs=n_jobs)
    return VotingRegressor(estimators=estimators, n_jobs=n_jobs)


def build_stacking(models_best: List[Tuple[str, Dict[str, Any]]], final_estimator_key: str, cv_folds: int = 5, n_jobs: int = -1) -> StackingRegressor:
    estimators = [(f"{k}", build_estimator(k, p)) for k, p in models_best]
    if final_estimator_key.lower() in {"ridge", "linear", "lasso", "elasticnet"}:
        fe = build_estimator(final_estimator_key, {})
    elif final_estimator_key.lower() in {"lightgbm", "lgbm"}:
        fe = build_estimator("lightgbm", {})
    else:
        fe = build_estimator("ridge", {})
    return StackingRegressor(estimators=estimators, final_estimator=fe, cv=cv_folds, n_jobs=n_jobs, passthrough=False)