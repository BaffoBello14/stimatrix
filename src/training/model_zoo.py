from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

from .constants import (
    CATBOOST_KEY,
    XGBOOST_KEY,
    LIGHTGBM_KEY,
    RANDOM_FOREST_KEY,
    GRADIENT_BOOSTING_KEY,
    HIST_GRADIENT_BOOSTING_KEY,
    DECISION_TREE_KEY,
    KNN_KEY,
    SVR_KEY,
    LINEAR_KEY,
    RIDGE_KEY,
    LASSO_KEY,
    ELASTICNET_KEY,
)


ModelSpec = Tuple[str, Dict[str, Any]]  # (name, params)


def build_estimator(model_key: str, params: Dict[str, Any]):
    mk = model_key.lower()
    if mk == LINEAR_KEY:
        return LinearRegression(**params)
    if mk == RIDGE_KEY:
        return Ridge(**params)
    if mk == LASSO_KEY:
        return Lasso(**params)
    if mk == ELASTICNET_KEY:
        return ElasticNet(**params)
    if mk == DECISION_TREE_KEY:
        return DecisionTreeRegressor(**params)
    if mk == KNN_KEY:
        return KNeighborsRegressor(**params)
    if mk == SVR_KEY:
        return SVR(**params)
    if mk == RANDOM_FOREST_KEY:
        return RandomForestRegressor(**params)
    if mk == GRADIENT_BOOSTING_KEY:
        return GradientBoostingRegressor(**params)
    if mk == HIST_GRADIENT_BOOSTING_KEY:
        return HistGradientBoostingRegressor(**params)
    if mk == XGBOOST_KEY:
        try:
            import xgboost as xgb  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("xgboost non installato. Installa 'xgboost' o disabilita il modello nel config.") from exc
        return xgb.XGBRegressor(**params)
    if mk == LIGHTGBM_KEY:
        try:
            import lightgbm as lgb  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("lightgbm non installato. Installa 'lightgbm' o disabilita il modello nel config.") from exc
        return lgb.LGBMRegressor(**params)
    if mk == CATBOOST_KEY:
        try:
            import catboost as cb  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("catboost non installato. Installa 'catboost' o disabilita il modello nel config.") from exc
        return cb.CatBoostRegressor(**params)
    raise ValueError(f"Unknown model key: {model_key}")