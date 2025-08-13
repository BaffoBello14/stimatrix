from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor


ModelSpec = Tuple[str, Dict[str, Any]]  # (name, params)


def build_estimator(model_key: str, params: Dict[str, Any]):
    mk = model_key.lower()
    if mk == "linear":
        return LinearRegression(**params)
    if mk == "ridge":
        return Ridge(**params)
    if mk == "lasso":
        return Lasso(**params)
    if mk == "elasticnet":
        return ElasticNet(**params)
    if mk == "dt":
        return DecisionTreeRegressor(**params)
    if mk == "knn":
        return KNeighborsRegressor(**params)
    if mk == "svr":
        return SVR(**params)
    if mk == "rf":
        return RandomForestRegressor(**params)
    if mk == "gbr":
        return GradientBoostingRegressor(**params)
    if mk == "hgbt":
        return HistGradientBoostingRegressor(**params)
    if mk == "xgboost":
        try:
            import xgboost as xgb  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("xgboost non installato. Installa 'xgboost' o disabilita il modello nel config.") from exc
        return xgb.XGBRegressor(**params)
    if mk == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("lightgbm non installato. Installa 'lightgbm' o disabilita il modello nel config.") from exc
        return lgb.LGBMRegressor(**params)
    if mk == "catboost":
        try:
            import catboost as cb  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("catboost non installato. Installa 'catboost' o disabilita il modello nel config.") from exc
        return cb.CatBoostRegressor(**params)
    raise ValueError(f"Unknown model key: {model_key}")