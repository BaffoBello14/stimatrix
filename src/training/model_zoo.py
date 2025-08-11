from __future__ import annotations

from typing import Dict, Any, Tuple, List

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

import xgboost as xgb
import lightgbm as lgb
import catboost as cb


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
        return xgb.XGBRegressor(**params)
    if mk == "lightgbm":
        return lgb.LGBMRegressor(**params)
    if mk == "catboost":
        return cb.CatBoostRegressor(**params)
    raise ValueError(f"Unknown model key: {model_key}")