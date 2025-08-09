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


def default_params(model_key: str) -> Dict[str, Any]:
    mk = model_key.lower()
    if mk in {"linear"}:
        return {"n_jobs": -1}
    if mk == "ridge":
        return {"random_state": 42}
    if mk == "lasso":
        return {"random_state": 42, "max_iter": 20000}
    if mk == "elasticnet":
        return {"random_state": 42, "max_iter": 20000}
    if mk == "dt":
        return {"random_state": 42}
    if mk == "knn":
        return {}
    if mk == "svr":
        return {"kernel": "rbf"}
    if mk == "rf":
        return {"n_estimators": 300, "n_jobs": -1, "random_state": 42}
    if mk == "gbr":
        return {"random_state": 42}
    if mk == "hgbt":
        return {"random_state": 42}
    if mk == "xgboost":
        return {"n_estimators": 800, "tree_method": "hist", "random_state": 42, "n_jobs": -1}
    if mk == "lightgbm":
        return {"n_estimators": 800, "random_state": 42, "n_jobs": -1}
    if mk == "catboost":
        return {"iterations": 800, "random_seed": 42, "verbose": False}
    return {}


def suggest_params(model_key: str, trial) -> Dict[str, Any]:
    mk = model_key.lower()
    p: Dict[str, Any] = default_params(model_key).copy()
    if mk == "linear":
        return p
    if mk == "ridge":
        p["alpha"] = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
        return p
    if mk == "lasso":
        p["alpha"] = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
        return p
    if mk == "elasticnet":
        p["alpha"] = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
        p["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        return p
    if mk == "dt":
        p["max_depth"] = trial.suggest_int("max_depth", 3, 30)
        p["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 20)
        return p
    if mk == "knn":
        p["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 50)
        p["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
        p["p"] = trial.suggest_int("p", 1, 2)
        return p
    if mk == "svr":
        p["kernel"] = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
        p["C"] = trial.suggest_float("C", 1e-2, 1e3, log=True)
        p["epsilon"] = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
        if p["kernel"] in ("rbf", "poly", "sigmoid"):
            p["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
        if p["kernel"] == "poly":
            p["degree"] = trial.suggest_int("degree", 2, 5)
        return p
    if mk == "rf":
        p["n_estimators"] = trial.suggest_int("n_estimators", 200, 1200)
        p["max_depth"] = trial.suggest_int("max_depth", 4, 40)
        p["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 20)
        p["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        return p
    if mk == "gbr":
        p["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        p["n_estimators"] = trial.suggest_int("n_estimators", 200, 1500)
        p["max_depth"] = trial.suggest_int("max_depth", 2, 8)
        p["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 20)
        return p
    if mk == "hgbt":
        p["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        p["max_depth"] = trial.suggest_int("max_depth", 2, 16)
        p["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 7, 255)
        p["l2_regularization"] = trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True)
        return p
    if mk == "xgboost":
        p["n_estimators"] = trial.suggest_int("n_estimators", 300, 1500)
        p["max_depth"] = trial.suggest_int("max_depth", 3, 12)
        p["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        p["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        p["min_child_weight"] = trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True)
        p["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
        p["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
        return p
    if mk == "lightgbm":
        p["n_estimators"] = trial.suggest_int("n_estimators", 300, 1500)
        p["max_depth"] = trial.suggest_int("max_depth", -1, 12)
        p["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        p["num_leaves"] = trial.suggest_int("num_leaves", 15, 255)
        p["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        p["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
        p["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
        return p
    if mk == "catboost":
        p["depth"] = trial.suggest_int("depth", 4, 10)
        p["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        p["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1.0, 15.0)
        p["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 5.0)
        p["border_count"] = trial.suggest_int("border_count", 16, 255)
        return p
    raise ValueError(f"Unknown model key for suggestions: {model_key}")