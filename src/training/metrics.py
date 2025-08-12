from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn import metrics as skm


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = skm.mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = skm.mean_absolute_error(y_true, y_pred)
    r2 = skm.r2_score(y_true, y_pred)
    evs = skm.explained_variance_score(y_true, y_pred)
    medae = skm.median_absolute_error(y_true, y_pred)
    mape = _safe_mape(y_true, y_pred)
    return {
        "r2": float(r2),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "explained_variance": float(evs),
        "medae": float(medae),
    }


def overfit_diagnostics(train: Dict[str, float], test: Dict[str, float]) -> Dict[str, float]:
    diag: Dict[str, float] = {}
    # Positive is better for r2/evs, so gap = train - test (overfit if large positive)
    diag["gap_r2"] = float(train.get("r2", np.nan) - test.get("r2", np.nan))
    diag["gap_explained_variance"] = float(train.get("explained_variance", np.nan) - test.get("explained_variance", np.nan))
    # Errors: lower is better, so ratio = test/train (overfit if >> 1)
    for err in ["rmse", "mse", "mae", "mape", "medae"]:
        tr = train.get(err, np.nan)
        te = test.get(err, np.nan)
        diag[f"ratio_{err}"] = float(np.nan) if tr in (0.0, np.nan) else float(te / tr)
        diag[f"delta_{err}"] = float(te - tr)
    return diag


def select_primary_value(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # We always MAXIMIZE. For neg_* errors we compute according to sklearn convention.
    name = metric_name.lower()
    if name == "r2":
        return float(skm.r2_score(y_true, y_pred))
    elif name == "neg_mean_squared_error":
        return float(-skm.mean_squared_error(y_true, y_pred))
    elif name == "neg_root_mean_squared_error":
        return float(-np.sqrt(skm.mean_squared_error(y_true, y_pred)))
    elif name == "neg_mean_absolute_error":
        return float(-skm.mean_absolute_error(y_true, y_pred))
    elif name == "neg_mean_absolute_percentage_error":
        return float(-_safe_mape(y_true, y_pred))
    else:
        raise ValueError(f"Unsupported primary metric: {metric_name}")