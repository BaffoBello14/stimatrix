from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn import metrics as skm

import pandas as pd


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Filter out NaN values to avoid sklearn errors
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        # All values are NaN or empty, raise ValueError
        raise ValueError("Cannot compute metrics: all values are NaN or data is empty")
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    mse = skm.mean_squared_error(y_true_clean, y_pred_clean)
    rmse = float(np.sqrt(mse))
    mae = skm.mean_absolute_error(y_true_clean, y_pred_clean)
    r2 = skm.r2_score(y_true_clean, y_pred_clean)
    evs = skm.explained_variance_score(y_true_clean, y_pred_clean)
    medae = skm.median_absolute_error(y_true_clean, y_pred_clean)
    mape = _safe_mape(y_true_clean, y_pred_clean)
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


def _build_price_bands(
    y_true_orig: pd.Series,
    method: str,
    quantiles: list[float] | None = None,
    fixed_edges: list[float] | None = None,
    label_prefix: str = "PREZZO_",
) -> pd.Series:
    if method == "quantile":
        qs = quantiles or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        edges = np.unique(np.quantile(y_true_orig.values, qs))
        # ensure strictly increasing by nudging duplicates
        edges = np.unique(edges)
        bands = pd.cut(y_true_orig.values, bins=np.concatenate([[-np.inf], edges[1:-1], [np.inf]]), include_lowest=True)
    elif method == "fixed":
        if not fixed_edges:
            raise ValueError("fixed_edges must be provided when price_band.method='fixed'")
        edges = np.array(fixed_edges, dtype=float)
        bands = pd.cut(y_true_orig.values, bins=edges, include_lowest=True)
    else:
        raise ValueError(f"Unsupported price band method: {method}")
    return bands.astype(str).map(lambda s: f"{label_prefix}{s}")


def grouped_regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    groups: pd.Series,
    report_metrics: list[str] | None = None,
    min_group_size: int = 30,
    mape_floor: float = 1e-8,
) -> pd.DataFrame:
    report_metrics = report_metrics or ["r2", "rmse", "mse", "mae", "mape", "medae"]
    df = pd.DataFrame({"y_true": y_true.values, "y_pred": y_pred.values, "__grp__": groups.values})
    df["__valid__"] = (~pd.isna(df["y_true"]) & ~pd.isna(df["y_pred"]) & ~pd.isna(df["__grp__"]))
    df = df.loc[df["__valid__"]].copy()
    if df.empty:
        return pd.DataFrame(columns=["group", *report_metrics, "count"]).astype({"group": str, "count": int})
    out_rows: list[dict[str, float | str | int]] = []
    for group_value, g in df.groupby("__grp__"):
        if len(g) < int(min_group_size):
            continue
        yt = g["y_true"].values
        yp = g["y_pred"].values
        # custom mape with floor
        denom = np.where(np.abs(yt) < max(mape_floor, 1e-8), max(mape_floor, 1e-8), np.abs(yt))
        mape = float(np.mean(np.abs((yt - yp) / denom)))
        mse = skm.mean_squared_error(yt, yp)
        rmse = float(np.sqrt(mse))
        mae = skm.mean_absolute_error(yt, yp)
        medae = skm.median_absolute_error(yt, yp)
        # r2 may be undefined for constant yt; guard
        try:
            r2 = skm.r2_score(yt, yp)
        except Exception:
            r2 = np.nan
        metrics_map = {"r2": float(r2), "mse": float(mse), "rmse": float(rmse), "mae": float(mae), "mape": float(mape), "medae": float(medae)}
        row = {"group": str(group_value), "count": int(len(g))}
        for m in report_metrics:
            if m in metrics_map:
                row[m] = metrics_map[m]
        out_rows.append(row)
    if not out_rows:
        return pd.DataFrame(columns=["group", *report_metrics, "count"]).astype({"group": str, "count": int})
    result = pd.DataFrame(out_rows).sort_values(by=report_metrics[0] if report_metrics else "rmse", ascending=False)
    return result


__all__ = [
    "regression_metrics",
    "overfit_diagnostics",
    "select_primary_value",
    "grouped_regression_metrics",
    "_build_price_bands",
]