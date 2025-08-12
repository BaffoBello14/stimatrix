from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import shap

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None
try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None
try:
    import catboost as cb  # type: ignore
except Exception:  # pragma: no cover
    cb = None


def _is_tree_based(model) -> bool:
    tree_types = (
        DecisionTreeRegressor,
        RandomForestRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
    )
    if isinstance(model, tree_types):
        return True
    if xgb is not None and isinstance(model, xgb.XGBRegressor):
        return True
    if lgb is not None and isinstance(model, lgb.LGBMRegressor):
        return True
    if cb is not None and isinstance(model, cb.CatBoostRegressor):
        return True
    return False


def _is_linear(model) -> bool:
    return isinstance(model, (LinearRegression, Ridge, Lasso, ElasticNet))


def compute_shap(
    model,
    X: pd.DataFrame,
    sample_size: int = 2000,
    max_display: int = 30,
    background_size: int = 500,
) -> Dict[str, Any]:
    if len(X) > sample_size:
        Xs = X.sample(n=sample_size, random_state=42)
    else:
        Xs = X

    shap_values = None
    explainer = None

    try:
        if _is_tree_based(model):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(Xs)
        elif _is_linear(model):
            try:
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer(Xs)
            except Exception:
                X_bg = X.sample(n=min(len(X), background_size), random_state=42)
                explainer = shap.Explainer(model.predict, X_bg.values)
                shap_values = explainer(Xs.values)
        else:
            # Generic fallback: use function handle with numpy arrays to avoid feature-name warnings
            X_bg = X.sample(n=min(len(X), background_size), random_state=42)
            explainer = shap.Explainer(model.predict, X_bg.values)
            shap_values = explainer(Xs.values)
    except Exception:
        # Final safety fallback: force numpy arrays
        X_bg = X.sample(n=min(len(X), background_size), random_state=42)
        explainer = shap.Explainer(getattr(model, "predict", model), X_bg.values)
        shap_values = explainer(Xs.values)

    try:
        abs_mean = np.abs(shap_values.values).mean(axis=0)
        importance = (
            pd.Series(abs_mean, index=(Xs.columns if hasattr(Xs, 'columns') else range(len(abs_mean)))).sort_values(ascending=False).to_dict()
        )
    except Exception:
        importance = {}

    return {
        "sample_size": int(len(Xs)),
        "importance": importance,
        "values": shap_values,
        "data_sample": Xs,
        "max_display": max_display,
    }


ess = __import__("pathlib").Path

def save_shap_plots(out_dir: str, shap_bundle: Dict[str, Any], model_name: str) -> None:
    out_dir_path = ess(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    plot_base = out_dir_path / f"shap_{model_name}"
    try:
        shap.plots.beeswarm(shap_bundle["values"], max_display=shap_bundle.get("max_display", 30), show=False)
        __import__("matplotlib.pyplot").pyplot.savefig(f"{plot_base}_beeswarm.png", bbox_inches="tight", dpi=200)
        __import__("matplotlib.pyplot").pyplot.close()
    except Exception:
        pass
    try:
        shap.plots.bar(shap_bundle["values"], max_display=shap_bundle.get("max_display", 30), show=False)
        __import__("matplotlib.pyplot").pyplot.savefig(f"{plot_base}_bar.png", bbox_inches="tight", dpi=200)
        __import__("matplotlib.pyplot").pyplot.close()
    except Exception:
        pass