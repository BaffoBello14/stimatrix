from __future__ import annotations

from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
import shap

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to avoid Tkinter warnings

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
    keep_as_numpy: bool = False,
    random_state: int = 42,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # Convert numpy array to DataFrame if needed, but remember original format
    was_numpy = isinstance(X, np.ndarray)
    if isinstance(X, np.ndarray):
        # Use provided feature names or generate generic ones
        if feature_names is not None:
            columns = feature_names
        else:
            columns = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=columns)
    elif not isinstance(X, pd.DataFrame):
        # Handle other array-like objects
        if feature_names is not None:
            X = pd.DataFrame(X, columns=feature_names)
        else:
            X = pd.DataFrame(X)
    
    if len(X) > sample_size:
        Xs = X.sample(n=sample_size, random_state=random_state)
    else:
        Xs = X

    shap_values = None
    explainer = None

    try:
        if _is_tree_based(model):
            explainer = shap.TreeExplainer(model)
            # For tree models, use the same format as training
            if keep_as_numpy or was_numpy:
                shap_values = explainer(Xs.values)
            else:
                shap_values = explainer(Xs)
        elif _is_linear(model):
            try:
                # Linear models work better with consistent format
                if keep_as_numpy or was_numpy:
                    explainer = shap.LinearExplainer(model, X.values)
                    shap_values = explainer(Xs.values)
                else:
                    explainer = shap.LinearExplainer(model, X)
                    shap_values = explainer(Xs)
            except Exception:
                X_bg = X.sample(n=min(len(X), background_size), random_state=random_state)
                explainer = shap.Explainer(model.predict, X_bg.values)
                shap_values = explainer(Xs.values)
        else:
            # Generic fallback: use function handle with numpy arrays to avoid feature-name warnings
            X_bg = X.sample(n=min(len(X), background_size), random_state=random_state)
            explainer = shap.Explainer(model.predict, X_bg.values)
            shap_values = explainer(Xs.values)
    except Exception:
        # Final safety fallback: force numpy arrays
        X_bg = X.sample(n=min(len(X), background_size), random_state=random_state)
        explainer = shap.Explainer(getattr(model, "predict", model), X_bg.values)
        shap_values = explainer(Xs.values)

    # Ensure SHAP values have correct feature names
    if shap_values is not None and hasattr(Xs, 'columns'):
        try:
            # Set feature names on SHAP values for correct plotting
            shap_values.feature_names = list(Xs.columns)
        except Exception:
            pass

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
    # Beeswarm plot
    try:
        shap.plots.beeswarm(shap_bundle["values"], max_display=shap_bundle.get("max_display", 30), show=False)
        __import__("matplotlib.pyplot").pyplot.savefig(f"{plot_base}_beeswarm.png", bbox_inches="tight", dpi=200)
        __import__("matplotlib.pyplot").pyplot.close()
    except Exception:
        try:
            shap.summary_plot(shap_bundle["values"], shap_bundle.get("data_sample"), show=False, plot_type="dot", max_display=shap_bundle.get("max_display", 30))
            __import__("matplotlib.pyplot").pyplot.savefig(f"{plot_base}_beeswarm.png", bbox_inches="tight", dpi=200)
            __import__("matplotlib.pyplot").pyplot.close()
        except Exception:
            pass
    # Bar plot
    try:
        shap.plots.bar(shap_bundle["values"], max_display=shap_bundle.get("max_display", 30), show=False)
        __import__("matplotlib.pyplot").pyplot.savefig(f"{plot_base}_bar.png", bbox_inches="tight", dpi=200)
        __import__("matplotlib.pyplot").pyplot.close()
    except Exception:
        try:
            shap.summary_plot(shap_bundle["values"], shap_bundle.get("data_sample"), show=False, plot_type="bar", max_display=shap_bundle.get("max_display", 30))
            __import__("matplotlib.pyplot").pyplot.savefig(f"{plot_base}_bar.png", bbox_inches="tight", dpi=200)
            __import__("matplotlib.pyplot").pyplot.close()
        except Exception:
            pass
    # Always persist importances as CSV for traceability
    try:
        import pandas as _pd  # local import to keep module import light
        importance: Dict[str, Any] = shap_bundle.get("importance", {}) or {}
        if importance:
            (_pd.Series(importance).rename("importance")
                .to_frame()
                .to_csv(out_dir_path / f"{model_name}_importance.csv"))
    except Exception:
        pass