from __future__ import annotations

from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import shap


def compute_shap(
    model,
    X: pd.DataFrame,
    sample_size: int = 2000,
    max_display: int = 30,
) -> Dict[str, Any]:
    if len(X) > sample_size:
        Xs = X.sample(n=sample_size, random_state=42)
    else:
        Xs = X
    try:
        explainer = shap.Explainer(model, Xs)
    except Exception:
        explainer = shap.Explainer(model)
    shap_values = explainer(Xs)
    # Summary stats per feature
    try:
        import numpy as np
        abs_mean = np.abs(shap_values.values).mean(axis=0)
        importance = (
            pd.Series(abs_mean, index=Xs.columns).sort_values(ascending=False).to_dict()
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


def save_shap_plots(out_dir: str, shap_bundle: Dict[str, Any], model_name: str) -> None:
    out_dir_path = __import__("pathlib").Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    plot_base = out_dir_path / f"shap_{model_name}"
    try:
        # Beeswarm
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