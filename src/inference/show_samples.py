from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure relative imports work when executed as a module/script
import sys as _sys
_root = Path(__file__).resolve().parents[1]
if str(_root) not in _sys.path:
    _sys.path.append(str(_root))

from utils.config import load_config
# NOTE: Avoid importing heavy deps (numpy/pandas/sklearn) at module import time
# We'll import training.metrics._build_price_bands lazily inside main()
_build_price_bands_ref = None


def _detect_best_model(models_dir: Path) -> Tuple[str, Path]:
    summary_path = models_dir / "summary.json"
    if not summary_path.exists():
        # Fallback: choose first directory with model.pkl
        for p in sorted(models_dir.iterdir()):
            if p.is_dir() and (p / "model.pkl").exists():
                return p.name, p
        raise FileNotFoundError(f"Nessun modello trovato in {models_dir}")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    models = data.get("models", {})
    if not models:
        # try ensembles
        ensembles = data.get("ensembles", {})
        if ensembles:
            # pick voting if present, else first
            if "voting" in ensembles and (models_dir / "voting" / "model.pkl").exists():
                return "voting", models_dir / "voting"
            key, _ = next(iter(ensembles.items()))
            return key, models_dir / key
        # fallback first model dir
        for p in sorted(models_dir.iterdir()):
            if p.is_dir() and (p / "model.pkl").exists():
                return p.name, p
        raise FileNotFoundError("summary.json presente ma nessun modello valido trovato")
    # rank by best_primary_value descending
    ranked = sorted(((k, v.get("best_primary_value", float("-inf"))) for k, v in models.items()), key=lambda x: x[1], reverse=True)
    best_key = ranked[0][0]
    return best_key, models_dir / best_key


def _profile_for_model(model_key: str, cfg: dict) -> Optional[str]:
    tr_cfg = cfg.get("training", {}) or {}
    profile_map = tr_cfg.get("profile_map") or {}
    # In config the profile is set per model in training.models[model_key].profile
    mdl_cfg = (tr_cfg.get("models", {}) or {}).get(model_key, {})
    return mdl_cfg.get("profile", profile_map.get(model_key))


def _load_test_split(pre_dir: Path, prefix: Optional[str]) -> Tuple["pd.DataFrame", "pd.Series", "pd.DataFrame"]:
    import pandas as pd
    def name(base: str) -> Path:
        return pre_dir / (f"{base}_{prefix}.parquet" if prefix else f"{base}.parquet")
    X_test = pd.read_parquet(name("X_test"))
    y_test_df = pd.read_parquet(name("y_test"))
    target_col = y_test_df.columns[0]
    y_test = y_test_df[target_col]
    # group columns sidecar if present
    gc_path = name("group_cols_test")
    group_cols = pd.read_parquet(gc_path) if gc_path.exists() else pd.DataFrame(index=X_test.index)
    return X_test, y_test, group_cols


def _maybe_inverse_log(cfg: dict, y_pred: "np.ndarray", y_true_series: "pd.Series", smearing_factor: Optional[float]) -> Tuple["np.ndarray", "np.ndarray"]:
    import numpy as np
    import pandas as pd
    # Determine if global log transform was applied
    pre_dir = Path(cfg.get("paths", {}).get("preprocessed_data", "data/preprocessed"))
    prep_info_path = pre_dir / "preprocessing_info.json"
    log_applied_global = False
    if prep_info_path.exists():
        try:
            info = json.loads(prep_info_path.read_text(encoding="utf-8"))
            log_applied_global = bool(((info or {}).get("log_transformation", {}) or {}).get("applied", False))
        except Exception:
            log_applied_global = False
    if log_applied_global:
        # Prefer y_test_orig file if available to avoid numerical drift
        try:
            y_test_orig_path = pre_dir / "y_test_orig.parquet"
            if y_test_orig_path.exists():
                y_true_orig = pd.read_parquet(y_test_orig_path).iloc[:, 0].values
            else:
                y_true_orig = np.expm1(y_true_series.values)
        except Exception:
            y_true_orig = np.expm1(y_true_series.values)
        # Duan's smearing if provided
        smear = float(smearing_factor) if smearing_factor is not None else 1.0
        y_pred_orig = np.expm1(np.asarray(y_pred)) * smear
        return y_true_orig, y_pred_orig
    # No log: identity
    return y_true_series.values, np.asarray(y_pred)


def select_diverse_samples(
    df: "pd.DataFrame",
    wanted: int,
    cols_zone: str = "AI_ZonaOmi",
    cols_type: str = "AI_IdTipologiaEdilizia",
    price_band_col: str = "__PRICE_BAND__",
    random_state: int = 42,
) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd
    # Stratify across OMI zone, building type, and price band
    rng = np.random.default_rng(random_state)
    groups = df[[cols_zone, cols_type, price_band_col]].fillna("MISSING").astype(str)
    df = df.copy()
    df["__grp__"] = groups[cols_zone] + "|" + groups[cols_type] + "|" + groups[price_band_col]
    out_idx: List[int] = []
    # Round-robin sampling one from each group
    per_group = wanted  # upper bound; we'll take min(1, size) per round
    grouped = {g: list(idx.values) for g, idx in df.groupby("__grp__").groups.items()}
    # Ensure deterministic order
    keys = sorted(grouped.keys())
    # First pass: one per group
    for k in keys:
        if grouped[k]:
            out_idx.append(grouped[k][0])
    if len(out_idx) >= wanted:
        return df.loc[out_idx[:wanted]].copy()
    # Second pass: fill randomly from remaining
    remaining = [i for k in keys for i in grouped[k][1:]]
    if remaining:
        rng.shuffle(remaining)
        need = wanted - len(out_idx)
        out_idx.extend(remaining[:need])
    return df.loc[out_idx].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mostra sample di test con prezzo reale e predetto")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path al config YAML")
    parser.add_argument("--model", type=str, default=None, help="Chiave modello da caricare (es. 'rf', 'xgboost'); default migliore")
    parser.add_argument("--n", type=int, default=12, help="Numero di sample da mostrare")
    parser.add_argument("--profile", type=str, default=None, help="Profilo dataset (scaled/tree/catboost). Default basato sul modello")
    parser.add_argument("--features", type=str, nargs="*", default=[
        "AI_ZonaOmi", "AI_IdTipologiaEdilizia", "AI_IdCategoriaCatastale", "AI_Superficie"
    ], help="Feature da stampare insieme a prezzi")
    args = parser.parse_args()

    # Heavy imports after parsing, so --help works without deps installed
    import numpy as np
    import pandas as pd
    from joblib import load
    from training.metrics import _build_price_bands as _bp
    global _build_price_bands_ref
    _build_price_bands_ref = _bp

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    pre_dir = Path(paths.get("preprocessed_data", "data/preprocessed"))
    models_dir = Path(paths.get("models_dir", "models"))

    if args.model is None:
        model_key, model_dir = _detect_best_model(models_dir)
    else:
        model_key = args.model
        model_dir = models_dir / model_key
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    # Determine dataset profile
    prefix = args.profile or _profile_for_model(model_key, cfg)
    X_test, y_test, group_cols = _load_test_split(pre_dir, prefix)

    # Align columns for models that expect numeric-only arrays
    # Heuristic: if all columns are numeric, use values; otherwise let estimator handle
    numeric_only = X_test.select_dtypes(include=[np.number]).shape[1] == X_test.shape[1]

    estimator = load(model_path)

    if numeric_only:
        y_pred = estimator.predict(X_test.values)
    else:
        try:
            y_pred = estimator.predict(X_test)
        except Exception:
            # Fallback to numeric values
            y_pred = estimator.predict(X_test.select_dtypes(include=[np.number]).values)

    # Try to read smearing factor for consistent original-scale inversion
    metrics_path = model_dir / "metrics.json"
    smear: Optional[float] = None
    if metrics_path.exists():
        try:
            meta = json.loads(metrics_path.read_text(encoding="utf-8"))
            smear = meta.get("smearing_factor")
        except Exception:
            smear = None

    y_true_orig, y_pred_orig = _maybe_inverse_log(cfg, y_pred, y_test, smear)

    # Build a working frame
    df = X_test.copy()
    # Attach grouping columns sidecar if present
    for c in group_cols.columns:
        if c not in df.columns:
            df[c] = group_cols[c].values
    df["__y_true__"] = y_true_orig
    df["__y_pred__"] = y_pred_orig

    # Compute price bands from true original prices
    eval_cfg = (cfg.get("evaluation", {}) or {}).get("group_metrics", {}) or {}
    price_cfg = eval_cfg.get("price_band", {}) or {}
    bands = _build_price_bands_ref(
        y_true_orig=pd.Series(y_true_orig, index=df.index),
        method=str(price_cfg.get("method", "quantile")),
        quantiles=price_cfg.get("quantiles"),
        fixed_edges=price_cfg.get("fixed_edges"),
        label_prefix=str(price_cfg.get("label_prefix", "PREZZO_")),
    )
    df["__PRICE_BAND__"] = bands.astype(str)

    # Select diverse samples
    wanted = max(3, int(args.n))
    diverse = select_diverse_samples(df, wanted=wanted)

    # Print results
    cols_to_show: List[str] = []
    for c in args.features:
        if c in diverse.columns:
            cols_to_show.append(c)
    # Add latitude/longitude columns if present, replacing the price band in output
    for geo_col in [
        "AI_Latitudine",
        "AI_Longitudine",
        "latitudine",
        "longitudine",
        "latitude",
        "longitude",
    ]:
        if geo_col in diverse.columns and geo_col not in cols_to_show:
            cols_to_show.append(geo_col)
    # Always append actual and predicted prices
    cols_to_show.extend(["__y_true__", "__y_pred__"])
    out = diverse[cols_to_show].copy()

    # Pretty formatting
    def fmt_money(x: float) -> str:
        try:
            return f"{float(x):,.0f}".replace(",", "_").replace(".", ",").replace("_", ".")
        except Exception:
            return str(x)

    out = out.rename(columns={"__y_true__": "PrezzoReale", "__y_pred__": "PrezzoPredetto"})
    with pd.option_context("display.max_columns", None, "display.width", 200):
        # Apply formatting to price columns
        printable = out.copy()
        for col in ["PrezzoReale", "PrezzoPredetto"]:
            if col in printable.columns:
                printable[col] = printable[col].apply(fmt_money)
        print(printable.to_string(index=False))


if __name__ == "__main__":
    main()

