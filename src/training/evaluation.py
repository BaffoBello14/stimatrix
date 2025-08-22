from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.io import save_json
from joblib import load as joblib_load
from .metrics import regression_metrics

logger = get_logger(__name__)


def _load_preprocessed_for_profile(pre_dir: Path, prefix: Optional[str]) -> Dict[str, pd.DataFrame]:
    def name(base: str) -> Path:
        return pre_dir / (f"{base}_{prefix}.parquet" if prefix else f"{base}.parquet")

    X_test = pd.read_parquet(name("X_test"))
    y_test = pd.read_parquet(name("y_test"))
    y_test_orig_path = name("y_test_orig")
    y_test_orig = pd.read_parquet(y_test_orig_path) if y_test_orig_path.exists() else y_test.copy()
    group_keys_path = name("group_keys_test")
    group_keys = pd.read_parquet(group_keys_path) if group_keys_path.exists() else None

    return {
        "X_test": X_test,
        "y_test": y_test,
        "y_test_orig": y_test_orig,
        "group_keys": group_keys,
    }


def _group_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    rows = []
    ser = pd.Series(groups).astype("object")
    for grp, idx in ser.groupby(ser).groups.items():
        mask = np.zeros(len(y_true), dtype=bool)
        mask[list(idx)] = True
        try:
            m = regression_metrics(y_true[mask], y_pred[mask])
        except Exception:
            m = {"mae": float(np.nan), "rmse": float(np.nan), "mape": float(np.nan), "r2": float(np.nan)}
        rows.append({
            "group": grp,
            "count": int(mask.sum()),
            "mae": m.get("mae"),
            "rmse": m.get("rmse"),
            "mape": m.get("mape"),
            "r2": m.get("r2"),
        })
    return pd.DataFrame(rows).sort_values(by=["count"], ascending=False)


def _align_test_to_train(pre_dir: Path, prefix: Optional[str], X_test: pd.DataFrame) -> pd.DataFrame:
    def name(base: str) -> Path:
        return pre_dir / (f"{base}_{prefix}.parquet" if prefix else f"{base}.parquet")
    try:
        X_train = pd.read_parquet(name("X_train"))
    except Exception:
        return X_test
    train_cols = list(X_train.columns)
    train_dtypes = X_train.dtypes
    # Add missing columns with dtype-aware defaults
    missing = [c for c in train_cols if c not in X_test.columns]
    for c in missing:
        if str(train_dtypes[c]) in ("object", "category"):
            X_test[c] = "UNKNOWN"
        else:
            X_test[c] = 0
    # Drop extras and order columns
    X_test = X_test.reindex(columns=train_cols, fill_value=0)
    return X_test


def run_evaluation(config: Dict[str, Any]) -> Dict[str, Any]:
    paths = config.get("paths", {})
    pre_dir = Path(paths.get("preprocessed_data", "data/preprocessed"))
    models_dir = Path(paths.get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== EVALUATION: caricamento modelli e dataset ===")

    # Carica sommario training
    summary_path = models_dir / "summary.json"
    if not summary_path.exists():
        logger.warning("summary.json non trovato: eseguire prima il training")
        return {}
    try:
        training_summary = pd.read_json(summary_path)
    except Exception:
        # Fallback a lettura testuale json
        import json
        training_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # Decidi profilo di riferimento: se presente catboost, usa quello; altrimenti primo profilo salvato in preprocessing_info
    prep_info_path = pre_dir / "preprocessing_info.json"
    prefix: Optional[str] = None
    log_applied: bool = False
    if prep_info_path.exists():
        try:
            import json
            prep_info = json.loads(prep_info_path.read_text(encoding="utf-8"))
            profiles = prep_info.get("profiles_saved", [])
            if profiles:
                prefix = profiles[0]
            log_applied = bool(prep_info.get("log_transformation", {}).get("applied", False))
        except Exception:
            pass

    data = _load_preprocessed_for_profile(pre_dir, prefix)
    X_test_default = data["X_test"]
    y_test_default = data["y_test"].iloc[:, 0].values
    y_test_orig_default = data["y_test_orig"].iloc[:, 0].values
    group_keys_default = data.get("group_keys")
    # Resolve group column names from config
    eval_cfg = config.get("evaluation", {}) if isinstance(config, dict) else {}
    group_cfg = eval_cfg.get("group_columns", {})
    if isinstance(group_cfg, dict):
        col_omi = group_cfg.get("omi", "AI_ZonaOmi")
        col_cat = group_cfg.get("cat", "AI_IdCategoriaCatastale")
    else:
        col_omi, col_cat = "AI_ZonaOmi", "AI_IdCategoriaCatastale"

    # Ricarica i migliori modelli: usiamo il ranking da validation_results.csv se presente
    ranking_csv = models_dir / "validation_results.csv"
    ranked: Optional[pd.DataFrame] = None
    if ranking_csv.exists():
        try:
            ranked = pd.read_csv(ranking_csv)
        except Exception:
            ranked = None

    results: Dict[str, Any] = {}
    if ranked is not None and not ranked.empty:
        results["top_models"] = ranked.head(10).to_dict(orient="records")
    else:
        results["top_models"] = []

    # Non avendo un registry dei modelli caricabili qui, ci limitiamo a riportare le metriche già calcolate su test
    # durante il training (summary.json) e salvarne un estratto per report.
    try:
        import json
        if isinstance(training_summary, dict):
            models = training_summary.get("models", {})
        else:
            models = json.loads(summary_path.read_text(encoding="utf-8")).get("models", {})
        extract = []
        for name, meta in models.items():
            mt = meta.get("metrics_test", {})
            extract.append({
                "model": name,
                "r2": mt.get("r2"),
                "rmse": mt.get("rmse"),
                "mae": mt.get("mae"),
            })
        results["test_metrics"] = extract
    except Exception as e:
        logger.warning(f"Impossibile estrarre test metrics dai risultati di training: {e}")

    # Calcolo statistiche di errore per Zona OMI e Categoria Catastale per ciascun modello
    grouped_stats: Dict[str, Any] = {}
    for model_name in list(models.keys()) if 'models' in locals() else []:
        model_dir = models_dir / model_name
        model_file = model_dir / "model.pkl"
        meta_file = model_dir / "metrics.json"
        if not model_file.exists() or not meta_file.exists():
            continue
        try:
            import json
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            model_prefix = meta.get("prefix", prefix)
            pre = _load_preprocessed_for_profile(pre_dir, model_prefix)
            X_test = pre["X_test"].copy()
            y_test = pre["y_test"].iloc[:, 0].values
            y_test_orig = pre["y_test_orig"].iloc[:, 0].values
            group_keys = pre.get("group_keys")
            # Fallback a default se non presenti file per prefisso
            if X_test is None or X_test.empty:
                X_test = X_test_default.copy()
                y_test = y_test_default
                y_test_orig = y_test_orig_default
                group_keys = group_keys_default
            # Align columns to training features
            X_test = _align_test_to_train(pre_dir, model_prefix, X_test)

            estimator = joblib_load(model_file)
            try:
                y_pred = estimator.predict(X_test)
            except Exception:
                y_pred = estimator.predict(X_test.values)

            if log_applied:
                y_true_for_groups = y_test_orig
                y_pred_for_groups = np.expm1(y_pred)
            else:
                y_true_for_groups = y_test
                y_pred_for_groups = y_pred

            stats_entry: Dict[str, Any] = {}
            if group_keys is not None and len(group_keys) == len(y_true_for_groups):
                if col_omi in group_keys.columns:
                    df_omi = _group_metrics(y_true_for_groups, y_pred_for_groups, group_keys[col_omi]) \
                        .to_dict(orient="records")
                    stats_entry["by_zona_omi"] = df_omi
                if col_cat in group_keys.columns:
                    df_cat = _group_metrics(y_true_for_groups, y_pred_for_groups, group_keys[col_cat]) \
                        .to_dict(orient="records")
                    stats_entry["by_categoria_catastale"] = df_cat
            else:
                logger.warning(f"Group keys non disponibili o non allineati per il modello {model_name}")

            grouped_stats[model_name] = stats_entry
        except Exception as e:
            logger.warning(f"Impossibile calcolare grouped stats per {model_name}: {e}")

    results["grouped_stats"] = grouped_stats

    out = models_dir / "evaluation_summary.json"
    save_json(results, str(out))
    try:
        _out_log = out.as_posix()
    except Exception:
        _out_log = str(out)
    logger.info(f"Evaluation completata. Report: {_out_log}")
    return results

