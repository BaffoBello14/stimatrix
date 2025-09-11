from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger
from .metrics import grouped_regression_metrics
from joblib import load as joblib_load
from utils.io import save_json

logger = get_logger(__name__)


def _load_preprocessed_for_profile(pre_dir: Path, prefix: Optional[str]) -> Dict[str, pd.DataFrame]:
    def name(base: str) -> Path:
        return pre_dir / (f"{base}_{prefix}.parquet" if prefix else f"{base}.parquet")

    X_test = pd.read_parquet(name("X_test"))
    y_test = pd.read_parquet(name("y_test"))
    y_test_orig_path = name("y_test_orig")
    y_test_orig = pd.read_parquet(y_test_orig_path) if y_test_orig_path.exists() else y_test.copy()

    return {
        "X_test": X_test,
        "y_test": y_test,
        "y_test_orig": y_test_orig,
    }


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
    if prep_info_path.exists():
        try:
            import json
            prep_info = json.loads(prep_info_path.read_text(encoding="utf-8"))
            profiles = prep_info.get("profiles_saved", [])
            if profiles:
                prefix = profiles[0]
        except Exception:
            pass

    data = _load_preprocessed_for_profile(pre_dir, prefix)
    X_test = data["X_test"]
    y_test = data["y_test"].iloc[:, 0].values
    y_test_orig = data["y_test_orig"].iloc[:, 0].values

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
            mt_orig = meta.get("metrics_test_original", {}) or {}
            extract.append({
                "model": name,
                "r2": mt.get("r2"),
                "rmse": mt.get("rmse"),
                "mae": mt.get("mae"),
                "r2_orig": mt_orig.get("r2"),
                "rmse_orig": mt_orig.get("rmse"),
                "mae_orig": mt_orig.get("mae"),
                "mape_floor_orig": mt_orig.get("mape_floor"),
            })
        results["test_metrics"] = extract
    except Exception as e:
        logger.warning(f"Impossibile estrarre test metrics dai risultati di training: {e}")

    out = models_dir / "evaluation_summary.json"
    save_json(results, str(out))
    try:
        _out_log = out.as_posix()
    except Exception:
        _out_log = str(out)
    logger.info(f"Evaluation completata. Report: {_out_log}")

    # Compute group metrics for ensembles (evaluation-time fallback)
    try:
        import json as _json
        prep_info_path = pre_dir / "preprocessing_info.json"
        log_applied_global = False
        if prep_info_path.exists():
            try:
                prep_info = _json.loads(prep_info_path.read_text(encoding="utf-8"))
                log_applied_global = bool(((prep_info or {}).get("log_transformation", {}) or {}).get("applied", False))
            except Exception:
                pass
        gm_cfg = ( ( ( ({} if results is None else {}) ) ) )  # placeholder to avoid linter
        # Reload evaluation config for group settings
        from utils.config import load_config as _load_cfg
        cfg = _load_cfg("config/config.yaml") if (Path("config/config.yaml").exists()) else {}
        eval_cfg: Dict[str, Any] = cfg.get("evaluation", {}) or {}
        gm = eval_cfg.get("group_metrics", {}) or {}
        gb_cols = [c for c in (gm.get("group_by_columns", []) or []) if isinstance(c, str)]
        if gb_cols:
            group_cols_path = pre_dir / (f"group_cols_test_{prefix}.parquet" if prefix else "group_cols_test.parquet")
            grp_df = pd.read_parquet(group_cols_path) if group_cols_path.exists() else pd.DataFrame()
            # Helper to compute and persist
            def _ensemble_group_metrics(model_name: str, subdir: str) -> None:
                model_dir = models_dir / subdir
                model_path = model_dir / "model.pkl"
                if not model_path.exists():
                    return
                try:
                    est = joblib_load(model_path)
                except Exception:
                    return
                try:
                    y_pred = est.predict(X_test.values)
                except Exception:
                    try:
                        y_pred = est.predict(X_test)
                    except Exception:
                        return
                if log_applied_global:
                    y_true_series = pd.Series(y_test_orig)
                    y_pred_series = pd.Series(np.expm1(y_pred))
                else:
                    y_true_series = pd.Series(y_test)
                    y_pred_series = pd.Series(y_pred)
                for gb_col in gb_cols:
                    if gb_col not in grp_df.columns:
                        continue
                    groups = grp_df[gb_col].fillna("MISSING")
                    gm_df = grouped_regression_metrics(
                        y_true=y_true_series,
                        y_pred=y_pred_series,
                        groups=groups,
                        report_metrics=eval_cfg.get("report_metrics", ["r2", "rmse", "mse", "mae", "mape", "medae"]),
                        min_group_size=int(gm.get("min_group_size", 30)),
                        mape_floor=float((gm.get("price_band", {}) or {}).get("mape_floor", 1e-8)),
                    )
                    if not gm_df.empty:
                        out_csv = model_dir / f"group_metrics_{gb_col}.csv"
                        gm_df.to_csv(out_csv, index=False)
                logger.info(f"Group metrics salvati per ensemble '{subdir}'")

            # Voting
            _ensemble_group_metrics("voting", "voting")
            # Stacking
            _ensemble_group_metrics("stacking", "stacking")
    except Exception as e:
        logger.warning(f"Ensemble group metrics (evaluation) failed: {e}")

    return results

