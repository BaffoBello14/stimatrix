from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger
from .metrics import grouped_regression_metrics, _build_price_bands
from joblib import load as joblib_load
from utils.io import save_json
from preprocessing.target_transforms import inverse_target_transform

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

    eval_cfg: Dict[str, Any] = config.get("evaluation", {}) or {}
    gm_cfg: Dict[str, Any] = eval_cfg.get("group_metrics", {}) or {}
    gm_enabled: bool = bool(gm_cfg.get("enabled", True))
    gm_original_scale: bool = bool(gm_cfg.get("original_scale", True))
    gm_report_metrics = gm_cfg.get("report_metrics", ["r2", "rmse", "mse", "mae", "mape", "medae"])
    gm_min_group_size = int(gm_cfg.get("min_group_size", 30))
    price_cfg: Dict[str, Any] = gm_cfg.get("price_band", {}) or {}

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
    transform_metadata: Dict[str, Any] = {"transform": "none"}
    if prep_info_path.exists():
        try:
            import json
            prep_info = json.loads(prep_info_path.read_text(encoding="utf-8"))
            profiles = prep_info.get("profiles_saved", [])
            if profiles:
                prefix = profiles[0]
            transform_metadata = prep_info.get("target_transformation", {"transform": "none"}) or {"transform": "none"}
        except Exception as exc:
            logger.warning(f"Impossibile leggere preprocessing_info.json: {exc}")

    data = _load_preprocessed_for_profile(pre_dir, prefix)
    X_test = data["X_test"]
    y_test = data["y_test"].iloc[:, 0].values
    y_test_orig = data["y_test_orig"].iloc[:, 0].values

    transform_applied = str(transform_metadata.get("transform", "none")).lower() != "none"
    if transform_applied and np.array_equal(y_test_orig, y_test):
        try:
            y_test_orig = np.asarray(inverse_target_transform(y_test, transform_metadata))
        except Exception as exc:
            logger.warning(f"Impossibile invertire la trasformazione del target per i valori di test: {exc}")

    def _inverse_predictions(values: np.ndarray) -> np.ndarray:
        if not transform_applied:
            return np.asarray(values)
        try:
            return np.asarray(inverse_target_transform(np.asarray(values), transform_metadata))
        except Exception as exc:
            logger.warning(f"Impossibile invertire la trasformazione del target per le predizioni: {exc}")
            return np.asarray(values)

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

    # Compute group metrics per-ensemble (evaluation-time fallback)
    try:
        gb_cols = [c for c in (gm_cfg.get("group_by_columns", []) or []) if isinstance(c, str)]
        if gm_enabled and (gb_cols or price_cfg):
            group_cols_path = pre_dir / (f"group_cols_test_{prefix}.parquet" if prefix else "group_cols_test.parquet")
            grp_df = pd.read_parquet(group_cols_path) if group_cols_path.exists() else pd.DataFrame()

            def _ensemble_group_metrics(subdir: str) -> None:
                model_dir = models_dir / subdir
                model_path = model_dir / "model.pkl"
                if not model_path.exists():
                    return
                
                # Load ensemble metadata to get correct profile
                ensemble_meta_path = model_dir / "metrics.json"
                ensemble_prefix = prefix  # Default to global prefix
                if ensemble_meta_path.exists():
                    try:
                        import json
                        ensemble_meta = json.loads(ensemble_meta_path.read_text(encoding="utf-8"))
                        ensemble_prefix = ensemble_meta.get("profile", prefix)
                    except Exception:
                        pass
                
                # CRITICAL FIX: Load ALL data with the SAME ensemble_prefix to ensure alignment
                # This prevents "All arrays must be of the same length" errors
                ensemble_data = _load_preprocessed_for_profile(pre_dir, ensemble_prefix)
                X_test_ensemble = ensemble_data["X_test"]
                y_test_ensemble = ensemble_data["y_test"].iloc[:, 0].values
                y_test_ensemble_orig = ensemble_data["y_test_orig"].iloc[:, 0].values
                
                # Apply inverse transform if needed and y_test_orig == y_test (not already inverted)
                if transform_applied and np.array_equal(y_test_ensemble_orig, y_test_ensemble):
                    try:
                        y_test_ensemble_orig = np.asarray(inverse_target_transform(y_test_ensemble, transform_metadata))
                    except Exception as exc:
                        logger.warning(f"Cannot invert target transform for ensemble '{subdir}': {exc}")
                
                # Load group columns with SAME ensemble_prefix
                group_cols_path_ensemble = pre_dir / (f"group_cols_test_{ensemble_prefix}.parquet" if ensemble_prefix else "group_cols_test.parquet")
                grp_df_ensemble = pd.read_parquet(group_cols_path_ensemble) if group_cols_path_ensemble.exists() else pd.DataFrame()
                
                try:
                    est = joblib_load(model_path)
                except Exception as exc:
                    logger.warning(f"Impossibile caricare il modello ensemble '{subdir}': {exc}")
                    return
                try:
                    y_pred = est.predict(X_test_ensemble.values)
                except Exception:
                    try:
                        y_pred = est.predict(X_test_ensemble)
                    except Exception as exc:
                        logger.warning(f"Predizione fallita per ensemble '{subdir}': {exc}")
                        return

                # Use ensemble-specific y_test to ensure same length as predictions
                y_true_series = pd.Series(y_test_ensemble_orig if gm_original_scale else y_test_ensemble)
                if gm_original_scale:
                    y_pred_series = pd.Series(_inverse_predictions(y_pred))
                else:
                    y_pred_series = pd.Series(y_pred)

                for gb_col in gb_cols:
                    if gb_col not in grp_df_ensemble.columns:
                        continue
                    groups = grp_df_ensemble[gb_col].fillna("MISSING")
                    gm_df = grouped_regression_metrics(
                        y_true=y_true_series,
                        y_pred=y_pred_series,
                        groups=groups,
                        report_metrics=gm_report_metrics,
                        min_group_size=gm_min_group_size,
                        mape_floor=float(price_cfg.get("mape_floor", 1e-8)),
                    )
                    if not gm_df.empty:
                        out_csv = model_dir / f"group_metrics_{gb_col}.csv"
                        gm_df.to_csv(out_csv, index=False)

                if price_cfg:
                    try:
                        bands = _build_price_bands(
                            y_true_orig=y_true_series,
                            method=str(price_cfg.get("method", "quantile")),
                            quantiles=price_cfg.get("quantiles"),
                            fixed_edges=price_cfg.get("fixed_edges"),
                            label_prefix=str(price_cfg.get("label_prefix", "PREZZO_")),
                        )
                        gm_price = grouped_regression_metrics(
                            y_true=y_true_series,
                            y_pred=y_pred_series,
                            groups=bands,
                            report_metrics=gm_report_metrics,
                            min_group_size=gm_min_group_size,
                            mape_floor=float(price_cfg.get("mape_floor", 1e-8)),
                        )
                        if not gm_price.empty:
                            out_csv = model_dir / "group_metrics_price_band.csv"
                            gm_price.to_csv(out_csv, index=False)
                    except Exception as exc:
                        logger.warning(f"Price-band metrics falliti per ensemble '{subdir}': {exc}")

                logger.info(f"Group metrics salvati per ensemble '{subdir}'")

            _ensemble_group_metrics("voting")
            _ensemble_group_metrics("stacking")
    except Exception as e:
        logger.warning(f"Ensemble group metrics (evaluation) failed: {e}")

    return results

