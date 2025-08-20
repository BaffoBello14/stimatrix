from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger
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
            extract.append({
                "model": name,
                "r2": mt.get("r2"),
                "rmse": mt.get("rmse"),
                "mae": mt.get("mae"),
            })
        results["test_metrics"] = extract
    except Exception as e:
        logger.warning(f"Impossibile estrarre test metrics dai risultati di training: {e}")

    out = models_dir / "evaluation_summary.json"
    save_json(results, str(out))
    logger.info(f"Evaluation completata. Report: {out}")
    return results

