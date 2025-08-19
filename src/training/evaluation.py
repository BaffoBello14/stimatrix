from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import json
import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def _load_xy(pre_dir: Path, prefix: str | None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    def name(base: str) -> Path:
        return pre_dir / (f"{base}_{prefix}.parquet" if prefix else f"{base}.parquet")

    X_train = pd.read_parquet(name("X_train"))
    y_train_df = pd.read_parquet(name("y_train"))
    target_col = y_train_df.columns[0]
    y_train = y_train_df[target_col]

    X_test = pd.read_parquet(name("X_test"))
    y_test = pd.read_parquet(name("y_test"))[target_col]
    return X_train, y_train, X_test, y_test


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        mean_absolute_percentage_error,
    )

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}


def run_evaluation(config: Dict[str, Any]) -> Dict[str, Any]:
    paths = config.get("paths", {})
    pre_dir = Path(paths.get("preprocessed_data", "data/preprocessed"))
    models_dir = Path(paths.get("models_dir", "models"))

    eval_cfg = config.get("evaluation", {})
    outputs_dir = Path(eval_cfg.get("outputs_dir", models_dir / "evaluation"))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Se esiste un riepilogo training, usalo per ottenere i modelli
    summary_path = models_dir / "summary.json"
    if not summary_path.exists():
        logger.warning("Nessun summary di training trovato. Eseguo evaluation solo su dataset senza modello.")

    results: Dict[str, Any] = {"timestamp": pd.Timestamp.utcnow().isoformat()}

    # Metriche baseline: regressione triviale (media del train) come contesto
    try:
        # Usa il primo profilo attivo per caricare i dati
        prefix = None
        # Preferisci 'scaled' se presente
        for cand in ["scaled", "tree", "catboost", None]:
            try:
                X_train, y_train, X_test, y_test = _load_xy(pre_dir, cand)
                prefix = cand
                break
            except Exception:
                continue
        if prefix is None:
            raise FileNotFoundError("Impossibile caricare X_train/X_test/y_train/y_test")

        y_pred_baseline = np.full_like(y_test.values, fill_value=np.mean(y_train.values), dtype=float)
        baseline_metrics = _metrics(y_test.values, y_pred_baseline)
        results["baseline"] = {"profile": prefix, "metrics_test": baseline_metrics}
    except Exception as e:
        logger.warning(f"Baseline evaluation fallita: {e}")

    # Aggrega metriche dei modelli se disponibili
    try:
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            results["models"] = summary.get("models", {})
            results["ensembles"] = summary.get("ensembles", {})
    except Exception as e:
        logger.warning(f"Caricamento summary fallito: {e}")

    # Salva report JSON
    (outputs_dir / "evaluation_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Evaluation completata. Report: {outputs_dir / 'evaluation_summary.json'}")
    return results

