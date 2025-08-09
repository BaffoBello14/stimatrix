from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from dataset_builder.utils.logger import get_logger

logger = get_logger(__name__)


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    paths = config.get("paths", {})
    pre_dir = Path(paths.get("preprocessed_data", "data/preprocessed"))
    models_dir = Path(paths.get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    pre_filename = paths.get("preprocessed_filename", "preprocessed.parquet")
    pre_path = pre_dir / pre_filename
    if not pre_path.exists():
        raise FileNotFoundError(f"File preprocessed non trovato: {pre_path}")

    df = pd.read_parquet(pre_path)

    metrics = {
        "num_rows": int(len(df)),
        "num_cols": int(len(df.columns)),
    }
    model_path = models_dir / "model.info"
    model_path.write_text(str(metrics), encoding="utf-8")

    logger.info(f"Training completato. Metrics: {metrics}. Modello: {model_path}")
    return metrics