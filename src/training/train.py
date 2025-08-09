from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    paths = config.get("paths", {})
    processed_dir = Path(paths.get("processed_data", "data/processed"))
    processed_path = processed_dir / "processed.parquet"

    if not processed_path.exists():
        raise FileNotFoundError(f"File processed non trovato: {processed_path}")

    df = pd.read_parquet(processed_path)

    # Placeholder: calcolo metrica fittizia
    metrics = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
    }
    logger.info(f"Training completato. Metrics: {metrics}")
    return metrics