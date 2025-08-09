from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def run_preprocessing(config: Dict[str, Any]) -> Path:
    paths = config.get("paths", {})
    raw_dir = Path(paths.get("raw_data", "data/raw"))
    processed_dir = Path(paths.get("processed_data", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Semplice esempio: carica il primo parquet nel raw_dir e salva come processed
    raw_files = list(raw_dir.glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"Nessun file parquet trovato in {raw_dir}")

    df = pd.read_parquet(raw_files[0])

    # Qui inserire i veri passi di preprocessing
    # Esempio: drop colonne completamente vuote
    df = df.dropna(axis=1, how="all")

    out_path = processed_dir / "processed.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Preprocessing completato: {out_path}")
    return out_path