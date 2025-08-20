from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def ensure_parent_dir(path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def check_file_exists(path: str | os.PathLike) -> bool:
    return Path(path).exists()


def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON salvato: {path}")


def save_dataframe(
    df: pd.DataFrame, path: str, format: str = "parquet", compression: Optional[str] = None
) -> None:
    format = format.lower()
    ensure_parent_dir(path)
    if format == "parquet":
        kwargs = {}
        if compression:
            kwargs["compression"] = compression
        df.to_parquet(path, index=False, **kwargs)
        logger.info(f"DataFrame salvato come Parquet: {path}")
    elif format == "csv":
        kwargs = {}
        if compression in {"gzip", "bz2", "zip", "xz"}:
            kwargs["compression"] = compression
        df.to_csv(path, index=False, **kwargs)
        logger.info(f"DataFrame salvato come CSV: {path}")
    else:
        raise ValueError(f"Formato non supportato: {format}")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    return data