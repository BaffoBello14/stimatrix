#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Assicura che il repo root sia nel PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_builder import DatasetBuilder  # noqa: E402
from src.utils.io import load_config, load_dataframe, save_dataframe  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Costruisce il dataset di base da parquet raw")
    parser.add_argument("--config", default="config/config.yaml", help="Path al file di config")
    parser.add_argument("--input", default="data/raw/dataset.parquet", help="Path al parquet raw")
    parser.add_argument("--output", default="data/processed/dataset_base.parquet", help="Path output")
    return parser.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()

    config = load_config(args.config)
    df = load_dataframe(args.input)

    builder = DatasetBuilder(config)
    df_out, stats = builder.build_dataset(df)

    save_dataframe(df_out, args.output, format="parquet")
    print(f"OK: dataset costruito in {args.output} (shape={df_out.shape})")


if __name__ == "__main__":
    main()