#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Assicura che il repo root sia nel PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.retrieval import retrieve_data  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recupera dati dal DB e salva parquet raw")
    parser.add_argument("--schema", required=True, help="Path allo schema JSON estratto")
    parser.add_argument(
        "--aliases",
        nargs="+",
        required=True,
        help="Alias di tabelle da includere (es. A AI PC OV OZ ISC II)",
    )
    parser.add_argument(
        "--output",
        default="data/raw/dataset.parquet",
        help="Percorso di output parquet",
    )
    parser.add_argument("--no-poi", action="store_true", help="Non includere conteggi POI")
    parser.add_argument("--no-ztl", action="store_true", help="Non includere info ZTL")
    return parser.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()

    df = retrieve_data(
        schema_path=args.schema,
        selected_aliases=args.aliases,
        output_path=args.output,
        include_poi=not args.no_poi,
        include_ztl=not args.no_ztl,
    )
    print(f"OK: salvate {len(df)} righe in {args.output}")


if __name__ == "__main__":
    main()