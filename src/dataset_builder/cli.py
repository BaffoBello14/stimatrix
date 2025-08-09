from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .retrieval import retrieve_data
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estrae dati dal DB e crea un raw dataset"
    )
    parser.add_argument(
        "--schema",
        type=str,
        required=True,
        help="Path al file schema JSON",
    )
    parser.add_argument(
        "--aliases",
        type=str,
        nargs="*",
        default=[],
        help="Lista alias da includere (es. A AI PC OV)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path file di output (es. data/raw.parquet)",
    )
    parser.add_argument(
        "--no-poi",
        action="store_true",
        help="Non includere conteggi dei punti di interesse",
    )
    parser.add_argument(
        "--no-ztl",
        action="store_true",
        help="Non includere informazione ZTL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.out)
    selected_aliases: List[str] = args.aliases if args.aliases else None

    retrieve_data(
        schema_path=args.schema,
        selected_aliases=selected_aliases,
        output_path=str(output_path),
        include_poi=not args.no_poi,
        include_ztl=not args.no_ztl,
        poi_categories=None,
    )


if __name__ == "__main__":
    main()