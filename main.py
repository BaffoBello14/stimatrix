from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

# Ensure 'src' is on sys.path when running directly
import sys as _sys
_src_path = str(Path(__file__).resolve().parent / "src")
if _src_path not in _sys.path:
    _sys.path.append(_src_path)

from dataset_builder.retrieval import run_dataset
from preprocessing.pipeline import run_preprocessing
from training.train import run_training
from utils.config import load_config
from utils.logger import setup_logger
from db.schema_extract import run_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stimatrix pipeline orchestrator")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path al config YAML")
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        choices=["schema", "dataset", "preprocessing", "training", "all"],
        required=False,
        help="Passi da eseguire (uno o più). Usa 'all' per tutti",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Initialize logging according to config
    setup_logger(args.config)

    steps: List[str] = args.steps or []
    if not steps:
        print("Seleziona i passi da eseguire separati da spazio (schema, dataset, preprocessing, training, all):")
        user_input = input().strip()
        steps = user_input.split()
    if "all" in steps:
        steps = ["schema", "dataset", "preprocessing", "training"]

    for step in steps:
        if step == "schema":
            run_schema(config)
        elif step == "dataset":
            run_dataset(config)
        elif step == "preprocessing":
            run_preprocessing(config)
        elif step == "training":
            run_training(config)


if __name__ == "__main__":
    main()