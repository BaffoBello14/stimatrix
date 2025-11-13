from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

# Ensure 'src' is on sys.path when running directly
import sys as _sys
_src_path = str(Path(__file__).resolve().parent / "src")
if _src_path not in _sys.path:
    _sys.path.append(_src_path)

from utils.config import load_config
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stimatrix pipeline orchestrator")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path al config YAML")
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        choices=["schema", "dataset", "preprocessing", "training", "evaluation", "ml", "all"],
        required=False,
        help="Passi da eseguire (uno o più). Usa 'all' per eseguire tutti i passi o 'ml' per eseguire solo preprocessing, training ed evaluation.",
    )
    args = parser.parse_args()

    # Mapping speciale per shorthand
    if args.config.strip().lower() == "fast":
        args.config = "config/config_fast.yaml"

    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Ensure execution section exists
    execution_cfg = config.setdefault("execution", {})

    # Initialize logging according to config
    setup_logger(args.config)

    # Steps can come from CLI or from config.execution.steps
    steps: List[str] = args.steps or []
    if not steps:
        cfg_steps = execution_cfg.get("steps", [])
        if isinstance(cfg_steps, str):
            cfg_steps = [cfg_steps]
        steps = list(cfg_steps)
    if not steps:
        print("Seleziona i passi da eseguire separati da spazio (schema, dataset, preprocessing, training, evaluation, all):")
        user_input = input().strip()
        steps = user_input.split()
    if "all" in steps:
        steps = ["schema", "dataset", "preprocessing", "training", "evaluation"]
    if "ml" in steps:
        steps = ["preprocessing", "training", "evaluation"]

    for step in steps:
        if step == "schema":
            from db.schema_extract import run_schema  # lazy import
            run_schema(config)
        elif step == "dataset":
            from dataset_builder.retrieval import run_dataset  # lazy import
            run_dataset(config)
        elif step == "preprocessing":
            from preprocessing.pipeline import run_preprocessing  # lazy import
            run_preprocessing(config)
        elif step == "training":
            from training.train import run_training  # lazy import
            run_training(config)
        elif step == "evaluation":
            from training.evaluation import run_evaluation  # lazy import
            run_evaluation(config)


if __name__ == "__main__":
    main()
