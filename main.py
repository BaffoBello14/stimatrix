from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import yaml

# Ensure 'src' is on sys.path when running directly
import sys as _sys
_src_path = str(Path(__file__).resolve().parent / "src")
if _src_path not in _sys.path:
    _sys.path.append(_src_path)

from db.schema_extract import main as schema_main
from dataset_builder.retrieval import retrieve_data
from preprocessing.pipeline import run_preprocessing
from training.train import run_training
from utils.logger import setup_logger


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


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_schema(config) -> None:
    # Reuse CLI entry of schema module by setting args
    # Simpler: call function directly replicating defaults
    from db.schema_extract import extract_schema, catch_unrecognized_types
    from db.connect import get_engine
    from utils.io import ensure_parent_dir

    # Attiva intercettazione SAWarning per tipi non riconosciuti
    catch_unrecognized_types()

    out = config.get("paths", {}).get("schema", "data/db_schema.json")
    ensure_parent_dir(out)

    schema_name = config.get("database", {}).get("schema_name", None)

    engine = get_engine()
    schema_dict = extract_schema(engine, schema_name=schema_name)
    import json

    with open(out, "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=4, ensure_ascii=False)


def run_dataset(config) -> None:
    db_cfg = config.get("database", {})
    paths = config.get("paths", {})

    schema_path = paths.get("schema", "data/db_schema.json")
    raw_dir = Path(paths.get("raw_data", "data/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_filename = paths.get("raw_filename", f"raw.{db_cfg.get('output_format', 'parquet')}")
    out_path = str(raw_dir / raw_filename)

    aliases = db_cfg.get("selected_aliases", [])
    include_poi = bool(db_cfg.get("use_poi", True))
    include_ztl = bool(db_cfg.get("use_ztl", True))

    # retrieval salva con formato interno; passiamo via config usando globali
    retrieve_data(
        schema_path=schema_path,
        selected_aliases=aliases,
        output_path=out_path,
        include_poi=include_poi,
        include_ztl=include_ztl,
        poi_categories=None,
    )


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