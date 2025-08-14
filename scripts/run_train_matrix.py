#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def slug(s: str) -> str:
    return (
        str(s)
        .replace("/", "-")
        .replace(" ", "_")
        .replace("=", "-")
        .replace(",", "_")
        .replace("[", "").replace("]", "")
        .replace("(", "").replace(")", "")
    )


def variant_matrix() -> Iterable[Dict[str, Any]]:
    # Keep the grid compact but representative
    split_modes: List[Tuple[str, Dict[str, Any]]] = [
        ("splitFraction_0.8_val0.1", {"mode": "fraction", "train_fraction": 0.8, "valid_fraction": 0.1}),
        ("splitDate_2023-01_val0.1", {"mode": "date", "test_start_year": 2023, "test_start_month": 1, "valid_fraction": 0.1}),
    ]
    outlier_sets: List[Tuple[str, Dict[str, Any]]] = [
        ("out_iqr", {"method": "iqr"}),
        ("out_iso002", {"method": "iso_forest", "iso_forest_contamination": 0.02}),
        ("out_ens", {"method": "ensemble"}),
    ]
    enc_sets: List[Tuple[str, Dict[str, Any]]] = [
        ("ohe8_corr095", {"encoding.max_ohe": 8, "correlation.numeric_threshold": 0.95}),
        ("ohe12_corr098", {"encoding.max_ohe": 12, "correlation.numeric_threshold": 0.98}),
    ]
    train_sets: List[Tuple[str, Dict[str, Any]]] = [
        ("metric_r2_seed42", {"primary_metric": "r2", "seed": 42}),
        ("metric_rmse_seed123", {"primary_metric": "neg_root_mean_squared_error", "seed": 123}),
    ]

    for split_name, split_cfg in split_modes:
        for out_name, out_cfg in outlier_sets:
            for enc_name, enc_cfg in enc_sets:
                for tr_name, tr_cfg in train_sets:
                    name_parts = [split_name, out_name, enc_name, tr_name]
                    yield {
                        "name": "__".join(name_parts),
                        "temporal_split": split_cfg,
                        "outliers": out_cfg,
                        "encoding": enc_cfg,
                        "training": tr_cfg,
                    }


def apply_variant(base: Dict[str, Any], var: Dict[str, Any], out_root: Path, run_name: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)

    # Temporal split
    ts = cfg.setdefault("temporal_split", {})
    ts.update(var.get("temporal_split", {}))

    # Outliers
    out = cfg.setdefault("outliers", {})
    out.update(var.get("outliers", {}))
    # Ensure reproducibility seed propagated
    out.setdefault("random_state", int(cfg.get("training", {}).get("seed", 42)))
    out.setdefault("group_by_col", "AI_IdCategoriaCatastale")
    out.setdefault("min_group_size", 30)
    out.setdefault("fallback_strategy", "global")

    # Encoding & correlation thresholds applied to both global and per-profile
    enc_global = cfg.setdefault("encoding", {})
    max_ohe = int(var.get("encoding", {}).get("encoding.max_ohe", enc_global.get("max_ohe_cardinality", 12)))
    enc_global["max_ohe_cardinality"] = max_ohe

    corr_thr = float(var.get("encoding", {}).get("correlation.numeric_threshold", cfg.get("correlation", {}).get("numeric_threshold", 0.98)))
    cfg.setdefault("correlation", {})["numeric_threshold"] = corr_thr

    # Propagate to profiles when present
    profiles = cfg.setdefault("profiles", {})
    for prof_key in ("scaled", "tree"):
        prof = profiles.setdefault(prof_key, {})
        prof.setdefault("encoding", {})["max_ohe_cardinality"] = max_ohe
        prof.setdefault("correlation", {})["numeric_threshold"] = corr_thr

    # Training
    tr = cfg.setdefault("training", {})
    tr["primary_metric"] = var.get("training", {}).get("primary_metric", tr.get("primary_metric", "r2"))
    tr["seed"] = int(var.get("training", {}).get("seed", tr.get("seed", 42)))
    # Disable heavy extras by default for sweep
    tr.setdefault("shap", {})["enabled"] = False
    tr.setdefault("ensembles", {"voting": {"enabled": False}, "stacking": {"enabled": False}})
    tr["ensembles"] = {"voting": {"enabled": False}, "stacking": {"enabled": False}}

    # Isolate outputs per run
    paths = cfg.setdefault("paths", {})
    paths["preprocessed_data"] = str(out_root / "preprocessed" / run_name)
    paths["models_dir"] = str(out_root / "models" / run_name)

    return cfg


def run_once(main_py: Path, cfg_path: Path, steps: List[str]) -> int:
    cmd = [sys.executable, str(main_py), "--config", str(cfg_path), "--steps", *steps]
    proc = subprocess.run(cmd)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a matrix of training configurations")
    p.add_argument("--base-config", type=str, default="config/config_fast_test.yaml", help="Base YAML config to start from")
    p.add_argument("--output-root", type=str, default="runs", help="Root directory to store per-run outputs")
    p.add_argument("--steps", type=str, nargs="+", default=["preprocessing", "training"], choices=["schema", "dataset", "preprocessing", "training"], help="Pipeline steps to execute")
    p.add_argument("--limit", type=int, default=0, help="Optional limit to number of runs (0 = no limit)")
    p.add_argument("--dry-run", action="store_true", help="Only generate configs, do not execute")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    main_py = repo_root / "main.py"
    base_cfg = load_yaml(Path(args.base_config))
    out_root = Path(args.output_root)

    runs: List[Tuple[str, Path]] = []

    for i, var in enumerate(variant_matrix(), start=1):
        if args.limit and i > args.limit:
            break
        run_name = slug(var["name"]) \
            + f"__{slug(base_cfg.get('training', {}).get('sampler', 'tpe'))}"
        out_cfg = apply_variant(base_cfg, var, out_root, run_name)
        cfg_path = out_root / "configs" / f"{run_name}.yaml"
        dump_yaml(out_cfg, cfg_path)
        runs.append((run_name, cfg_path))

    print(f"Prepared {len(runs)} run configs under {out_root}/configs")

    if args.dry_run:
        for name, path in runs:
            print(f"DRY-RUN: {name} -> {path}")
        return

    failures = 0
    for name, path in runs:
        print(f"\n=== Running {name} ===")
        code = run_once(main_py, path, args.steps)
        if code != 0:
            print(f"Run {name} failed with exit code {code}")
            failures += 1
        else:
            print(f"Run {name} completed successfully")

    if failures:
        print(f"\nCompleted with {failures} failures out of {len(runs)} runs")
        sys.exit(1)
    else:
        print(f"\nAll {len(runs)} runs completed successfully")


if __name__ == "__main__":
    main()