#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Assicura che il repo root sia nel PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.db.schema_extract import main  # noqa: E402


if __name__ == "__main__":
    main()