#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

# Make 'src' importable like in main.py
_src_path = Path(__file__).resolve().parent / "src"
if str(_src_path) not in sys.path:
    sys.path.append(str(_src_path))

from inference.show_samples import main as _main


if __name__ == "__main__":
    _main()

