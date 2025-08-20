from __future__ import annotations

from typing import Any, Dict
import os
import re

import yaml


_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def _expand_env_value(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            var = match.group(1)
            default = match.group(2)
            if default is None:
                return os.environ.get(var, "")
            # Strip matching quotes around default if present
            d = default.strip()
            if (d.startswith("'") and d.endswith("'")) or (d.startswith('"') and d.endswith('"')):
                d = d[1:-1]
            return os.environ.get(var, d)

        expanded = _ENV_PATTERN.sub(repl, value)
        if expanded.strip().lower() in {"null", "~"}:
            return None
        return expanded
    if isinstance(value, dict):
        return {k: _expand_env_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_value(v) for v in value]
    return value


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return _expand_env_value(loaded)