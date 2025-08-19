import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import yaml

# Add optional Optuna import
try:
    import optuna  # type: ignore
except Exception:  # pragma: no cover
    optuna = None

_LOGGER_INITIALIZED = False


def setup_logger(config_path: str = "config/config.yaml") -> logging.Logger:
    global _LOGGER_INITIALIZED
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_config = config.get("logging", {})

    log_file = log_config.get("file", "logs/pipeline.log")
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    file_output = bool(log_config.get("file_output", True))
    console_output = bool(log_config.get("console_output", True))
    max_file_size = str(log_config.get("max_file_size", "0"))  # e.g., "10MB" or "0" (no rotation)
    backup_count = int(log_config.get("backup_count", 0))

    def _parse_size(expr: str) -> int:
        try:
            s = expr.strip().upper()
            if s.endswith("KB"):
                return int(float(s[:-2]) * 1024)
            if s.endswith("MB"):
                return int(float(s[:-2]) * 1024 * 1024)
            if s.endswith("GB"):
                return int(float(s[:-2]) * 1024 * 1024 * 1024)
            return int(float(s))
        except Exception:
            return 0

    if not _LOGGER_INITIALIZED:
        handlers: list[logging.Handler] = []
        level = getattr(logging, log_config.get("level", "INFO"))
        fmt = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        if file_output:
            rotate_bytes = _parse_size(max_file_size)
            if rotate_bytes and backup_count > 0:
                fh = RotatingFileHandler(log_file, maxBytes=rotate_bytes, backupCount=backup_count, encoding="utf-8")
            else:
                fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(fmt))
            handlers.append(fh)

        if console_output:
            sh = logging.StreamHandler()
            sh.setLevel(level)
            sh.setFormatter(logging.Formatter(fmt))
            handlers.append(sh)

        logging.basicConfig(level=level, handlers=handlers)
        # Configure Optuna logging to propagate into Python logging and avoid duplicate default handler
        if optuna is not None:
            try:
                optuna.logging.enable_propagation()
                optuna.logging.set_verbosity(optuna.logging.INFO)
                optuna.logging.disable_default_handler()
            except Exception:
                pass
        _LOGGER_INITIALIZED = True

    logger = logging.getLogger("ML_Pipeline")
    logger.info("Logger inizializzato")
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "ML_Pipeline")