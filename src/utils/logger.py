import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .config import load_config

# Add optional Optuna import
try:
    import optuna  # type: ignore
except Exception:  # pragma: no cover
    optuna = None

_LOGGER_INITIALIZED = False


def setup_logger(config_path: str = "config/config.yaml") -> logging.Logger:
    global _LOGGER_INITIALIZED
    config = load_config(config_path)

    log_config = config.get("logging", {})

    def _as_bool(value, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "yes", "on"}:
                return True
            if v in {"0", "false", "no", "off", ""}:
                return False
        return default

    level_name = str(log_config.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = log_config.get("file", "logs/pipeline.log")
    console_enabled = _as_bool(log_config.get("console", True), True)
    rotate_cfg = log_config.get("rotate", {"enabled": False})

    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    if not _LOGGER_INITIALIZED:
        handlers: list[logging.Handler] = []
        if _as_bool(rotate_cfg.get("enabled", False), False):
            max_bytes = int(rotate_cfg.get("max_bytes", 10 * 1024 * 1024))
            backup_count = int(rotate_cfg.get("backup_count", 5))
            handlers.append(RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"))
        else:
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        if console_enabled:
            handlers.append(logging.StreamHandler())

        logging.basicConfig(level=level, format=log_format, handlers=handlers)

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