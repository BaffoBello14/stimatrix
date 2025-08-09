import logging
from pathlib import Path
from typing import Optional

import yaml

_LOGGER_INITIALIZED = False


def setup_logger(config_path: str = "config/config.yaml") -> logging.Logger:
    global _LOGGER_INITIALIZED
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_config = config.get("logging", {})

    log_file = log_config.get("file", "logs/pipeline.log")
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    if not _LOGGER_INITIALIZED:
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        )
        _LOGGER_INITIALIZED = True

    logger = logging.getLogger("ML_Pipeline")
    logger.info("Logger inizializzato")
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "ML_Pipeline")