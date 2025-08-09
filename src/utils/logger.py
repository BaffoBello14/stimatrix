import logging
from pathlib import Path
from typing import Optional
import yaml


def setup_logger(config_path: str = "config/config.yaml") -> logging.Logger:
    """
    Setup del logger basato sulla configurazione YAML, con fallback a default sensati.

    Args:
        config_path: Path al file di configurazione

    Returns:
        Logger configurato (root 'ML_Pipeline')
    """
    log_config_default = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "logs/ml_pipeline.log",
    }

    config = {}
    try:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
    except Exception:
        # Se la config non è leggibile, useremo i default
        config = {}

    log_config = (config.get("logging") or {})
    level = getattr(logging, str(log_config.get("level", log_config_default["level"])) .upper(), logging.INFO)
    fmt = str(log_config.get("format", log_config_default["format"]))
    log_file = str(log_config.get("file", log_config_default["file"]))

    # Crea directory logs se non esiste
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Evita di aggiungere handler duplicati se setup_logger è chiamato più volte
    logger = logging.getLogger("ML_Pipeline")
    logger.setLevel(level)

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(file_handler)

        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(stream_handler)

        logger.propagate = False
        logger.info("Logger inizializzato")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Ottiene un logger con il nome specificato. Se non è stato configurato, esegue un setup minimo di default.

    Args:
        name: Nome del logger (usa 'ML_Pipeline' come default)

    Returns:
        Logger
    """
    base_logger_name = name or "ML_Pipeline"
    logger = logging.getLogger(base_logger_name)

    if not logger.handlers and base_logger_name == "ML_Pipeline":
        # Setup minimo se non inizializzato altrove
        setup_logger()
        logger = logging.getLogger(base_logger_name)

    return logger