import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Carica la configurazione dal file YAML.

    Args:
        config_path: Path al file di configurazione

    Returns:
        Dizionario con la configurazione
    """
    import yaml

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configurazione caricata da {config_path}")
        return config
    except Exception as e:
        logger.error(f"Errore nel caricamento della configurazione: {e}")
        raise


def ensure_dir(path: Path | str) -> None:
    """
    Crea una directory se non esiste.

    Args:
        path: Path della directory da creare
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, filepath: str, format: str = "parquet", engine: Optional[str] = None) -> None:
    """
    Salva un DataFrame nel formato specificato.

    Args:
        df: DataFrame da salvare
        filepath: Path del file
        format: Formato ('parquet', 'csv', 'pickle')
        engine: Engine parquet opzionale (es. 'pyarrow')
    """
    ensure_dir(Path(filepath).parent)

    if format == "parquet":
        # Preferisci pyarrow se disponibile
        try:
            df.to_parquet(filepath, index=False, engine=engine or "pyarrow")
        except Exception:
            # Fallback a default pandas
            df.to_parquet(filepath, index=False)
    elif format == "csv":
        df.to_csv(filepath, index=False)
    elif format == "pickle":
        df.to_pickle(filepath)
    else:
        raise ValueError(f"Formato non supportato: {format}")

    logger.info(f"DataFrame salvato: {filepath} (formato: {format}, shape: {df.shape})")


essential_extensions = {".parquet", ".csv", ".pkl", ".pickle"}


def load_dataframe(filepath: str, format: str | None = None) -> pd.DataFrame:
    """
    Carica un DataFrame dal file specificato.

    Args:
        filepath: Path del file
        format: Formato ('parquet', 'csv', 'pickle'). Se None, inferito dall'estensione

    Returns:
        DataFrame caricato
    """
    path = Path(filepath)
    if format is None:
        format = path.suffix[1:]  # Rimuove il punto

    if format == "parquet":
        df = pd.read_parquet(filepath)
    elif format == "csv":
        df = pd.read_csv(filepath)
    elif format in ["pickle", "pkl"]:
        df = pd.read_pickle(filepath)
    else:
        raise ValueError(f"Formato non supportato: {format}")

    logger.info(f"DataFrame caricato: {filepath} (shape: {df.shape})")
    return df


def save_model(model: Any, filepath: str) -> None:
    """
    Salva un modello usando pickle.

    Args:
        model: Modello da salvare
        filepath: Path del file
    """
    ensure_dir(Path(filepath).parent)

    with open(filepath, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Modello salvato: {filepath}")


def load_model(filepath: str) -> Any:
    """
    Carica un modello da file pickle.

    Args:
        filepath: Path del file

    Returns:
        Modello caricato
    """
    with open(filepath, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Modello caricato: {filepath}")
    return model


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Salva un dizionario in formato JSON.

    Args:
        data: Dizionario da salvare
        filepath: Path del file
    """
    ensure_dir(Path(filepath).parent)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"JSON salvato: {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Carica un dizionario da file JSON.

    Args:
        filepath: Path del file

    Returns:
        Dizionario caricato
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"JSON caricato: {filepath}")
    return data


def check_file_exists(filepath: str) -> bool:
    """
    Verifica se un file esiste.

    Args:
        filepath: Path del file

    Returns:
        True se il file esiste
    """
    return Path(filepath).exists()