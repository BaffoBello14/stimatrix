from __future__ import annotations

import os
import urllib.parse

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Carica variabili d'ambiente da .env se presente
load_dotenv()


def _get_env(name: str, fallback: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is not None:
        return value
    return fallback


def get_engine() -> Engine:
    """
    Crea e ritorna un engine SQLAlchemy per la connessione al database SQL Server.

    Env supportate (preferite): DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD
    Compatibilità (deprecata): SERVER, DATABASE, USER, PASSWORD

    Returns:
        Engine SQLAlchemy configurato

    Raises:
        ValueError: Se le variabili di ambiente non sono configurate
        Exception: Se la connessione non può essere stabilita
    """
    # Preferisci prefisso DB_ ma consenti compatibilità
    server = _get_env("DB_SERVER", _get_env("SERVER"))
    database = _get_env("DB_NAME", _get_env("DATABASE"))
    user = _get_env("DB_USER", _get_env("USER"))
    password = _get_env("DB_PASSWORD", _get_env("PASSWORD"))

    # Avviso su variabili deprecate
    if os.getenv("SERVER") or os.getenv("DATABASE") or os.getenv("USER") or os.getenv("PASSWORD"):
        logger.warning(
            "Le variabili SERVER/DATABASE/USER/PASSWORD sono deprecate. Usa DB_SERVER/DB_NAME/DB_USER/DB_PASSWORD."
        )

    # Verifica che tutte le variabili siano presenti
    missing = [
        var
        for var, val in [
            ("DB_SERVER", server),
            ("DB_NAME", database),
            ("DB_USER", user),
            ("DB_PASSWORD", password),
        ]
        if not val
    ]
    if missing:
        raise ValueError(f"Variabili d'ambiente mancanti: {missing}")

    # Codifica credenziali per URL
    user_encoded = urllib.parse.quote_plus(user)
    password_encoded = urllib.parse.quote_plus(password)

    # Connection string per SQL Server tramite ODBC Driver 18
    # Nota: assicurarsi che l'ODBC driver sia installato nel sistema
    connection_string = (
        f"mssql+pyodbc://{user_encoded}:{password_encoded}@{server}/"
        f"{database}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&"
        f"TrustServerCertificate=yes&Connection+Timeout=30"
    )

    engine = create_engine(connection_string, pool_pre_ping=True)

    # Test connessione
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1 AS test"))
        value = result.scalar_one()
        logger.info(
            f"Connessione al database '{database}' su '{server}' stabilita con successo (test={value})"
        )

    return engine


def test_connection() -> bool:
    """
    Testa la connessione al database.

    Returns:
        True se la connessione è riuscita, False altrimenti
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 AS test"))
            test_value = result.scalar_one()
            if test_value == 1:
                logger.info("Test connessione database: SUCCESSO")
                return True
            logger.error("Test connessione database: FALLITO - Valore inaspettato")
            return False
    except Exception as e:
        logger.error(f"Test connessione database: FALLITO - {e}")
        return False