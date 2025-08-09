from __future__ import annotations

import os
import urllib.parse

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Carica variabili d'ambiente da .env se presente (override per evitare conflitti con USER di sistema)
load_dotenv(override=True)


def _get_env(var_name: str) -> str | None:
    value = os.getenv(var_name)
    return value.strip() if isinstance(value, str) else value


def get_engine() -> Engine:
    """
    Crea e ritorna un engine SQLAlchemy per la connessione al database.

    Returns:
        Engine SQLAlchemy configurato

    Raises:
        ValueError: Se le variabili di ambiente non sono configurate
        Exception: Se la connessione non può essere stabilita
    """
    # Recupera variabili d'ambiente (preferisce DB_USER/DB_PASSWORD per evitare collisioni)
    server = _get_env("SERVER")
    database = _get_env("DATABASE")
    user = _get_env("DB_USER") or _get_env("USER")
    password = _get_env("DB_PASSWORD") or _get_env("PASSWORD")

    # Verifica che tutte le variabili siano presenti
    if not all([server, database, user, password]):
        missing = [
            var
            for var, val in (
                ("SERVER", server),
                ("DATABASE", database),
                ("DB_USER/USER", user),
                ("DB_PASSWORD/PASSWORD", password),
            )
            if not val
        ]
        raise ValueError(f"Variabili d'ambiente mancanti: {missing}")

    # Codifica credenziali per URL
    user_encoded = urllib.parse.quote_plus(user)
    password_encoded = urllib.parse.quote_plus(password)

    # Costruisce connection string MSSQL via pyodbc
    connection_string = (
        f"mssql+pyodbc://{user_encoded}:{password_encoded}@{server}/"
        f"{database}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&"
        f"TrustServerCertificate=no&Connection+Timeout=30"
    )

    # Crea engine e testa la connessione
    engine = create_engine(connection_string)

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        logger.info(
            f"Connessione al database '{database}' su '{server}' stabilita con successo"
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
            result = conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            test_value = row[0] if row is not None else None
            if test_value == 1:
                logger.info("Test connessione database: SUCCESSO")
                return True
            logger.error("Test connessione database: FALLITO - Valore inaspettato")
            return False
    except Exception as exc:
        logger.error(f"Test connessione database: FALLITO - {exc}")
        return False