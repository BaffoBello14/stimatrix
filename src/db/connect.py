from __future__ import annotations

import os
import urllib.parse

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from utils.logger import get_logger

logger = get_logger(__name__)

# Carica variabili d'ambiente da .env se presente (override per evitare conflitti con USER di sistema)
load_dotenv(override=True)


def _get_env(var_name: str) -> str | None:
    value = os.getenv(var_name)
    return value.strip() if isinstance(value, str) else value


def get_engine() -> Engine:
    """
    Crea e ritorna un engine SQLAlchemy per la connessione al database.
    """
    server = _get_env("SERVER")
    database = _get_env("DATABASE")
    user = _get_env("DB_USER") or _get_env("USER")
    password = _get_env("DB_PASSWORD") or _get_env("PASSWORD")

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

    user_encoded = urllib.parse.quote_plus(user)
    password_encoded = urllib.parse.quote_plus(password)

    connection_string = (
        f"mssql+pyodbc://{user_encoded}:{password_encoded}@{server}/"
        f"{database}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&"
        f"TrustServerCertificate=no&Connection+Timeout=30"
    )

    engine = create_engine(connection_string)

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        logger.info(
            f"Connessione al database '{database}' su '{server}' stabilita con successo"
        )

    return engine


def test_connection() -> bool:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            test_value = row[0] if row is not None else None
            return test_value == 1
    except Exception as exc:
        logger.error(f"Test connessione database: FALLITO - {exc}")
        return False