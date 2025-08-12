from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from utils.logger import get_logger
from utils.security import SecureCredentialManager, InputValidator, audit_log, SecurityError

logger = get_logger(__name__)

load_dotenv()


def _get_env(var_name: str, credential_manager: Optional[SecureCredentialManager] = None) -> str | None:
    """Get environment variable with optional security validation."""
    # Validate variable name for security
    try:
        InputValidator.sanitize_column_name(var_name)
    except SecurityError:
        logger.error(f"Invalid environment variable name: {var_name}")
        return None
    
    if credential_manager:
        value = credential_manager.get_secure_env_var(var_name, encrypted=False)
    else:
        value = os.getenv(var_name)
    
    return value.strip() if isinstance(value, str) else value


@dataclass
class DatabaseConfig:
    server: str
    database: str
    user: str
    password: str

    @staticmethod
    def from_env(use_secure_manager: bool = True) -> "DatabaseConfig":
        """Create database config from environment variables with enhanced security."""
        credential_manager = SecureCredentialManager() if use_secure_manager else None
        
        # Audit database configuration access
        audit_log("database_config_access", {"source": "environment_variables"})
        
        server = _get_env("SERVER", credential_manager)
        database = _get_env("DATABASE", credential_manager)
        user = _get_env("DB_USER", credential_manager)
        password = _get_env("DB_PASSWORD", credential_manager)
        
        # Validate all required variables are present
        missing = [
            var
            for var, val in (
                ("SERVER", server),
                ("DATABASE", database),
                ("DB_USER", user),
                ("DB_PASSWORD", password),
            )
            if not val
        ]
        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            raise ValueError(f"Variabili d'ambiente mancanti: {missing}")
        
        # Additional validation for server and database names
        try:
            InputValidator.validate_sql_input(server)
            InputValidator.validate_sql_input(database)
            InputValidator.validate_sql_input(user)
        except SecurityError as e:
            logger.error(f"Security validation failed for database config: {e}")
            raise ValueError(f"Invalid database configuration: {e}")
        
        return DatabaseConfig(server=server, database=database, user=user, password=password)


class DatabaseConnector:
    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self.config = config or DatabaseConfig.from_env()
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    def _create_engine(self) -> Engine:
        """Create database engine with enhanced security measures."""
        try:
            # Audit database connection attempt
            audit_log("database_connection_attempt", {
                "server": self.config.server,
                "database": self.config.database,
                "user": self.config.user
            })
            
            # URL encode credentials to prevent injection
            user_encoded = urllib.parse.quote_plus(self.config.user)
            password_encoded = urllib.parse.quote_plus(self.config.password)
            
            # Build connection string with security options
            connection_string = (
                f"mssql+pyodbc://{user_encoded}:{password_encoded}@{self.config.server}/"
                f"{self.config.database}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&"
                f"TrustServerCertificate=no&Connection+Timeout=30&"
                f"ApplicationIntent=ReadWrite&ConnectRetryCount=3&ConnectRetryInterval=10"
            )
            
            # Create engine with security configurations
            engine = create_engine(
                connection_string,
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=3600,   # Recycle connections after 1 hour
                echo=False,          # Don't echo SQL statements in logs
                future=True          # Use SQLAlchemy 2.0 style
            )
            
            # Test connection with a simple query
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as connection_test"))
                test_value = result.scalar()
                if test_value != 1:
                    raise Exception("Connection test failed: unexpected result")
                
                # Log successful connection (without sensitive info)
                logger.info(
                    f"Connessione al database '{self.config.database}' su '{self.config.server}' stabilita con successo"
                )
                
                # Audit successful connection
                audit_log("database_connection_success", {
                    "server": self.config.server,
                    "database": self.config.database
                })
            
            return engine
            
        except Exception as exc:
            # Audit failed connection
            audit_log("database_connection_failure", {
                "server": self.config.server,
                "database": self.config.database,
                "error_type": type(exc).__name__
            })
            
            logger.error(f"Errore durante la creazione della connessione database: {type(exc).__name__}")
            raise

    def test_connection(self) -> bool:
        """Test database connection with enhanced error handling."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                row = result.fetchone()
                test_value = row[0] if row is not None else None
                
                if test_value == 1:
                    audit_log("database_test_success", {"result": "connection_healthy"})
                    return True
                else:
                    audit_log("database_test_failure", {"result": "unexpected_test_value"})
                    return False
                    
        except Exception as exc:
            # Log error without exposing sensitive details
            logger.error(f"Test connessione database: FALLITO - {type(exc).__name__}")
            audit_log("database_test_failure", {
                "error_type": type(exc).__name__,
                "result": "connection_failed"
            })
            return False

    def execute_safe_query(self, query: str, parameters: dict = None) -> any:
        """Execute a query with input validation and audit logging."""
        try:
            # Validate query for potential SQL injection
            InputValidator.validate_sql_input(query)
            
            # Audit query execution
            audit_log("database_query_execution", {
                "query_type": "SELECT" if query.strip().upper().startswith("SELECT") else "OTHER",
                "has_parameters": parameters is not None
            })
            
            with self.engine.connect() as conn:
                if parameters:
                    result = conn.execute(text(query), parameters)
                else:
                    result = conn.execute(text(query))
                return result
                
        except SecurityError as e:
            logger.error(f"Query validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Query execution failed: {type(e).__name__}")
            audit_log("database_query_failure", {"error_type": type(e).__name__})
            raise
