import argparse
import json
import re
import warnings
from collections import defaultdict
from typing import Dict, Any, Optional

from sqlalchemy import inspect
from sqlalchemy.exc import SAWarning

from utils.io import ensure_parent_dir
from db.connect import DatabaseConnector

unrecognized_types: Dict[str, str] = defaultdict(str)
SA_TYPE_REGEX = re.compile(r"Did not recognize type '(\w+)' of column '(\w+)'")


def catch_unrecognized_types() -> None:
    def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
        match = SA_TYPE_REGEX.search(str(message))
        if match:
            raw_type, col_name = match.groups()
            unrecognized_types[col_name] = raw_type.lower()

    warnings.simplefilter("always", SAWarning)
    warnings.showwarning = custom_warning_handler  # type: ignore[assignment]


def normalize_type(col_name: str, raw_type_str: str) -> str:
    # Preferisci il tipo catturato dai warning per la colonna
    if col_name in unrecognized_types and unrecognized_types[col_name]:
        return unrecognized_types[col_name]

    raw = str(raw_type_str).upper()
    if "GEOGRAPHY" in raw:
        return "geography"
    if "GEOMETRY" in raw:
        return "geometry"
    if raw == "SYSNAME":
        return "varchar"
    if "INT" in raw:
        return "integer"
    if any(t in raw for t in ["FLOAT", "REAL", "DECIMAL", "NUMERIC"]):
        return "float"
    if "NVARCHAR" in raw:
        return "nvarchar"
    if "VARCHAR" in raw:
        return "varchar"
    if "CHAR" in raw:
        return "char"
    if "DATE" in raw or "TIME" in raw or "DATETIME" in raw:
        return "datetime"
    if "MONEY" in raw:
        return "money"
    if "BINARY" in raw:
        return "binary"
    if "TEXT" in raw:
        return "text"
    if "BIT" in raw:
        return "boolean"
    if "UNIQUEIDENTIFIER" in raw:
        return "uuid"
    return "unknown"


def generate_table_alias(table_name: str, custom_aliases: Optional[Dict[str, str]] = None) -> str:
    """
    Generate alias for a table/view name.
    
    Args:
        table_name: Name of the table or view
        custom_aliases: Optional dictionary mapping table names to custom aliases
    
    Returns:
        Alias string (e.g., 'A', 'AI', 'C1', etc.)
    """
    # Check for custom alias first
    if custom_aliases and table_name in custom_aliases:
        return custom_aliases[table_name]
    
    # Default: extract uppercase letters or first letters
    parts = table_name.split("_")
    aliases = []
    for part in parts:
        uppercase_letters = "".join([c for c in part if c.isupper()])
        if uppercase_letters:
            aliases.append(uppercase_letters)
        else:
            aliases.append(part[0].upper())
    return "_".join(aliases)


def extract_schema(engine, schema_name: Optional[str] = None, custom_aliases: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Extract database schema including tables and views.
    
    Args:
        engine: SQLAlchemy engine
        schema_name: Database schema name (optional)
        custom_aliases: Dictionary mapping table/view names to custom aliases
    
    Returns:
        Dictionary containing schema information
    """
    inspector = inspect(engine)
    schema: Dict[str, Any] = {}
    
    # Define default custom aliases for common cases
    if custom_aliases is None:
        custom_aliases = {
            'attiimmobili_cened1': 'C1',
            'attiimmobili_cened2': 'C2',
        }

    # Extract tables
    for table_name in inspector.get_table_names(schema=schema_name):
        columns = inspector.get_columns(table_name, schema=schema_name)
        schema[table_name] = {
            "alias": generate_table_alias(table_name, custom_aliases),
            "type": "table",
            "columns": []
        }

        for col in columns:
            try:
                raw_type = str(col["type"])  # type: ignore[index]
                if raw_type.upper() in ("NULL", "NULLTYPE", "UNKNOWN"):
                    raw_type = col["type"].compile(dialect=engine.dialect)  # type: ignore[index]
            except Exception:
                raw_type = "unknown"

            normalized_type = normalize_type(str(col.get("name")), raw_type)

            schema[table_name]["columns"].append(
                {
                    "name": col.get("name"),
                    "type": normalized_type,
                    "retrieve": True,
                }
            )

    # Extract views (same as tables from a retrieval perspective)
    try:
        for view_name in inspector.get_view_names(schema=schema_name):
            columns = inspector.get_columns(view_name, schema=schema_name)
            schema[view_name] = {
                "alias": generate_table_alias(view_name, custom_aliases),
                "type": "view",
                "columns": []
            }

            for col in columns:
                try:
                    raw_type = str(col["type"])  # type: ignore[index]
                    if raw_type.upper() in ("NULL", "NULLTYPE", "UNKNOWN"):
                        raw_type = col["type"].compile(dialect=engine.dialect)  # type: ignore[index]
                except Exception:
                    raw_type = "unknown"

                normalized_type = normalize_type(str(col.get("name")), raw_type)

                schema[view_name]["columns"].append(
                    {
                        "name": col.get("name"),
                        "type": normalized_type,
                        "retrieve": True,
                    }
                )
    except Exception as e:
        # Some databases might not support views or have permission issues
        print(f"Warning: Could not extract views: {e}")

    return schema

def run_schema(config: Dict[str, Any]) -> None:
    catch_unrecognized_types()

    out = config.get("paths", {}).get("schema", "data/db_schema.json")
    ensure_parent_dir(out)

    schema_name = config.get("database", {}).get("schema_name", None)
    custom_aliases = config.get("database", {}).get("custom_aliases", None)

    engine = DatabaseConnector().engine
    schema_dict = extract_schema(engine, schema_name=schema_name, custom_aliases=custom_aliases)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=4, ensure_ascii=False)

