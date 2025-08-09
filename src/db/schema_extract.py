from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import defaultdict
from typing import Dict, List, Optional

from sqlalchemy import inspect
from sqlalchemy.exc import SAWarning

from .connect import get_engine
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Dizionario per raccogliere tipi personalizzati rilevati dai warning
unrecognized_types = defaultdict(str)

# Regex per estrarre tipo non riconosciuto da SAWarning
SA_TYPE_REGEX = re.compile(r"Did not recognize type '(\w+)' of column '(\w+)'")


def catch_unrecognized_types() -> None:
    """Intercetta warning SAWarning per registrare tipi non riconosciuti per colonne.

    Nota: questa funzione modifica temporaneamente warnings.showwarning.
    """

    def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
        match = SA_TYPE_REGEX.search(str(message))
        if match:
            raw_type, col_name = match.groups()
            unrecognized_types[col_name] = raw_type.lower()

    warnings.simplefilter("always", SAWarning)
    warnings.showwarning = custom_warning_handler  # type: ignore[assignment]


def normalize_type(col_name: str, raw_type_str: str) -> str:
    # Usa tipo rilevato da warning se presente
    if col_name in unrecognized_types:
        return unrecognized_types[col_name]

    raw = (raw_type_str or "").upper()
    if "INT" in raw:
        return "integer"
    if any(t in raw for t in ["FLOAT", "REAL", "DECIMAL", "NUMERIC"]):
        return "float"
    if "GEOMETRY" in raw:
        return "geometry"
    if "GEOGRAPHY" in raw:
        return "geography"
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


def generate_table_alias(table_name: str) -> str:
    parts = table_name.split("_")
    aliases: List[str] = []
    for part in parts:
        uppercase_letters = "".join([c for c in part if c.isupper()])
        if uppercase_letters:
            aliases.append(uppercase_letters)
        else:
            aliases.append(part[0].upper())
    return "_".join(aliases)


def extract_schema(engine, schema_name: Optional[str] = None, include_spatial: bool = False) -> Dict[str, dict]:
    inspector = inspect(engine)
    schema: Dict[str, dict] = {}

    # Gestione alias univoci
    used_aliases: Dict[str, int] = {}

    # Backup warning handler
    original_showwarning = warnings.showwarning
    try:
        catch_unrecognized_types()

        for table_name in inspector.get_table_names(schema=schema_name):
            base_alias = generate_table_alias(table_name)
            # Assicura univocità
            if base_alias in used_aliases:
                used_aliases[base_alias] += 1
                alias = f"{base_alias}{used_aliases[base_alias]}"
            else:
                used_aliases[base_alias] = 1
                alias = base_alias

            columns = inspector.get_columns(table_name, schema=schema_name)
            schema[table_name] = {"alias": alias, "columns": []}

            for col in columns:
                try:
                    raw_type = str(col.get("type"))
                    if raw_type and raw_type.upper() in ("NULL", "NULLTYPE", "UNKNOWN"):
                        raw_type = col["type"].compile(dialect=engine.dialect)  # type: ignore[index]
                except Exception:
                    raw_type = "unknown"

                normalized_type = normalize_type(col["name"], raw_type)

                retrieve = True
                if normalized_type in ("geometry", "geography") and not include_spatial:
                    retrieve = False

                schema[table_name]["columns"].append(
                    {
                        "name": col["name"],
                        "type": normalized_type,
                        "retrieve": retrieve,
                    }
                )
    finally:
        warnings.showwarning = original_showwarning  # ripristina handler

    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Estrai schema DB e salvalo in JSON")
    parser.add_argument(
        "--output", type=str, default="data/db_schema.json", help="Path file JSON di output"
    )
    parser.add_argument("--schema", type=str, default=None, help="Schema DB (opzionale)")
    parser.add_argument(
        "--include-spatial",
        action="store_true",
        help="Includi colonne geometry/geography convertite in WKT",
    )
    args = parser.parse_args()

    engine = get_engine()
    schema_dict = extract_schema(engine, schema_name=args.schema, include_spatial=args.include_spatial)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=4, ensure_ascii=False)

    logger.info(f"Schema esportato in {args.output}")


if __name__ == "__main__":
    main()