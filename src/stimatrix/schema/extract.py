import argparse
import json
import re
import warnings
from collections import defaultdict
from typing import Dict, Any, Optional

from sqlalchemy import inspect
from sqlalchemy.exc import SAWarning

from stimatrix.db.connection import get_engine
from utils.io import ensure_parent_dir

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


def generate_table_alias(table_name: str) -> str:
    parts = table_name.split("_")
    aliases = []
    for part in parts:
        uppercase_letters = "".join([c for c in part if c.isupper()])
        if uppercase_letters:
            aliases.append(uppercase_letters)
        else:
            aliases.append(part[0].upper())
    return "_".join(aliases)


def extract_schema(engine, schema_name: Optional[str] = None) -> Dict[str, Any]:
    inspector = inspect(engine)
    schema: Dict[str, Any] = {}

    for table_name in inspector.get_table_names(schema=schema_name):
        columns = inspector.get_columns(table_name, schema=schema_name)
        schema[table_name] = {"alias": generate_table_alias(table_name), "columns": []}

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

    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Estrai schema DB e salvalo in JSON")
    parser.add_argument("--output", type=str, default="data/db_schema.json", help="Path file JSON di output")
    parser.add_argument("--schema", type=str, default=None, help="Schema DB (opzionale)")
    args = parser.parse_args()

    catch_unrecognized_types()
    engine = get_engine()
    schema_dict = extract_schema(engine, schema_name=args.schema)

    ensure_parent_dir(args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=4, ensure_ascii=False)

    print(f"✅ Schema esportato in {args.output}")


if __name__ == "__main__":
    main()