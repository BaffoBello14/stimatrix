from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# Regex for basic WKT parsing without external spatial deps
_WKT_POINT_RE = re.compile(r"^POINT\s*\(\s*([+-]?[0-9]*\.?[0-9]+)\s+([+-]?[0-9]*\.?[0-9]+)\s*\)$", re.IGNORECASE)
_WKT_POLYGON_RE = re.compile(r"^POLYGON\s*\(\((.*?)\)\)$", re.IGNORECASE)


def extract_point_xy_from_wkt(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    xs: List[float | None] = []
    ys: List[float | None] = []
    for val in series.astype(str).fillna(""):
        match = _WKT_POINT_RE.match(val)
        if match:
            try:
                xs.append(float(match.group(1)))
                ys.append(float(match.group(2)))
            except Exception:
                xs.append(None)
                ys.append(None)
        else:
            xs.append(None)
            ys.append(None)
    return pd.Series(xs, index=series.index), pd.Series(ys, index=series.index)


def polygon_vertex_count_from_wkt(series: pd.Series) -> pd.Series:
    counts: List[int | None] = []
    for val in series.astype(str).fillna(""):
        match = _WKT_POLYGON_RE.match(val)
        if not match:
            counts.append(None)
            continue
        coords_str = match.group(1)
        # split on comma for vertices of the outer ring
        vertices = [c.strip() for c in coords_str.split(',') if c.strip()]
        counts.append(len(vertices) if vertices else None)
    return pd.Series(counts, index=series.index)


def detect_geometry_wkt_columns(df: pd.DataFrame) -> Dict[str, str]:
    geometry_cols: Dict[str, str] = {}
    sample = df.head(100)
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            vals = sample[col].dropna().astype(str)
            if vals.empty:
                continue
            v0 = vals.iloc[0]
            if v0.startswith("POINT("):
                geometry_cols[col] = "POINT"
            elif v0.startswith("POLYGON("):
                geometry_cols[col] = "POLYGON"
            elif v0.startswith("LINESTRING("):
                geometry_cols[col] = "LINESTRING"
    return geometry_cols


def extract_geometry_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    geometry_cols = detect_geometry_wkt_columns(df)
    cols_to_drop: List[str] = []
    for col, gtype in geometry_cols.items():
        try:
            if gtype == "POINT":
                x, y = extract_point_xy_from_wkt(df[col])
                df[f"{col}_x"] = x
                df[f"{col}_y"] = y
                cols_to_drop.append(col)
            elif gtype == "POLYGON":
                df[f"{col}_vertex_count"] = polygon_vertex_count_from_wkt(df[col])
                cols_to_drop.append(col)
            else:
                # Not extracting for LINESTRING for now; keep raw as is
                pass
        except Exception as exc:
            logger.warning(f"Geometry extraction failed for column {col}: {exc}")
    return df, cols_to_drop


def maybe_extract_json_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    cols_to_drop: List[str] = []
    for col in df.columns:
        if not pd.api.types.is_object_dtype(df[col]):
            continue
        sample_vals = df[col].dropna().astype(str).head(20)
        if sample_vals.empty:
            continue
        looks_like_json = any(v.startswith("{") or v.startswith("[") for v in sample_vals)
        if not looks_like_json:
            continue
        # Try to parse and extract shallow keys if dict-like with flat primitives
        def _extract_first_level(obj) -> Dict[str, object]:
            try:
                if isinstance(obj, str):
                    obj = json.loads(obj)
                if isinstance(obj, dict):
                    flat = {}
                    for k, v in obj.items():
                        if isinstance(v, (int, float, str, bool)) or v is None:
                            flat[k] = v
                    return flat
            except Exception:
                return {}
            return {}

        extracted_rows = sample_vals.apply(_extract_first_level)
        # Determine candidate keys (present in at least 20% of sample)
        key_counts: Dict[str, int] = {}
        for d in extracted_rows:
            for k in d.keys():
                key_counts[k] = key_counts.get(k, 0) + 1
        candidate_keys = [k for k, c in key_counts.items() if c >= max(1, int(0.2 * len(sample_vals)))]
        if not candidate_keys:
            continue
        # Extract for full column
        def _safe_get(obj, key):
            try:
                if isinstance(obj, str):
                    obj = json.loads(obj)
                if isinstance(obj, dict):
                    return obj.get(key)
            except Exception:
                return None
            return None

        for k in candidate_keys:
            new_col = f"{col}__{k}"
            df[new_col] = df[col].apply(lambda v, kk=k: _safe_get(v, kk))
        cols_to_drop.append(col)
        logger.info(f"Estratti {len(candidate_keys)} campi da JSON nella colonna {col}: {candidate_keys}")
    return df, cols_to_drop