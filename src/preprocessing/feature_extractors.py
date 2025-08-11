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
_WKT_MULTIPOLYGON_PREFIX_RE = re.compile(r"^MULTIPOLYGON\s*\(\(", re.IGNORECASE)


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


def multipolygon_stats_from_wkt(series: pd.Series) -> pd.DataFrame:
    poly_count: List[int | None] = []
    total_vertex_count: List[int | None] = []
    outer_ring_vertex_avg: List[float | None] = []
    for s in series.astype(str).fillna(""):
        s_up = s.upper().strip()
        if not s_up.startswith("MULTIPOLYGON("):
            poly_count.append(None)
            total_vertex_count.append(None)
            outer_ring_vertex_avg.append(None)
            continue
        try:
            # Remove outer wrapper MULTIPOLYGON( ... )
            inner = s_up[len("MULTIPOLYGON("):-1]
            chunks = [c for c in inner.split(") ),( (".replace(" ", "")) if c]
            # Fallback splitting if exact pattern not matched
            if len(chunks) <= 1:
                chunks = [c for c in inner.split(")),((") if c]
            pc = len(chunks)
            poly_count.append(pc if pc > 0 else None)
            v_tot = 0
            v_outer_list: List[int] = []
            for ch in chunks:
                outer = ch.split("),(")[0]
                vertices = [c for c in outer.split(",") if c.strip()]
                v_tot += len(vertices)
                v_outer_list.append(len(vertices))
            total_vertex_count.append(v_tot if v_tot > 0 else None)
            outer_ring_vertex_avg.append((sum(v_outer_list) / len(v_outer_list)) if v_outer_list else None)
        except Exception:
            poly_count.append(None)
            total_vertex_count.append(None)
            outer_ring_vertex_avg.append(None)
    return pd.DataFrame({
        "wkt_mpoly_count": poly_count,
        "wkt_mpoly_vertices": total_vertex_count,
        "wkt_mpoly_outer_vertices_avg": outer_ring_vertex_avg,
    }, index=series.index)


def detect_geometry_wkt_columns(df: pd.DataFrame) -> Dict[str, str]:
    geometry_cols: Dict[str, str] = {}
    sample = df.select_dtypes(include=["object"]).head(200)
    for col in sample.columns:
        vals = sample[col].dropna().astype(str).str.upper().str.strip()
        if vals.empty:
            continue
        if (vals.str.startswith("POINT(")).any():
            geometry_cols[col] = "POINT"
        elif (vals.str.startswith("POLYGON(")).any():
            geometry_cols[col] = "POLYGON"
        elif (vals.str.startswith("MULTIPOLYGON(")).any():
            geometry_cols[col] = "MULTIPOLYGON"
        elif (vals.str.startswith("LINESTRING(")).any():
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
            elif gtype == "MULTIPOLYGON":
                stats = multipolygon_stats_from_wkt(df[col])
                df = pd.concat([df, stats.add_prefix(f"{col}__")], axis=1)
                cols_to_drop.append(col)
            else:
                # Not extracting for LINESTRING for now; keep raw as is
                pass
        except Exception as exc:
            logger.warning(f"Geometry extraction failed for column {col}: {exc}")
    return df, cols_to_drop


def extract_geojson_polygon_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    cols_to_drop: List[str] = []
    for col in df.columns:
        if not pd.api.types.is_object_dtype(df[col]):
            continue
        sample_vals = df[col].dropna().astype(str).str.strip().head(5)
        if sample_vals.empty or not sample_vals.str.startswith("{").any():
            continue
        try:
            first_obj = json.loads(sample_vals.iloc[0])
        except Exception:
            continue
        if not isinstance(first_obj, dict) or first_obj.get("type") not in {"Feature", "FeatureCollection"}:
            continue

        def _extract_first_feature(obj_str: str) -> Dict[str, object] | None:
            try:
                obj = json.loads(obj_str)
                feat = None
                if obj.get("type") == "FeatureCollection":
                    feats = obj.get("features") or []
                    feat = feats[0] if feats else None
                elif obj.get("type") == "Feature":
                    feat = obj
                if not feat:
                    return None
                props = feat.get("properties") or {}
                out: Dict[str, object] = {
                    f"{col}__areaMq": props.get("areaMq"),
                    f"{col}__perimetroM": props.get("perimetroM"),
                    f"{col}__codiceCatastale": props.get("codiceCatastale"),
                    f"{col}__foglio": props.get("foglio"),
                    f"{col}__sezione": props.get("sezione"),
                    f"{col}__particella": props.get("particella"),
                }
                geom = (feat.get("geometry") or {})
                coords = geom.get("coordinates") or []
                flat: List[float] = []
                def _flatten(xs):
                    if isinstance(xs, (list, tuple)):
                        for y in xs:
                            yield from _flatten(y)
                    else:
                        yield xs
                for z in _flatten(coords):
                    flat.append(z)
                # Build bbox if pairs
                pts = []
                i = 0
                while i + 1 < len(flat):
                    if isinstance(flat[i], (int, float)) and isinstance(flat[i+1], (int, float)):
                        pts.append((float(flat[i]), float(flat[i+1])))
                        i += 2
                    else:
                        i += 1
                if pts:
                    xs, ys = zip(*pts)
                    out[f"{col}__minx"] = min(xs)
                    out[f"{col}__miny"] = min(ys)
                    out[f"{col}__maxx"] = max(xs)
                    out[f"{col}__maxy"] = max(ys)
                return out
            except Exception:
                return None

        extracted = df[col].astype(str).apply(_extract_first_feature)
        if extracted.notna().any():
            extr_df = pd.DataFrame(extracted.tolist(), index=df.index)
            df = pd.concat([df, extr_df], axis=1)
            cols_to_drop.append(col)
            logger.info(f"Estratti campi GeoJSON da {col}")
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