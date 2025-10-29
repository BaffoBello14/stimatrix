from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.exceptions import (
    FeatureExtractionError, 
    with_error_handling, 
    raise_for_data_validation
)

logger = get_logger(__name__)

# Regex for basic WKT parsing without external spatial deps
_WKT_POINT_RE = re.compile(r"^POINT\s*\(\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s+([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*\)$", re.IGNORECASE)
_WKT_POLYGON_RE = re.compile(r"^POLYGON\s*\(\((.*?)\)\)$", re.IGNORECASE)


@with_error_handling(error_types=FeatureExtractionError)
def extract_point_xy_from_wkt(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Extract X,Y coordinates from WKT POINT geometries with robust error handling."""
    raise_for_data_validation(
        isinstance(series, pd.Series), 
        "Input must be a pandas Series",
        column=getattr(series, 'name', 'unknown')
    )
    
    xs: List[float | None] = []
    ys: List[float | None] = []
    
    try:
        for idx, val in enumerate(series.astype(str).fillna("")):
            match = _WKT_POINT_RE.match(val)
            if match:
                try:
                    x_coord = float(match.group(1))
                    y_coord = float(match.group(2))
                    
                    # Allow all coordinate values - validation removed for flexibility
                    xs.append(x_coord)
                    ys.append(y_coord)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse coordinates at index {idx}: {e}")
                    xs.append(None)
                    ys.append(None)
            else:
                xs.append(None)
                ys.append(None)
                
    except Exception as e:
        raise FeatureExtractionError(
            f"Failed to extract point coordinates: {e}",
            feature_type="wkt_point",
            column=getattr(series, 'name', 'unknown')
        )
    
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
        # Allow for optional spaces in WKT patterns
        if (vals.str.contains(r"^POINT\s*\(", regex=True)).any():
            geometry_cols[col] = "POINT"
        elif (vals.str.contains(r"^POLYGON\s*\(", regex=True)).any():
            geometry_cols[col] = "POLYGON"
        elif (vals.str.contains(r"^MULTIPOLYGON\s*\(", regex=True)).any():
            geometry_cols[col] = "MULTIPOLYGON"
        elif (vals.str.contains(r"^LINESTRING\s*\(", regex=True)).any():
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


def extract_temporal_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract advanced temporal features from date columns.
    
    Features:
    - quarter (Q1-Q4)
    - is_summer (June-August)
    - month_sin, month_cos (cyclic encoding)
    - months_since_start (progressive time)
    
    Args:
        df: Input DataFrame (must have year/month columns)
        config: Configuration dict
    
    Returns:
        DataFrame with temporal features
    """
    df = df.copy()
    
    temp_cfg = config.get('advanced_features', {}).get('temporal', {})
    if not temp_cfg.get('enabled', False):
        logger.info("Advanced temporal features disabled, skipping")
        return df
    
    logger.info("Extracting advanced temporal features")
    
    # Get year and month columns
    temporal_split_cfg = config.get('temporal_split', {})
    year_col = temporal_split_cfg.get('year_col', 'A_AnnoStipula')
    month_col = temporal_split_cfg.get('month_col', 'A_MeseStipula')
    
    if year_col not in df.columns or month_col not in df.columns:
        logger.warning(f"Temporal columns not found: {year_col}, {month_col}")
        return df
    
    features_to_create = temp_cfg.get('features', [])
    
    # Quarter
    if 'quarter' in features_to_create:
        df['quarter'] = ((df[month_col] - 1) // 3 + 1).astype(int)
        logger.info("  Created: quarter")
    
    # Summer flag
    if 'is_summer' in features_to_create:
        df['is_summer'] = df[month_col].isin([6, 7, 8]).astype(int)
        logger.info("  Created: is_summer")
    
    # Cyclic month encoding
    if 'month_sin' in features_to_create:
        df['month_sin'] = np.sin(2 * np.pi * df[month_col] / 12)
        logger.info("  Created: month_sin")
    
    if 'month_cos' in features_to_create:
        df['month_cos'] = np.cos(2 * np.pi * df[month_col] / 12)
        logger.info("  Created: month_cos")
    
    # Months since start
    if 'months_since_start' in features_to_create:
        if 'TemporalKey' in df.columns:
            min_key = df['TemporalKey'].min()
            df['months_since_start'] = (df['TemporalKey'] - min_key) // 100 * 12 + (df['TemporalKey'] % 100) - (min_key % 100)
        else:
            # Create from year/month
            min_year = df[year_col].min()
            min_month = df[df[year_col] == min_year][month_col].min()
            df['months_since_start'] = (df[year_col] - min_year) * 12 + (df[month_col] - min_month)
        logger.info("  Created: months_since_start")
    
    return df


def extract_geographic_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract advanced geographic features.
    
    Features:
    - spatial_clusters (KMeans on lat/lon)
    - distance_to_center (distance from city center)
    - density (property count within radius)
    
    Args:
        df: Input DataFrame (must have lat/lon columns)
        config: Configuration dict
    
    Returns:
        DataFrame with geographic features
    """
    df = df.copy()
    
    geo_cfg = config.get('advanced_features', {}).get('geographic', {})
    if not geo_cfg.get('enabled', False):
        logger.info("Advanced geographic features disabled, skipping")
        return df
    
    logger.info("Extracting advanced geographic features")
    
    # Find lat/lon columns
    lat_col = None
    lon_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'lat' in col_lower and '_y' in col_lower:
            lat_col = col
        elif 'lon' in col_lower and '_x' in col_lower:
            lon_col = col
        elif col_lower.endswith('_y') and 'posizione' in col_lower:
            lat_col = col
        elif col_lower.endswith('_x') and 'posizione' in col_lower:
            lon_col = col
    
    if lat_col is None or lon_col is None:
        logger.warning("Lat/Lon columns not found, skipping geographic features")
        return df
    
    logger.info(f"Using lat/lon columns: {lat_col}, {lon_col}")
    
    # Filter valid coordinates
    valid_coords = df[[lat_col, lon_col]].notna().all(axis=1)
    coords = df.loc[valid_coords, [lat_col, lon_col]].values
    
    if len(coords) == 0:
        logger.warning("No valid coordinates found")
        return df
    
    # Spatial clusters
    cluster_cfg = geo_cfg.get('spatial_clusters', {})
    if cluster_cfg.get('enabled', False):
        try:
            from sklearn.cluster import KMeans
            
            n_clusters = int(cluster_cfg.get('n_clusters', 8))
            feature_name = cluster_cfg.get('feature_name', 'geo_cluster')
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df[feature_name] = np.nan
            df.loc[valid_coords, feature_name] = kmeans.fit_predict(coords)
            
            logger.info(f"  Created: {feature_name} (n_clusters={n_clusters})")
        except Exception as e:
            logger.warning(f"  Failed to create spatial clusters: {e}")
    
    # Distance to center
    dist_cfg = geo_cfg.get('distance_to_center', {})
    if dist_cfg.get('enabled', False):
        try:
            center_lat = float(dist_cfg.get('center_lat', 45.1564))
            center_lon = float(dist_cfg.get('center_lon', 10.7914))
            feature_name = dist_cfg.get('feature_name', 'distance_to_center_km')
            
            # Haversine distance
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371  # Earth radius in km
                dlat = np.radians(lat2 - lat1)
                dlon = np.radians(lon2 - lon1)
                a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                return R * c
            
            df[feature_name] = np.nan
            df.loc[valid_coords, feature_name] = haversine(
                df.loc[valid_coords, lat_col].values,
                df.loc[valid_coords, lon_col].values,
                center_lat,
                center_lon
            )
            
            logger.info(f"  Created: {feature_name}")
        except Exception as e:
            logger.warning(f"  Failed to create distance_to_center: {e}")
    
    # Density (neighbors within radius)
    dens_cfg = geo_cfg.get('density', {})
    if dens_cfg.get('enabled', False):
        try:
            from sklearn.neighbors import BallTree
            
            radius_km = float(dens_cfg.get('radius_km', 0.5))
            feature_name = dens_cfg.get('feature_name', 'density_500m')
            
            # BallTree requires radians
            coords_rad = np.radians(coords)
            tree = BallTree(coords_rad, metric='haversine')
            
            # Query radius in radians (km / Earth radius)
            radius_rad = radius_km / 6371
            counts = tree.query_radius(coords_rad, r=radius_rad, count_only=True)
            
            df[feature_name] = np.nan
            df.loc[valid_coords, feature_name] = counts - 1  # Exclude self
            
            logger.info(f"  Created: {feature_name} (radius={radius_km}km)")
        except Exception as e:
            logger.warning(f"  Failed to create density: {e}")
    
    return df


def create_interaction_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create interaction features.
    
    Interactions:
    - categorical × numeric (e.g., Superficie × ZonaOmi)
    - categorical × categorical (e.g., Categoria × Zona)
    - polynomial (e.g., Superficie²)
    
    Args:
        df: Input DataFrame
        config: Configuration dict
    
    Returns:
        DataFrame with interaction features
    """
    df = df.copy()
    
    int_cfg = config.get('advanced_features', {}).get('interactions', {})
    if not int_cfg.get('enabled', False):
        logger.info("Interaction features disabled, skipping")
        return df
    
    logger.info("Creating interaction features")
    
    # Categorical × Numeric
    cat_num_pairs = int_cfg.get('categorical_numeric', [])
    for pair in cat_num_pairs:
        if len(pair) != 2:
            continue
        
        num_col, cat_col = pair
        
        if num_col not in df.columns or cat_col not in df.columns:
            logger.warning(f"  Skipping {num_col} × {cat_col}: columns not found")
            continue
        
        # Group-wise mean encoding
        feature_name = f'{num_col}_x_{cat_col}'
        group_means = df.groupby(cat_col)[num_col].transform('mean')
        df[feature_name] = df[num_col] * (group_means / (group_means.mean() + 1e-10))
        
        logger.info(f"  Created: {feature_name}")
    
    # Categorical × Categorical
    cat_cat_pairs = int_cfg.get('categorical_categorical', [])
    for pair in cat_cat_pairs:
        if len(pair) != 2:
            continue
        
        cat_col1, cat_col2 = pair
        
        if cat_col1 not in df.columns or cat_col2 not in df.columns:
            logger.warning(f"  Skipping {cat_col1} × {cat_col2}: columns not found")
            continue
        
        # Combine categories
        feature_name = f'{cat_col1}_x_{cat_col2}'
        df[feature_name] = df[cat_col1].astype(str) + '_' + df[cat_col2].astype(str)
        
        logger.info(f"  Created: {feature_name}")
    
    # Polynomial
    poly_cfg = int_cfg.get('polynomial', {})
    poly_cols = poly_cfg.get('columns', [])
    degree = int(poly_cfg.get('degree', 2))
    
    for col in poly_cols:
        if col not in df.columns:
            continue
        
        if df[col].dtype in ['int64', 'float64']:
            for d in range(2, degree + 1):
                feature_name = f'{col}_pow{d}'
                df[feature_name] = df[col] ** d
                logger.info(f"  Created: {feature_name}")
    
    return df


def create_missing_pattern_flags(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create binary flags for missing data patterns.
    
    Args:
        df: Input DataFrame
        config: Configuration dict
    
    Returns:
        DataFrame with missing pattern flags
    """
    df = df.copy()
    
    miss_cfg = config.get('advanced_features', {}).get('missing_patterns', {})
    if not miss_cfg.get('enabled', False):
        logger.info("Missing pattern flags disabled, skipping")
        return df
    
    logger.info("Creating missing pattern flags")
    
    prefixes = miss_cfg.get('create_flags_for_prefixes', [])
    template = miss_cfg.get('feature_name_template', 'has_{prefix}')
    
    for prefix in prefixes:
        # Find columns with this prefix
        cols = [c for c in df.columns if c.startswith(prefix)]
        
        if len(cols) == 0:
            logger.warning(f"  No columns found with prefix '{prefix}'")
            continue
        
        # Create flag: 1 if ANY column with prefix is non-null
        feature_name = template.format(prefix=prefix.rstrip('_'))
        df[feature_name] = df[cols].notna().any(axis=1).astype(int)
        
        logger.info(f"  Created: {feature_name} (from {len(cols)} columns)")
    
    return df