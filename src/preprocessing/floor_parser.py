from __future__ import annotations

import re
from typing import List

import numpy as np
import pandas as pd


def extract_floor_features_series(series: pd.Series) -> pd.DataFrame:
    floor_mapping = {
        "S2": -2, "S1": -1, "S": -1, "SEMI": -1,
        "ST": -0.5, "PT": 0, "T": 0, "RIAL": 0.5, "AMMEZZ": 0.5,
    }
    # normalize and strip spaces/hyphens
    s = series.astype(str).str.upper()
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.str.replace("PIANO", "", regex=False)

    def parse_one(v: str) -> List[float]:
        if v in ("", "NONE", "NULL", "NAN"):
            return []
        values: List[float] = []
        # tokenization on non-alphanum
        parts = re.split(r"[^A-Z0-9]+", v)
        for p in parts:
            if not p:
                continue
            m = re.fullmatch(r"P(\d{1,2})", p)
            if m:
                values.append(float(m.group(1)))
                continue
            key = p
            if key in floor_mapping:
                values.append(float(floor_mapping[key]))
                continue
            if re.fullmatch(r"\d{1,2}", p):
                n = int(p)
                if 1 <= n <= 20:
                    values.append(float(n))
        # explicit ranges like 1-3
        for ra in re.findall(r"(\d{1,2})-(\d{1,2})", v):
            a, b = sorted(map(int, ra))
            values.extend([float(x) for x in range(a, b + 1) if 1 <= x <= 20])
        return sorted(set(values))

    parsed = s.apply(parse_one)

    def agg(vals: List[float], fn):
        return (fn(vals) if vals else np.nan)

    out = pd.DataFrame({
        "min_floor": parsed.apply(lambda v: agg(v, min)),
        "max_floor": parsed.apply(lambda v: agg(v, max)),
        "n_floors": parsed.apply(lambda v: float(len(v))),
        "floor_span": parsed.apply(lambda v: (max(v) - min(v)) if v else 0.0),
        "has_basement": parsed.apply(lambda v: float(any(x < 0 for x in v))),
        "has_ground": parsed.apply(lambda v: float(any(-0.5 <= x <= 0.5 for x in v))),
        "has_upper": parsed.apply(lambda v: float(any(x >= 1 for x in v))),
        "n_upper_levels": parsed.apply(lambda v: float(sum(x >= 1 for x in v))),
        "n_basement_levels": parsed.apply(lambda v: float(sum(x < 0 for x in v))),
        "contains_range": s.str.contains(r"\d+\s*-\s*\d+", regex=True).astype(float),
    }, index=series.index)

    def weighted_mean(vals: List[float]) -> float:
        if not vals:
            return np.nan
        weights = [x + 3 for x in vals]
        return float(np.average(vals, weights=weights))

    out["floor_numeric_weighted"] = parsed.apply(weighted_mean)
    return out