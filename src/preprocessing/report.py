from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def dataframe_profile(df: pd.DataFrame, max_rows: int = 2000) -> str:
    lines: List[str] = []
    lines.append(f"Rows: {len(df)}, Cols: {len(df.columns)}")
    lines.append("")
    lines.append("| Column | Dtype | Non-Null | NA% | Unique | Sample |")
    lines.append("|---|---|---:|---:|---:|---|")
    for col in df.columns[:max_rows]:
        s = df[col]
        na_pct = s.isna().mean() * 100
        nunique = s.nunique(dropna=True)
        sample = s.dropna().head(1).tolist()
        sample_val = str(sample[0])[:80] if sample else ""
        lines.append(
            f"| {col} | {s.dtype} | {s.notna().sum()} | {na_pct:.2f} | {nunique} | {sample_val} |"
        )
    return "\n".join(lines)


def save_report(path: str | Path, sections: Dict[str, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content_lines: List[str] = []
    for title, body in sections.items():
        content_lines.append(f"### {title}")
        content_lines.append("")
        content_lines.append(body)
        content_lines.append("")
    path.write_text("\n".join(content_lines), encoding="utf-8")
    logger.info(f"Report salvato in {path}")