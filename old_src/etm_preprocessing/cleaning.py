from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import Iterable

def std_col(name: str) -> str:
    """Normalize column names to snake_case, strip symbols."""
    s = name.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("≥", "ge")
    s = s.replace("(", "_").replace(")", "")
    s = s.replace("/", "_").replace("%", "pct").replace("-", " ")
    s = re.sub(r"[^0-9a-zA-Z_ ]+", "", s)
    s = re.sub(r"\s+", "_", s).lower()
    return s

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standardized column names. O(M) column ops."""
    return df.rename(columns={c: std_col(str(c)) for c in df.columns})

def to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Coerce listed columns to numeric (in place). O(N·|cols|)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def clean_grades(series: pd.Series) -> pd.Series:
    """Uppercase, strip, unify NA. O(N)."""
    out = series.astype("string").str.strip().str.upper()
    out = out.replace({"N/A": None, "NA": None, "": None})
    return out
