from __future__ import annotations
import pandas as pd
import re
from pathlib import Path

def load_excel(path: str | Path) -> pd.ExcelFile:
    """Load Excel workbook.

    Complexity: O(1) to open; actual read cost in downstream readers.
    """
    return pd.ExcelFile(Path(path))

def read_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    """Read a single sheet into a DataFrame. Complexity: O(R·C)."""
    return pd.read_excel(xls, sheet_name=sheet_name)

def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a CSV to disk. Complexity: O(N·M) bytes written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)