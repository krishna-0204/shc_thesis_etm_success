from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_binary, ensure_outcome_col

MAX_MISSING_PCT = 0.60
CORR_EQUIV_THR = 0.985

DENY_PATTERNS = [
    r"degree",
    r"status",
    r"me_bs",
    r"outcome",
    r"graduated_me$",
    r"not_graduate_me$",
    r"transferred_out_of_me$",
]

NA_STRINGS = re.compile(r"^\s*(n/?a|na|n\.a\.|not\s+available)\s*$", re.I)

def near_equal_to_outcome(x: pd.Series, y: pd.Series) -> bool:
    x_ = pd.to_numeric(x, errors="coerce")
    if x_.notna().sum() < 100 or y.notna().sum() < 100:
        return False
    y_ = ensure_binary(y)
    common = x_.notna() & y_.notna()
    if common.sum() == 0:
        return False
    same = np.mean(np.isclose(x_[common].values, y_[common].values, equal_nan=False))
    inv = np.mean(np.isclose(x_[common].values, 1 - y_[common].values, equal_nan=False))
    return max(same, inv) >= CORR_EQUIV_THR

def is_name_denied(col: str) -> bool:
    lower = col.lower()
    return any(re.search(p, lower) for p in DENY_PATTERNS)

def normalize_missing_categories(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if df2[c].dtype == "object" or pd.api.types.is_string_dtype(df2[c]):
            s = df2[c].astype("string")
            mask = s.fillna("").str.match(NA_STRINGS)
            if mask.any():
                df2.loc[mask, c] = np.nan
    return df2

def sanitize(df: pd.DataFrame, outcome_col: str) -> tuple[pd.DataFrame, dict]:
    df = ensure_outcome_col(df, outcome_col)
    df = normalize_missing_categories(df)

    y = df[outcome_col] if outcome_col in df.columns else pd.Series(dtype=float)

    deny_cols, na_heavy_cols, eq_outcome_cols = [], [], []

    for c in df.columns:
        if c == outcome_col:
            continue
        if is_name_denied(c):
            deny_cols.append(c)
            continue
        miss = df[c].isna().mean()
        if miss > MAX_MISSING_PCT:
            na_heavy_cols.append(c)
            continue
        try:
            if near_equal_to_outcome(df[c], y):
                eq_outcome_cols.append(c)
        except Exception:
            pass

    drop_from_main = sorted(set(deny_cols) | set(na_heavy_cols) | set(eq_outcome_cols))
    df_main = df.drop(columns=[c for c in drop_from_main if c in df.columns]).copy()

    report = {
        "outcome": outcome_col,
        "dropped_from_main": drop_from_main,
        "na_heavy_cols": na_heavy_cols,
        "name_denied_cols": deny_cols,
        "equal_to_outcome_cols": eq_outcome_cols,
        "config": {
            "MAX_MISSING_PCT": MAX_MISSING_PCT,
            "CORR_EQUIV_THR": CORR_EQUIV_THR,
            "DENY_PATTERNS": DENY_PATTERNS,
        },
    }
    return df_main, report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--outcome", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.clean)
    df_main, report = sanitize(df, args.outcome)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_main.to_csv(args.out, index=False)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[sanitizer] wrote {args.out} and {args.report}")

if __name__ == "__main__":
    main()

