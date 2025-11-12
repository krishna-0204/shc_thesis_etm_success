from __future__ import annotations
import argparse, re, json
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import (
    DEFAULT_CLEAN, DEFAULT_LONG, PROC_DIR, INSIGHTS_DIR, OUTCOME_COL,
    ensure_outcome_col, ensure_binary
)

QUAR_DIR = INSIGHTS_DIR / "quarantine"
PLOTS_DIR = INSIGHTS_DIR / "plots"
SANITIZED = INSIGHTS_DIR / "_sanitized_clean.csv"

# ---- Config knobs (tweak freely)
DENY_PATTERNS = [
    r"graduating", r"degree", r"me_bs", r"outcome", r"_is_missing$"
]
WARN_PATTERNS = [
    r"cgpa$", r"gpa$", r"gpa_.*(final|last)", r"status", r"etm_window"
]
MAX_MISSING_PCT = 0.60        # features with >60% NA → excluded from main, copied to quarantine
CORR_EQUIV_THR  = 0.985       # if (almost) equal to outcome → quarantine
LEVEL_MIN_N     = 50          # minimum level size we consider stable in downstream steps

# Treat strings like "N/A (101)" as missing
NA_STRINGS = re.compile(r"^\s*(n/?a|na|n\.a\.|not\s+available)(.*101.*)?\s*$", re.I)

# Canonical credit bins (mutually exclusive)
CREDIT_BINS = [(0,19),(20,39),(40,59),(60,79),(80,99),(100,10_000)]
CREDIT_LABELS = [f"{a}-{b}" if b<10_000 else f"{a}+" for a,b in CREDIT_BINS]

def is_name_leaky(col: str) -> bool:
    lower = col.lower()
    return any(re.search(p, lower) for p in DENY_PATTERNS)

def is_name_warn(col: str) -> bool:
    lower = col.lower()
    return any(re.search(p, lower) for p in WARN_PATTERNS)

def near_equal_to_outcome(x: pd.Series, y: pd.Series) -> bool:
    """Flag if x equals y (or 1-y) on ~all non-NA rows."""
    x_ = pd.to_numeric(x, errors="coerce")
    if x_.notna().sum() < 100 or y.notna().sum() < 100:
        return False
    y_ = ensure_binary(y)
    common = x_.notna() & y_.notna()
    if common.sum() == 0:
        return False
    same = np.mean(np.isclose(x_[common].values, y_[common].values, equal_nan=False))
    inv  = np.mean(np.isclose(x_[common].values, 1 - y_[common].values, equal_nan=False))
    return max(same, inv) >= CORR_EQUIV_THR

def normalize_missing_categories(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if df2[c].dtype == "object" or pd.api.types.is_string_dtype(df2[c]):
            s = df2[c].astype("string")
            mask = s.fillna("").str.match(NA_STRINGS)
            if mask.any():
                df2.loc[mask, c] = np.nan
    return df2

def normalize_credit_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a canonical `credit_window_canon` with non-overlapping bins.
    Works if you already have something like 'credit_window' that looks like '29-55' etc.,
    or if you have a numeric credits column (try a few common names).
    """
    df2 = df.copy()
    target = None
    # 1) Try numeric credits columns first
    for cand in ["attempted_credits", "earned_credits", "credits_at_checkpoint", "credits_by_etm"]:
        if cand in df2.columns and pd.api.types.is_numeric_dtype(df2[cand]):
            target = cand
            break

    if target:
        v = pd.to_numeric(df2[target], errors="coerce")
        bins = [a for a,_ in CREDIT_BINS] + [CREDIT_BINS[-1][1]]
        df2["credit_window_canon"] = pd.cut(v, bins=bins, labels=CREDIT_LABELS, include_lowest=True, right=True)
        return df2

    # 2) Else parse from string ranges like "29-55"
    if "credit_window" in df2.columns:
        def mid(s):
            try:
                a,b = re.findall(r"(\d+)\s*-\s*(\d+)", str(s))[0]
                return (int(a)+int(b))/2.0
            except Exception:
                return np.nan
        m = df2["credit_window"].map(mid)
        bins = [a for a,_ in CREDIT_BINS] + [CREDIT_BINS[-1][1]]
        df2["credit_window_canon"] = pd.cut(m, bins=bins, labels=CREDIT_LABELS, include_lowest=True, right=True)
    return df2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(DEFAULT_CLEAN))
    ap.add_argument("--out", default=str(SANITIZED))
    ap.add_argument("--report", default=str(INSIGHTS_DIR / "_sanitizer_report.json"))
    ap.add_argument("--keep_warn", action="store_true", help="Keep WARN-pattern columns in main (default True).")
    args = ap.parse_args()

    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    QUAR_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.clean)
    df = ensure_outcome_col(df, OUTCOME_COL)
    y  = df[OUTCOME_COL] if OUTCOME_COL in df.columns else pd.Series(dtype=float)

    # Normalize obvious missing sentinels
    df = normalize_missing_categories(df)

    # Canonicalize credit window
    df = normalize_credit_window(df)

    # Build guard lists
    deny_cols, warn_cols, na_heavy_cols, eq_outcome_cols = [], [], [], []
    for c in df.columns:
        if c == OUTCOME_COL: 
            continue
        if is_name_leaky(c):
            deny_cols.append(c); continue
        if is_name_warn(c):
            warn_cols.append(c)

        miss = df[c].isna().mean()
        if miss > MAX_MISSING_PCT:
            na_heavy_cols.append(c); continue

        try:
            if near_equal_to_outcome(df[c], y):
                eq_outcome_cols.append(c)
        except Exception:
            pass

    # Quarantine dataframe for flagged columns
    quarantine_cols = sorted(set(deny_cols) | set(na_heavy_cols) | set(eq_outcome_cols))
    df_quar = df[[OUTCOME_COL] + [c for c in quarantine_cols if c in df.columns]].copy()
    if not df_quar.empty:
        df_quar.to_csv(QUAR_DIR / "_quarantined_columns.csv", index=False)

    # Build sanitized main dataframe
    drop_from_main = sorted(set(deny_cols) | set(na_heavy_cols) | set(eq_outcome_cols))
    df_main = df.drop(columns=[c for c in drop_from_main if c in df.columns]).copy()

    # Save outputs
    df_main.to_csv(args.out, index=False)

    report = {
        "sanitized_clean_path": str(SANITIZED),
        "dropped_from_main": drop_from_main,
        "warn_columns": warn_cols,
        "na_heavy_cols": na_heavy_cols,
        "name_denied_cols": deny_cols,
        "equal_to_outcome_cols": eq_outcome_cols,
        "config": {
            "MAX_MISSING_PCT": MAX_MISSING_PCT,
            "CORR_EQUIV_THR": CORR_EQUIV_THR,
            "LEVEL_MIN_N": LEVEL_MIN_N,
            "DENY_PATTERNS": DENY_PATTERNS,
            "WARN_PATTERNS": WARN_PATTERNS
        }
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[sanitizer] wrote {args.out} and {args.report}")
    if quarantine_cols:
        print(f"[sanitizer] quarantined cols: {len(quarantine_cols)} → {QUAR_DIR}/_quarantined_columns.csv")

if __name__ == "__main__":
    main()
