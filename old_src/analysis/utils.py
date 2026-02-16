from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

# --------- Paths
DATA_DIR       = Path("data")
PROC_DIR       = DATA_DIR / "processed"
INSIGHTS_DIR   = PROC_DIR / "insights"
DEFAULT_CLEAN  = PROC_DIR / "clean_features.csv"
DEFAULT_LONG   = PROC_DIR / "clean_features_terms_long.csv"
INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# --------- Study knobs
OUTCOME_COL    = "graduated_me"   # override via CLI as needed
MIN_N          = 25               # min cohort size to report
MAX_LEVELS_CAT = 60               # skip extreme high-cardinality categoricals

# --------- Loading
def load_frames(clean_path: str | os.PathLike = DEFAULT_CLEAN,
                long_path: str | os.PathLike | None = DEFAULT_LONG):
    df = pd.read_csv(clean_path)
    df = ensure_outcome_col(df, OUTCOME_COL)
    df_long = pd.read_csv(long_path) if (long_path and Path(long_path).exists()) else None
    return df, df_long

# --------- Outcome helpers
_TRUTHY = {"1","y","yes","true","t","graduate","graduated","degree awarded","awarded"}
_FALSY  = {"0","n","no","false","f","not","did not","none","missing"}

def ensure_binary(series: pd.Series) -> pd.Series:
    """Coerce a series into {0,1} floats; preserves NaN."""
    if pd.api.types.is_numeric_dtype(series):
        s = pd.to_numeric(series, errors="coerce")
        return (s > 0).astype(float)
    s = series.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out[s.isin(_TRUTHY)] = 1.0
    out[s.isin(_FALSY)]  = 0.0
    # common encodings like 'Y/N'
    out[s.eq("y")] = 1.0
    out[s.eq("n")] = 0.0
    return out

def ensure_outcome_col(df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    """Create/normalize the binary outcome column if an alias exists."""
    if outcome_col in df.columns:
        df[outcome_col] = ensure_binary(df[outcome_col])
        return df
    # Common aliases from your context
    for cand in ["ME_BS Degree Status", "me_bs_degree_status", "me_degree_awarded"]:
        if cand in df.columns:
            df[outcome_col] = ensure_binary(df[cand])
            return df
    return df  # leave as-is; downstream code will error if outcome missing

# --------- Typing helpers
def _is_id_like(name: str, s: pd.Series) -> bool:
    if "id" in name.lower():
        return True
    # many unique values relative to n ⇒ likely identifier
    nunique = s.nunique(dropna=True)
    return nunique > max(200, 0.5 * len(s))

def numeric_features(df: pd.DataFrame, exclude: set[str] = None):
    exclude = set(exclude or [])
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def categorical_features(df: pd.DataFrame, exclude: set[str] = None):
    exclude = set(exclude or [])
    nums = set(numeric_features(df, exclude))
    cats = []
    for c in df.columns:
        if c in exclude or c in nums:
            continue
        if _is_id_like(c, df[c]):
            continue
        if df[c].nunique(dropna=True) > MAX_LEVELS_CAT:
            continue
        cats.append(c)
    return cats

# --------- Metrics
def cohen_d(x: pd.Series, y_bin: pd.Series):
    a = x[y_bin == 1].dropna()
    b = x[y_bin == 0].dropna()
    n1, n0 = len(a), len(b)
    if n1 < MIN_N or n0 < MIN_N:
        return None
    m1, m0 = a.mean(), b.mean()
    v1, v0 = a.var(ddof=1), b.var(ddof=1)
    if n1 + n0 - 2 <= 0:
        return None
    s_pooled = np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2))
    if s_pooled == 0 or np.isnan(s_pooled):
        return None
    return dict(d=float((m1 - m0) / s_pooled),
                n1=int(n1), n0=int(n0), mean1=float(m1), mean0=float(m0))

def rr_ci(count_a1: int, n_a: int, count_b1: int, n_b: int, smooth: float = 0.5):
    """Risk ratio with log-normal 95% CI (Haldane–Anscombe smoothing)."""
    a = count_a1 + smooth
    b = (n_a - count_a1) + smooth
    c = count_b1 + smooth
    d = (n_b - count_b1) + smooth
    p1 = a / (a + b)
    p0 = c / (c + d)
    rr = p1 / p0
    se = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    lo, hi = np.exp(np.log(rr) - 1.96 * se), np.exp(np.log(rr) + 1.96 * se)
    return float(rr), float(lo), float(hi), float(p1), float(p0)

# --------- Bucketing
def quantile_buckets(s: pd.Series, q=(0, .25, .5, .75, 1.0)):
    non_na = s.dropna()
    if non_na.empty:
        return pd.Series(index=s.index, dtype="category"), []
    edges = np.unique(np.quantile(non_na, q))
    if len(edges) < 2:
        edges = np.array([non_na.min()-1e-6, non_na.max()+1e-6])
    labels = [f"[{edges[i]:.2f}, {edges[i+1]:.2f})" for i in range(len(edges)-1)]
    b = pd.cut(s, bins=edges, labels=labels, include_lowest=True, duplicates="drop")
    return b, edges.tolist()

def score_numeric(abs_d: float, n1: int, n0: int):
    return float(abs_d) * np.sqrt(n1 + n0)

def score_rr(abs_log_rr: float, n_level: int, n_rest: int):
    return float(abs_log_rr) * np.sqrt(min(n_level, n_rest))
