#!/usr/bin/env python3
"""
03_univariate_screens.py

Univariate "screening" of features vs outcomes using analysis_v2/out/analysis_view.csv.

Key rule: for outcome analyses, we restrict to rows where outcome_status_known == 1,
because unknown outcome rows should not be treated as negatives.

Writes to analysis_v2/out/:
  - 03_univariate_report.md
  - 03_numeric_screen_graduated_me.csv
  - 03_numeric_screen_any_bachelor.csv
  - 03_categorical_screen_graduated_me.csv
  - 03_categorical_screen_any_bachelor.csv
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

# Optional: chi-square p-values (nice-to-have)
try:
    from scipy.stats import chi2_contingency  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_CSV = PROJECT_ROOT / "src" / "analysis_v2" / "out" / "analysis_view.csv"

OUTDIR = PROJECT_ROOT / "src" / "analysis_v2" / "out"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_MD = OUTDIR / "03_univariate_report.md"

OUT_NUM_ME = OUTDIR / "03_numeric_screen_graduated_me.csv"
OUT_NUM_ANY = OUTDIR / "03_numeric_screen_any_bachelor.csv"
OUT_CAT_ME = OUTDIR / "03_categorical_screen_graduated_me.csv"
OUT_CAT_ANY = OUTDIR / "03_categorical_screen_any_bachelor.csv"


# -----------------------------
# Config knobs
# -----------------------------
OUTCOMES = ["outcome_graduated_me", "outcome_any_bachelor"]

# Columns to exclude from predictors (IDs, outcomes, and obvious leakage).
EXCLUDE_EXACT = {
    "random_id",
    "merge_id",
    "credit_window",  # keep credit_window_label
    "me_bs_degree_status",
    "me_bs_degree_status_clean",
    "me_bs_degree_status_is_missing",
    "me_degree_status_bucket",
    "has_me_bs_degree",
    "has_any_bachelor_degree",
    "graduated_me",
    "outcome_graduated_me",
    "outcome_any_bachelor",
    "outcome_status_known",
}

# Exclude any column name containing these substrings (extra leakage safety).
EXCLUDE_SUBSTRINGS = [
    "me_degree_status",
    "me_bs_degree_status",
    "outcome_",
]


# Treat as categorical if dtype is object/string OR if number of unique values <= this threshold
CATEGORICAL_UNIQUE_THRESHOLD = 20

# For categorical features, include only top K most frequent levels in report tables (rest pooled)
TOP_K_LEVELS = 8


# -----------------------------
# Helpers
# -----------------------------
def _to01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    # only keep 0/1; everything else -> NaN
    s = s.where(s.isin([0, 1]), np.nan)
    return s

def _is_categorical(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return True
    # treat small-cardinality numeric as categorical
    nunique = int(series.dropna().nunique())
    return nunique <= CATEGORICAL_UNIQUE_THRESHOLD

def _cohens_d(x0: np.ndarray, x1: np.ndarray) -> float:
    # Cohen's d for difference in means (1-group minus 0-group)
    if len(x0) < 2 or len(x1) < 2:
        return float("nan")
    m0, m1 = np.mean(x0), np.mean(x1)
    v0, v1 = np.var(x0, ddof=1), np.var(x1, ddof=1)
    n0, n1 = len(x0), len(x1)
    pooled = ((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2)
    if pooled <= 0 or np.isnan(pooled):
        return float("nan")
    return float((m1 - m0) / np.sqrt(pooled))

def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def _auc_by_ranks(x: np.ndarray, y: np.ndarray) -> float:
    """
    AUC for a continuous score x against binary label y using rank statistic.
    Returns NaN if not computable.
    """
    if len(x) < 3:
        return float("nan")
    y = y.astype(int)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")

    # average ranks for ties
    ranks = pd.Series(x).rank(method="average").to_numpy()
    rank_sum_pos = float(ranks[y == 1].sum())
    # Mann–Whitney U for positives
    u1 = rank_sum_pos - n1 * (n1 + 1) / 2
    auc = u1 / (n0 * n1)
    return float(auc)

def _cramers_v(cont: np.ndarray) -> float:
    """
    Cramer's V effect size for a contingency table.
    """
    if cont.size == 0:
        return float("nan")
    n = cont.sum()
    if n == 0:
        return float("nan")
    # chi-square (no correction)
    row_sums = cont.sum(axis=1, keepdims=True)
    col_sums = cont.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((cont - expected) ** 2 / expected)

    r, k = cont.shape
    denom = n * (min(r - 1, k - 1))
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / denom))

def _pick_predictor_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in EXCLUDE_EXACT:
            continue
        lc = c.lower()
        if any(sub in lc for sub in EXCLUDE_SUBSTRINGS):
            continue
        cols.append(c)
    return cols


# -----------------------------
# Screens
# -----------------------------
def numeric_screen(df: pd.DataFrame, ycol: str, predictors: list[str]) -> pd.DataFrame:
    rows = []
    y = _to01(df[ycol])

    for c in predictors:
        s = df[c]
        if _is_categorical(s):
            continue

        x = pd.to_numeric(s, errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 30:
            continue

        xv = x[mask].to_numpy(dtype=float)
        yv = y[mask].to_numpy(dtype=float)

        x0 = xv[yv == 0]
        x1 = xv[yv == 1]

        rows.append({
            "feature": c,
            "n": int(mask.sum()),
            "missing_rate": float(1 - mask.mean()),
            "mean_y0": float(np.mean(x0)) if len(x0) else float("nan"),
            "mean_y1": float(np.mean(x1)) if len(x1) else float("nan"),
            "std_y0": float(np.std(x0, ddof=1)) if len(x0) > 1 else float("nan"),
            "std_y1": float(np.std(x1, ddof=1)) if len(x1) > 1 else float("nan"),
            "cohens_d": _cohens_d(x0, x1),
            "pearson_r": _pearson_r(xv, yv),
            "auc_rank": _auc_by_ranks(xv, yv),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # rank by absolute effect size
    out["abs_d"] = out["cohens_d"].abs()
    out["abs_r"] = out["pearson_r"].abs()
    out = out.sort_values(["abs_d", "abs_r"], ascending=False).drop(columns=["abs_d", "abs_r"])
    return out


def categorical_screen(df: pd.DataFrame, ycol: str, predictors: list[str]) -> pd.DataFrame:
    rows = []
    y = _to01(df[ycol])

    for c in predictors:
        s = df[c]
        if not _is_categorical(s):
            continue

        sc = s.astype("string").fillna("missing").str.strip()
        mask = y.notna()  # keep all y-known rows; categories can be "missing"
        sc = sc[mask]
        yv = y[mask].astype(int)

        if sc.nunique() <= 1:
            continue

        # pool rare levels into "other" beyond top K
        top_levels = sc.value_counts().head(TOP_K_LEVELS).index.tolist()
        sc2 = sc.where(sc.isin(top_levels), other="other")

        # contingency
        tab = pd.crosstab(sc2, yv)
        # ensure both columns exist
        if 0 not in tab.columns:
            tab[0] = 0
        if 1 not in tab.columns:
            tab[1] = 0
        tab = tab[[0, 1]]

        cont = tab.to_numpy(dtype=float)
        v = _cramers_v(cont)

        pval = float("nan")
        chi2 = float("nan")
        if HAVE_SCIPY:
            try:
                chi2, pval, _, _ = chi2_contingency(cont, correction=False)
                chi2 = float(chi2)
                pval = float(pval)
            except Exception:
                pass

        # summarize top levels by lift vs overall
        overall_rate = float(yv.mean()) if len(yv) else float("nan")
        for lvl, row in tab.iterrows():
            n0 = float(row.get(0, 0.0))
            n1 = float(row.get(1, 0.0))
            n = n0 + n1
            if n <= 0:
                continue
            rate = n1 / n
            lift = (rate / overall_rate) if overall_rate and not np.isnan(overall_rate) else float("nan")

            rows.append({
                "feature": c,
                "level": str(lvl),
                "n": int(n),
                "rate_y1": float(rate),
                "overall_rate_y1": float(overall_rate),
                "lift_vs_overall": float(lift),
                "cramers_v": v,
                "chi2": chi2,
                "p_value": pval,
                "n_levels_used": int(tab.shape[0]),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # for each feature, keep levels sorted by n desc (so report is readable)
    out = out.sort_values(["feature", "n"], ascending=[True, False])
    return out


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing: {IN_CSV}. Run 00_build_analysis_view.py first.")

    df = pd.read_csv(IN_CSV)

    # Restrict to known outcome rows
    if "outcome_status_known" not in df.columns:
        raise ValueError("analysis_view.csv missing outcome_status_known")
    df_known = df[pd.to_numeric(df["outcome_status_known"], errors="coerce") == 1].copy()

    predictors = _pick_predictor_columns(df_known)

    # Run screens for each outcome
    results = {}
    for ycol in OUTCOMES:
        if ycol not in df_known.columns:
            raise ValueError(f"Missing outcome column: {ycol}")

        num = numeric_screen(df_known, ycol, predictors)
        cat = categorical_screen(df_known, ycol, predictors)

        results[ycol] = (num, cat)

    # Save CSVs
    results["outcome_graduated_me"][0].to_csv(OUT_NUM_ME, index=False)
    results["outcome_any_bachelor"][0].to_csv(OUT_NUM_ANY, index=False)

    results["outcome_graduated_me"][1].to_csv(OUT_CAT_ME, index=False)
    results["outcome_any_bachelor"][1].to_csv(OUT_CAT_ANY, index=False)

    # Build markdown report
    lines = []
    lines.append("# Univariate screens (known outcomes only)")
    lines.append("")
    lines.append(f"- input: `{IN_CSV}`")
    lines.append(f"- rows total: {len(df)}")
    lines.append(f"- rows with known outcome_status_known==1: {len(df_known)}")
    lines.append(f"- predictors screened: {len(predictors)}")
    lines.append(f"- scipy available (chi-square p-values): {HAVE_SCIPY}")
    lines.append("")
    lines.append("## Notes (important)")
    lines.append("")
    lines.append("- This report excludes degree-status-derived columns (leakage).")
    lines.append("- Unknown outcomes (status_known==0) are excluded from outcome comparisons.")
    lines.append("")

    def _top_table(df_table: pd.DataFrame, title: str, n: int = 15) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if df_table.empty:
            lines.append("_No results (table empty)._")
            lines.append("")
            return
        lines.append(df_table.head(n).to_markdown(index=False))
        lines.append("")

    # graduated_me
    num_me, cat_me = results["outcome_graduated_me"]
    _top_table(num_me, "Top numeric signals vs graduated_me (by |Cohen's d|)")
    # show top categorical levels by lift, but keep it readable: take top 30 rows after sorting by lift magnitude
    if not cat_me.empty:
        cat_me2 = cat_me.copy()
        cat_me2["abs_lift_dev"] = (cat_me2["lift_vs_overall"] - 1).abs()
        cat_me2 = cat_me2.sort_values(["cramers_v", "abs_lift_dev", "n"], ascending=[False, False, False]).drop(columns=["abs_lift_dev"])
    else:
        cat_me2 = cat_me
    _top_table(cat_me2, "Top categorical signals vs graduated_me (by Cramer's V, then lift)", n=25)

    # any_bachelor
    num_any, cat_any = results["outcome_any_bachelor"]
    _top_table(num_any, "Top numeric signals vs any_bachelor (by |Cohen's d|)")
    if not cat_any.empty:
        cat_any2 = cat_any.copy()
        cat_any2["abs_lift_dev"] = (cat_any2["lift_vs_overall"] - 1).abs()
        cat_any2 = cat_any2.sort_values(["cramers_v", "abs_lift_dev", "n"], ascending=[False, False, False]).drop(columns=["abs_lift_dev"])
    else:
        cat_any2 = cat_any
    _top_table(cat_any2, "Top categorical signals vs any_bachelor (by Cramer's V, then lift)", n=25)

    lines.append("## Files written")
    lines.append("")
    lines.append(f"- `{OUT_MD}`")
    lines.append(f"- `{OUT_NUM_ME}`")
    lines.append(f"- `{OUT_NUM_ANY}`")
    lines.append(f"- `{OUT_CAT_ME}`")
    lines.append(f"- `{OUT_CAT_ANY}`")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[ok] wrote: {OUT_MD}")
    print(f"[ok] wrote: {OUT_NUM_ME}")
    print(f"[ok] wrote: {OUT_NUM_ANY}")
    print(f"[ok] wrote: {OUT_CAT_ME}")
    print(f"[ok] wrote: {OUT_CAT_ANY}")


if __name__ == "__main__":
    main()
