#!/usr/bin/env python3
"""
00_build_analysis_view.py

Build an "analysis view" dataset from data/processed/clean_features.csv.

Outputs (all inside src/analysis_v2/out/):
  - analysis_view_full.csv        (debug: everything + engineered columns)
  - analysis_view.csv             (tidy: standardized + engineered + core predictors)
  - analysis_view.parquet         (tidy parquet, best for speed)
  - 00_build_summary.md

Design goals:
- Create stable outcome columns (ME graduation, any bachelor, status known)
- Standardize credit_window into a canonical label
- Remove confusing duplicate "raw Excel" columns when there is a standardized equivalent
- Produce a slim view that is easy to analyze in later scripts
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np


# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "clean_features.csv"

OUTDIR = PROJECT_ROOT / "src" / "analysis_v2" / "out"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV_FULL = OUTDIR / "analysis_view_full.csv"
OUT_CSV_TIDY = OUTDIR / "analysis_view.csv"
OUT_PARQUET_TIDY = OUTDIR / "analysis_view.parquet"
OUT_MD = OUTDIR / "00_build_summary.md"


# -----------------------------
# Helpers
# -----------------------------
def _as_int64_bool(series: pd.Series | None) -> pd.Series:
    """
    Convert to pandas nullable Int64 (0/1/<NA>).
    Accepts booleans, 0/1 ints, or strings.
    """
    if series is None:
        return pd.Series(pd.NA, dtype="Int64")

    s = series.copy()

    if pd.api.types.is_numeric_dtype(s):
        s = s.where(~s.isna(), pd.NA)
        s = s.where(s.isin([0, 1]), pd.NA)
        return s.astype("Int64")

    if pd.api.types.is_bool_dtype(s):
        return s.astype("Int64")

    s = s.astype("string").str.strip().str.lower()
    mapping = {
        "1": 1, "0": 0,
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "y": 1, "n": 0,
    }
    s = s.map(mapping)
    return s.astype("Int64")


def _standardize_credit_window_label(x: object) -> str | None:
    """
    Standardize credit_window into a label like: '40-59', '29-55', '40-49', etc.
    """
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None

    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return None

    s = s.replace("–", "-").replace("—", "-").replace(" ", "")

    if "-" in s:
        parts = s.split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return f"{int(parts[0])}-{int(parts[1])}"
        return s

    if s.isdigit():
        return str(int(s))

    return s


def _safe_value_counts(df: pd.DataFrame, col: str, top_n: int = 10) -> str:
    if col not in df.columns:
        return f"- {col}: (missing column)\n"
    vc = df[col].astype("string").fillna("missing").value_counts(dropna=False).head(top_n)
    lines = [f"- {col} (top {top_n}):"]
    for k, v in vc.items():
        lines.append(f"  - {k}: {v}")
    return "\n".join(lines) + "\n"


def _canonicalize_name(name: str) -> str:
    """
    Convert column name to a canonical snake-ish representation
    so we can detect duplicates (e.g. "Random ID" vs "random_id").
    """
    s = name.strip().lower()
    s = s.replace("≥", "ge").replace("<", "lt").replace(">", "gt")
    s = re.sub(r"[^\w]+", "_", s)         # non-alnum -> _
    s = re.sub(r"_+", "_", s).strip("_")  # collapse
    return s


def _drop_duplicate_named_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If multiple columns map to the same canonical name, keep the best one:
      1) Prefer the column whose name already equals the canonical name
      2) Otherwise prefer snake_case-looking (no spaces, all lower)
      3) Otherwise keep the shortest name (stable tie-break)
    """
    groups: dict[str, list[str]] = {}
    for c in df.columns:
        groups.setdefault(_canonicalize_name(c), []).append(c)

    keep_cols: set[str] = set()

    for canon, cols in groups.items():
        if len(cols) == 1:
            keep_cols.add(cols[0])
            continue

        # candidate scoring
        def score(col: str) -> tuple[int, int, int]:
            # higher is better
            is_exact = int(col == canon)
            looks_snake = int((" " not in col) and (col == col.lower()))
            # shorter is better => invert length later
            return (is_exact, looks_snake, -len(col))

        best = sorted(cols, key=score, reverse=True)[0]
        keep_cols.add(best)

    # preserve original order
    ordered = [c for c in df.columns if c in keep_cols]
    return df[ordered].copy()


def _select_tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Curate the analysis view to a manageable, consistent schema.

    We keep:
    - identifiers
    - credit window label + core grouping columns
    - the ME status bucket columns
    - the outcome columns
    - a set of interpretable predictors (GPA, term rates, course mastery rollups)
    - term trajectory summaries (not the whole grid, by default)
    """
    must_have = [
        "random_id",
        "credit_window",
        "credit_window_label",
        "first_term_label",
        "me_bs_degree_status",
        "me_bs_degree_status_clean",
        "me_bs_degree_status_is_missing",
        "me_degree_status_bucket",
        "has_me_bs_degree",
        "has_any_bachelor_degree",
        "outcome_graduated_me",
        "outcome_any_bachelor",
        "outcome_status_known",
    ]

    core_predictors = [
        # GPA snapshot features
        "cgpa_at_etm_to_any_campus",
        "highest_cgpa_during_credit_window",
        "peak_minus_etm",
        "graduating_cgpa",
        "cgpa_gap",

        # stability + “struggle” signals
        "dif_btw_max_and_min_term_gpa",
        "warnings_per_term",
        "low_gpa_term_rate_2_5",
        "low_gpa_term_rate_3_0",
        "grade_forgiveness_used",
        "multi_repeat_flag",

        # simple background
        "with_math_ap",
        "sat_verb_grouping",
        "1st_aleks_math_score_grouping",
        "1st_math_course",
        "1st_math_course_campus",

        # ETM mastery rollups
        "etm_total_attempts_to_abc",
        "etm_first_attempt_pass_count",
        "etm_never_passed_count",
        "etm_first_grade_dfw_count",

        # GPA trajectory summaries
        "terms_with_gpa",
        "mean_term_gpa",
        "median_term_gpa",
        "std_term_gpa",
        "min_term_gpa",
        "max_term_gpa",
        "first_term_gpa",
        "last_term_gpa",
        "n_summer_terms",
        "n_regular_terms",
        "low_gpa_terms_2_5",
        "low_gpa_terms_3_0",
        "gpa_trend_slope",
        "first_two_regular_mean_gpa",
        "summer_term_ratio",
    ]

    # keep only those that exist
    cols = []
    for c in must_have + core_predictors:
        if c in df.columns and c not in cols:
            cols.append(c)

    # Always keep any column starting with these prefixes if present
    # (helps later when you want per-course outcomes without editing this script)
    keep_prefixes = ("chem_110_", "edsgn_100_", "math_140_", "math_141_", "phys_211_")
    for c in df.columns:
        if c.startswith(keep_prefixes) and c not in cols:
            cols.append(c)

    return df[cols].copy()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Could not find input: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    if "random_id" not in df.columns:
        raise ValueError("clean_features.csv is missing required column: random_id")

    # Deduplicate by random_id (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["random_id"], keep="first").copy()
    after = len(df)

    # Standardize credit window label
    if "credit_window" in df.columns:
        df["credit_window_label"] = df["credit_window"].map(_standardize_credit_window_label).astype("string")
    else:
        df["credit_window_label"] = pd.Series([pd.NA] * len(df), dtype="string")

    # Outcomes: always present
    df["outcome_graduated_me"] = _as_int64_bool(df.get("has_me_bs_degree", df.get("graduated_me")))
    df["outcome_any_bachelor"] = _as_int64_bool(df.get("has_any_bachelor_degree"))

    # outcome_status_known
    if "me_bs_degree_status_is_missing" in df.columns:
        miss = _as_int64_bool(df["me_bs_degree_status_is_missing"])
        df["outcome_status_known"] = (miss == 0).astype("Int64")
    else:
        known = (
            df.get("graduating_cgpa").notna()
            | df.get("me_bs_degree_status").astype("string").fillna("").str.strip().ne("")
        )
        df["outcome_status_known"] = known.astype("Int64")

    if "me_degree_status_bucket" not in df.columns:
        df["me_degree_status_bucket"] = pd.Series([pd.NA] * len(df), dtype="string")
    else:
        df["me_degree_status_bucket"] = df["me_degree_status_bucket"].astype("string")

    # ---- Full debug output (drop duplicate-named columns for readability)
    df_full = _drop_duplicate_named_columns(df)
    df_full.to_csv(OUT_CSV_FULL, index=False)

    # ---- Tidy analysis view
    df_tidy = _select_tidy_columns(df_full)
    df_tidy.to_csv(OUT_CSV_TIDY, index=False)

    try:
        df_tidy.to_parquet(OUT_PARQUET_TIDY, index=False)
    except Exception:
        pass

    # Markdown summary (based on tidy)
    lines = []
    lines.append("# analysis_v2 build summary")
    lines.append("")
    lines.append(f"- input: `{INPUT_CSV}`")
    lines.append(f"- rows (before dedupe): {before}")
    lines.append(f"- rows (after dedupe by random_id): {after}")
    lines.append("")
    lines.append("## Files written")
    lines.append("")
    lines.append(f"- `{OUT_CSV_FULL}` (debug)")
    lines.append(f"- `{OUT_CSV_TIDY}` (tidy)")
    if OUT_PARQUET_TIDY.exists():
        lines.append(f"- `{OUT_PARQUET_TIDY}` (tidy parquet)")
    lines.append("")
    lines.append("## Key columns sanity check (tidy view)")
    lines.append("")
    lines.append(_safe_value_counts(df_tidy, "credit_window_label", top_n=10))
    lines.append(_safe_value_counts(df_tidy, "me_degree_status_bucket", top_n=10))
    lines.append(_safe_value_counts(df_tidy, "outcome_graduated_me", top_n=5))
    lines.append(_safe_value_counts(df_tidy, "outcome_any_bachelor", top_n=5))
    lines.append(_safe_value_counts(df_tidy, "outcome_status_known", top_n=5))

    lines.append("## Missingness (key columns)")
    lines.append("")
    for c in ["credit_window_label", "me_degree_status_bucket", "outcome_graduated_me", "outcome_any_bachelor", "outcome_status_known"]:
        if c in df_tidy.columns:
            lines.append(f"- {c}: {float(df_tidy[c].isna().mean()):.4f}")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[ok] wrote: {OUT_CSV_FULL}")
    print(f"[ok] wrote: {OUT_CSV_TIDY}")
    print(f"[ok] wrote: {OUT_MD}")
    if OUT_PARQUET_TIDY.exists():
        print(f"[ok] wrote: {OUT_PARQUET_TIDY}")


if __name__ == "__main__":
    main()
