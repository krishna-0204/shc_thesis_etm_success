# gpa_features.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from .cleaning import standardize_columns, to_numeric
from .features import decode_psu_term  # if you put it in features.py; else import from terms.py

# columns expected (after standardization)
SEM_COLS_ORDERED = [
    "1st_fall","1st_spring","1st_summer",
    "2nd_fall","2nd_spring","2nd_summer",
    "3rd_fall","3rd_spring","3rd_summer",
    "4th_fall","4th_spring","4th_summer",
    "5th_fall","5th_spring","5th_summer",
    "6th_fall","6th_spring","6th_summer",
    "7th_fall","7th_spring","7th_summer",
    "8th_fall","8th_spring","8th_summer",
    "9th_fall"
]

def _coerce_gpa_cols(df: pd.DataFrame) -> pd.DataFrame:
    gpa_cols = [c for c in SEM_COLS_ORDERED if c in df.columns]
    to_numeric(df, gpa_cols)
    return df, gpa_cols

def build_term_gpa_features(gpa_sheet: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - wide_features: per-student engineered features
      - long_terms: long table with one row per (student, term_index) with GPA & labels
    """
    df = standardize_columns(gpa_sheet).rename(columns={"random_id":"random_id", "1st_term":"first_term"})
    df, gpa_cols = _coerce_gpa_cols(df)

    # --- decode 1st term
    # keep original code too if needed
    decoded = df["first_term"].apply(lambda x: decode_psu_term(x)[0] if pd.notna(x) else None)
    df["first_term_label"] = decoded.astype("string")

    # --- long reshape for analysis
    present_cols = ["random_id"] + gpa_cols
    wide = df[present_cols].copy()
    long = wide.melt(id_vars=["random_id"], var_name="term_slot", value_name="term_gpa").dropna(subset=["term_gpa"])

    # assign an integer term index (ordered by SEM_COLS_ORDERED)
    order_map = {name: i + 1 for i, name in enumerate([c for c in SEM_COLS_ORDERED if c in gpa_cols])}
    long["term_index"] = long["term_slot"].map(order_map).astype(int)

    # label regular vs summer
    long["is_summer"] = long["term_slot"].str.contains("summer").astype("Int64")
    long["is_regular"] = 1 - long["is_summer"]

    # ensure deterministic order for first/last
    long = long.sort_values(["random_id", "term_index"], kind="stable")

    # --- engineered summaries (per student)
    grp = long.groupby("random_id", sort=False)

    def slope(x_idx, y):
        x = x_idx.to_numpy(dtype=float)
        y = y.to_numpy(dtype=float)
        n = x.size
        if n < 2:
            return np.nan
        xm, ym = x.mean(), y.mean()
        num = ((x - xm) * (y - ym)).sum()
        den = ((x - xm) ** 2).sum()
        return float(num / den) if den > 0 else np.nan

    agg = grp.agg(
        terms_with_gpa=("term_gpa", "count"),
        mean_term_gpa=("term_gpa", "mean"),
        median_term_gpa=("term_gpa", "median"),
        std_term_gpa=("term_gpa", "std"),
        min_term_gpa=("term_gpa", "min"),
        max_term_gpa=("term_gpa", "max"),
        first_term_gpa=("term_gpa", lambda s: s.iloc[0]),
        last_term_gpa=("term_gpa", lambda s: s.iloc[-1]),
        n_summer_terms=("is_summer", "sum"),
        n_regular_terms=("is_regular", "sum"),
        low_gpa_terms_2_5=("term_gpa", lambda s: int((s < 2.5).sum())),
        low_gpa_terms_3_0=("term_gpa", lambda s: int((s < 3.0).sum())),
    ).reset_index()

    # slope of GPA trajectory w.r.t. term order
    slope_df = grp.apply(
        lambda d: slope(d["term_index"], d["term_gpa"]), include_groups=False
    ).reset_index(name="gpa_trend_slope")

    wide_features = agg.merge(slope_df, on="random_id", how="left")

    # early-window features (first two regular terms when present)
    def first_two_regular_mean(d: pd.DataFrame):
        reg = d.loc[d["is_regular"] == 1, "term_gpa"].head(2)
        return float(reg.mean()) if len(reg) else np.nan

    early = grp.apply(first_two_regular_mean, include_groups=False).reset_index(name="first_two_regular_mean_gpa")
    wide_features = wide_features.merge(early, on="random_id", how="left")

    # proportion summer
    wide_features["summer_term_ratio"] = (
        wide_features["n_summer_terms"]
        / wide_features["terms_with_gpa"].replace(0, np.nan)
    )

    return wide_features, long[["random_id", "term_index", "term_slot", "term_gpa", "is_summer", "is_regular"]]
