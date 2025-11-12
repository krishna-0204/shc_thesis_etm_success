# src/analysis/gpa_trajectory_trends.py
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from .utils import (
    load_frames, OUTCOME_COL, INSIGHTS_DIR, ensure_binary,
    DEFAULT_CLEAN, DEFAULT_LONG
)

_TERM_ORDER = {
    "1st Fall":0,"1st Spring":1,"1st Summer":2,
    "2nd Fall":3,"2nd Spring":4,"2nd Summer":5,
    "3rd Fall":6,"3rd Spring":7,"3rd Summer":8,
    "4th Fall":9,"4th Spring":10,"4th Summer":11,
    "5th Fall":12,"5th Spring":13,"5th Summer":14,
    "6th Fall":15,"6th Spring":16,"6th Summer":17,
    "7th Fall":18,"7th Spring":19,"7th Summer":20,
    "8th Fall":21,"8th Spring":22,"8th Summer":23,
    "9th Fall":24
}

def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    xm, ym = x.mean(), y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = ((x - xm) ** 2).sum()
    return float(num / den) if den > 0 else np.nan

def _prepare_long(df_long: pd.DataFrame) -> pd.DataFrame:
    L = df_long.copy()
    if "term_index" not in L.columns and "term_slot" in L.columns:
        L["term_index"] = L["term_slot"].map(_TERM_ORDER)
    return L.dropna(subset=["term_index","term_gpa"])

def compute_gpa_trends(df: pd.DataFrame, df_long: pd.DataFrame,
                       outcome_col: str = OUTCOME_COL) -> dict[str, pd.DataFrame]:
    y = ensure_binary(df[outcome_col]) if outcome_col in df.columns else None
    L = _prepare_long(df_long)
    slopes = (L.sort_values(["random_id", "term_index"])
                .groupby("random_id")
                .apply(lambda d: _slope(
                    d["term_index"].to_numpy(dtype=float),
                    d["term_gpa"].to_numpy(dtype=float)
                ), include_groups=False)
                .reset_index(name="gpa_slope"))
    per_student = slopes
    if y is not None:
        per_student = per_student.merge(df[["random_id", outcome_col]], on="random_id", how="left")
        per_student.rename(columns={outcome_col: "graduated_me"}, inplace=True)

    agg = pd.DataFrame({
        "n_students": [int(slopes["gpa_slope"].notna().sum())],
        "median_slope": [float(slopes["gpa_slope"].median())],
        "pct_positive_slope": [float((slopes["gpa_slope"] > 0).mean())]
    })

    slope_vs_grad = None
    if y is not None:
        s = per_student.dropna(subset=["gpa_slope", "graduated_me"]).copy()
        s["slope_pos"] = (s["gpa_slope"] > 0).astype(int)
        tab = s.groupby("slope_pos")["graduated_me"].agg(["count","mean"]).reset_index()
        tab.rename(columns={"mean": "grad_rate"}, inplace=True)
        slope_vs_grad = tab

    return {"per_student": per_student, "aggregates": agg, "slope_vs_grad": slope_vs_grad}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(DEFAULT_CLEAN))
    ap.add_argument("--long",  default=str(DEFAULT_LONG))
    ap.add_argument("--out_prefix", default=str(INSIGHTS_DIR / "ins_gpa"))
    ap.add_argument("--outcome", default=OUTCOME_COL)
    args = ap.parse_args()

    df, df_long = load_frames(args.clean, args.long)
    if df_long is None:
        raise SystemExit("Long per-term file not found.")
    res = compute_gpa_trends(df, df_long, outcome_col=args.outcome)
    res["per_student"].to_csv(f"{args.out_prefix}_per_student.csv", index=False)
    res["aggregates"].to_csv(f"{args.out_prefix}_aggregates.csv", index=False)
    if res["slope_vs_grad"] is not None:
        res["slope_vs_grad"].to_csv(f'{args.out_prefix}_slope_vs_grad.csv', index=False)
    print("[gpa_trends] wrote CSVs with prefix", args.out_prefix)

if __name__ == "__main__":
    main()
