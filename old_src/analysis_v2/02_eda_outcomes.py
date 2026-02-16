#!/usr/bin/env python3
"""
02_eda_outcomes.py

First-pass EDA on outcomes using analysis_v2/out/analysis_view.csv.

Writes to analysis_v2/out/:
  - 02_outcomes_eda.md
  - 02_base_rates_overall.csv
  - 02_base_rates_by_credit_window.csv
  - 02_base_rates_by_bucket.csv
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_CSV = PROJECT_ROOT / "src" / "analysis_v2" / "out" / "analysis_view.csv"

OUTDIR = PROJECT_ROOT / "src" / "analysis_v2" / "out"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_MD = OUTDIR / "02_outcomes_eda.md"
OUT_OVERALL = OUTDIR / "02_base_rates_overall.csv"
OUT_CW = OUTDIR / "02_base_rates_by_credit_window.csv"
OUT_BUCKET = OUTDIR / "02_base_rates_by_bucket.csv"


def _rate(series: pd.Series) -> float:
    """Mean of 0/1 with NaNs ignored."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return float("nan")
    return float(s.mean())


def _counts(series: pd.Series) -> tuple[int, int, int]:
    """Return (n_total, n_nonmissing, n_missing)."""
    n_total = int(len(series))
    n_missing = int(series.isna().sum())
    n_nonmissing = n_total - n_missing
    return n_total, n_nonmissing, n_missing


def base_rate_table(df: pd.DataFrame, outcome_col: str, group_col: str | None = None) -> pd.DataFrame:
    """
    Build a base-rate table:
      - group value
      - n
      - outcome_rate
    """
    if group_col is None:
        n = len(df)
        return pd.DataFrame(
            [{
                "group": "overall",
                "n": int(n),
                "outcome_rate": _rate(df[outcome_col]),
            }]
        )

    g = (
        df.groupby(group_col, dropna=False)[outcome_col]
        .agg(n="size", outcome_rate=_rate)
        .reset_index()
        .rename(columns={group_col: "group"})
        .sort_values("n", ascending=False)
    )

    # make group readable for NaNs
    g["group"] = g["group"].astype("string").fillna("missing")
    return g


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing analysis view: {IN_CSV}. Run 00_build_analysis_view.py first.")

    df = pd.read_csv(IN_CSV)

    required = [
        "random_id",
        "credit_window_label",
        "me_degree_status_bucket",
        "outcome_graduated_me",
        "outcome_any_bachelor",
        "outcome_status_known",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"analysis_view.csv missing required columns: {missing}")

    # Basic sanity
    n_rows = len(df)
    n_ids = int(df["random_id"].nunique(dropna=True))

    # Overall base rates
    overall_me = base_rate_table(df, "outcome_graduated_me", None)
    overall_any = base_rate_table(df, "outcome_any_bachelor", None)
    overall_known = base_rate_table(df, "outcome_status_known", None)

    overall = pd.DataFrame(
        [
            {"metric": "n_rows", "value": n_rows},
            {"metric": "n_unique_random_id", "value": n_ids},
            {"metric": "rate_graduated_me", "value": _rate(df["outcome_graduated_me"])},
            {"metric": "rate_any_bachelor", "value": _rate(df["outcome_any_bachelor"])},
            {"metric": "rate_status_known", "value": _rate(df["outcome_status_known"])},
        ]
    )
    overall.to_csv(OUT_OVERALL, index=False)

    # By credit window
    cw_me = base_rate_table(df, "outcome_graduated_me", "credit_window_label")
    cw_any = base_rate_table(df, "outcome_any_bachelor", "credit_window_label")
    cw_known = base_rate_table(df, "outcome_status_known", "credit_window_label")

    cw = cw_me.rename(columns={"outcome_rate": "rate_graduated_me"}).merge(
        cw_any.rename(columns={"outcome_rate": "rate_any_bachelor"}), on=["group", "n"], how="outer"
    ).merge(
        cw_known.rename(columns={"outcome_rate": "rate_status_known"}), on=["group", "n"], how="outer"
    ).rename(columns={"group": "credit_window_label"}).sort_values("n", ascending=False)

    cw.to_csv(OUT_CW, index=False)

    # By degree bucket
    b_me = base_rate_table(df, "outcome_graduated_me", "me_degree_status_bucket")
    b_any = base_rate_table(df, "outcome_any_bachelor", "me_degree_status_bucket")
    b_known = base_rate_table(df, "outcome_status_known", "me_degree_status_bucket")

    b = b_me.rename(columns={"outcome_rate": "rate_graduated_me"}).merge(
        b_any.rename(columns={"outcome_rate": "rate_any_bachelor"}), on=["group", "n"], how="outer"
    ).merge(
        b_known.rename(columns={"outcome_rate": "rate_status_known"}), on=["group", "n"], how="outer"
    ).rename(columns={"group": "me_degree_status_bucket"}).sort_values("n", ascending=False)

    b.to_csv(OUT_BUCKET, index=False)

    # Markdown report (quick read)
    lines = []
    lines.append("# Outcomes EDA (analysis_v2)")
    lines.append("")
    lines.append(f"- input: `{IN_CSV}`")
    lines.append(f"- rows: {n_rows}")
    lines.append(f"- unique random_id: {n_ids}")
    lines.append("")
    lines.append("## Overall base rates")
    lines.append("")
    lines.append(f"- graduated_me: {_rate(df['outcome_graduated_me']):.4f}")
    lines.append(f"- any_bachelor: {_rate(df['outcome_any_bachelor']):.4f}")
    lines.append(f"- status_known: {_rate(df['outcome_status_known']):.4f}")
    lines.append("")
    lines.append("## Base rates by credit_window_label")
    lines.append("")
    lines.append(cw.to_markdown(index=False))
    lines.append("")
    lines.append("## Base rates by me_degree_status_bucket")
    lines.append("")
    lines.append(b.to_markdown(index=False))
    lines.append("")
    lines.append("## Files written")
    lines.append("")
    lines.append(f"- `{OUT_MD}`")
    lines.append(f"- `{OUT_OVERALL}`")
    lines.append(f"- `{OUT_CW}`")
    lines.append(f"- `{OUT_BUCKET}`")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[ok] wrote: {OUT_MD}")
    print(f"[ok] wrote: {OUT_OVERALL}")
    print(f"[ok] wrote: {OUT_CW}")
    print(f"[ok] wrote: {OUT_BUCKET}")


if __name__ == "__main__":
    main()
