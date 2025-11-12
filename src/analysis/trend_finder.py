from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from .utils import (
    INSIGHTS_DIR, OUTCOME_COL, DEFAULT_CLEAN, DEFAULT_LONG, load_frames
)
from .effect_sizes import compute_effect_sizes
from .risk_ratios import compute_rr
from .bucketing import bucket_summary
# lazy-import gpa_trends inside the function to avoid circular import when running that module as __main__

SANITIZED = INSIGHTS_DIR / "_sanitized_clean.csv"

def _prefer_sanitized(clean_path: str | None) -> str:
    """Prefer sanitized file if it exists; else use provided or DEFAULT_CLEAN."""
    if SANITIZED.exists():
        return str(SANITIZED)
    if clean_path:
        return clean_path
    return str(DEFAULT_CLEAN)

def _bucket_spreads(bucket_tbl: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    For each feature, compute grad-rate spread across its buckets and return top_k by spread.
    Assumes columns: feature, bucket, n, grad_rate
    """
    if bucket_tbl is None or bucket_tbl.empty:
        return pd.DataFrame()
    agg = (bucket_tbl
           .groupby("feature", as_index=False)
           .agg(n_total=("n", "sum"),
                buckets=("bucket", "nunique"),
                min_rate=("grad_rate", "min"),
                max_rate=("grad_rate", "max")))
    agg["spread"] = agg["max_rate"] - agg["min_rate"]
    # prefer well-powered features
    agg = agg.sort_values(["spread", "n_total"], ascending=[False, False]).head(top_k)
    return agg

def summarize_trends(df: pd.DataFrame,
                     df_long: pd.DataFrame | None,
                     outcome_col: str = OUTCOME_COL,
                     *,
                     num_k: int = 12,
                     cat_k: int = 12,
                     bucket_k: int = 5,
                     topn: int = 100) -> tuple[pd.DataFrame, str]:
    """
    Build many more bullets:
      - top `num_k` numeric effects (Cohen's d)
      - top `cat_k` categorical RRs
      - top `bucket_k` features by bucketed grad-rate spread
      - optional GPA-slope vs graduation
    Returns a combined table and markdown bullets (capped at `topn`).
    """

    # Numeric effect sizes (take more rows, not just 6/4)
    d_tbl  = compute_effect_sizes(df, outcome_col=outcome_col).head(num_k).copy()
    d_tbl["kind"] = "numeric_effect"

    # Categorical RRs
    rr_tbl = compute_rr(df, outcome_col=outcome_col).head(cat_k).copy()
    rr_tbl["kind"] = "categorical_rr"

    # Bucketed summaries (all numerics)
    bucket_tbl = bucket_summary(df, outcome_col=outcome_col)
    # top features by spread
    bucket_tops = _bucket_spreads(bucket_tbl, top_k=bucket_k)
    bucket_tops["kind"] = "bucket_spread"

    # GPA slope vs graduation (lazy import avoids circular import when running gpa module standalone)
    slope_vs_grad = None
    if df_long is not None:
        try:
            from .gpa_trajectory_trends import compute_gpa_trends
            gpa_dict = compute_gpa_trends(df, df_long, outcome_col=outcome_col)
            slope_vs_grad = gpa_dict.get("slope_vs_grad")
        except Exception:
            slope_vs_grad = None

    # --------- Bullets
    bullets: list[str] = []

    # Numeric bullets
    for _, r in d_tbl.iterrows():
        dir_ = "higher" if r["cohens_d"] > 0 else "lower"
        bullets.append(
            f"Students who graduate have {dir_} **{r['feature']}** on average "
            f"(Cohen’s d = {r['cohens_d']:.2f}, n1={int(r.get('n1', 0))}, n0={int(r.get('n0', 0))})."
        )

    # Categorical bullets
    for _, r in rr_tbl.iterrows():
        more = "higher" if r["risk_ratio"] > 1 else "lower"
        bullets.append(
            f"Being **{r['feature']} = {r['level']}** is associated with {more} graduation odds "
            f"(RR={r['risk_ratio']:.2f}, 95% CI [{r['rr_low']:.2f}, {r['rr_high']:.2f}], n={int(r['n_level'])})."
        )

    # Bucket-spread bullets (top features by separation across buckets)
    if bucket_tops is not None and not bucket_tops.empty:
        for _, r in bucket_tops.iterrows():
            bullets.append(
                f"**{r['feature']}** shows strong separation across its buckets: "
                f"graduation rates range from {r['min_rate']:.1%} to {r['max_rate']:.1%} "
                f"(spread {r['spread']:.1%}, buckets={int(r['buckets'])}, total n={int(r['n_total'])})."
            )
        # Special-case: early GPA ladder if present
        if "first_two_regular_mean_gpa" in (bucket_tbl["feature"].unique() if bucket_tbl is not None else []):
            early = bucket_tbl[bucket_tbl["feature"].eq("first_two_regular_mean_gpa")].copy()
            if not early.empty:
                # show full ladder in correct order
                ladder = (early.sort_values("bucket")
                          .apply(lambda x: f"{x['bucket']}: {x['grad_rate']:.0%}", axis=1)
                          .tolist())
                bullets.append("Early GPA ladder — " + " → ".join(ladder))

    # GPA slope bullet
    if slope_vs_grad is not None and not slope_vs_grad.empty:
        try:
            pos = slope_vs_grad.loc[slope_vs_grad["slope_pos"] == 1, "grad_rate"].iloc[0]
            neg = slope_vs_grad.loc[slope_vs_grad["slope_pos"] == 0, "grad_rate"].iloc[0]
            bullets.append(f"**Positive GPA slope** is associated with higher graduation "
                           f"({pos:.1%} vs {neg:.1%}).")
        except Exception:
            pass

    # Cap at topn (AFTER assembling)
    bullets = bullets[:topn]
    md = "\n".join([f"- {b}" for b in bullets]) if bullets else "_No salient trends found with current thresholds._"

    # Combined table (for spreadsheets)
    combined = pd.concat([d_tbl, rr_tbl, bucket_tops], ignore_index=True, sort=False)

    return combined, md

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(DEFAULT_CLEAN))
    ap.add_argument("--long",  default=str(DEFAULT_LONG))
    ap.add_argument("--out_csv", default=str(INSIGHTS_DIR / "top_trends_table.csv"))
    ap.add_argument("--out_md",  default=str(INSIGHTS_DIR / "top_trends.md"))
    ap.add_argument("--outcome", default=OUTCOME_COL)
    ap.add_argument("--topn", type=int, default=40, help="Max number of bullets in markdown.")
    ap.add_argument("--num_k", type=int, default=12, help="How many numeric effects to include.")
    ap.add_argument("--cat_k", type=int, default=12, help="How many categorical levels to include.")
    ap.add_argument("--bucket_k", type=int, default=5, help="How many top bucket-spread features to include.")
    args = ap.parse_args()

    clean_path = _prefer_sanitized(args.clean)
    df, df_long = load_frames(clean_path, args.long)
    if args.outcome not in df.columns:
        raise SystemExit(f"Outcome '{args.outcome}' not found in clean features at {clean_path}.")

    combo, md = summarize_trends(
        df, df_long, outcome_col=args.outcome,
        num_k=args.num_k, cat_k=args.cat_k, bucket_k=args.bucket_k, topn=args.topn
    )

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    combo.to_csv(args.out_csv, index=False)
    Path(args.out_md).write_text(md, encoding="utf-8")
    print(f"[trend_finder] wrote {args.out_csv} and {args.out_md}")

if __name__ == "__main__":
    main()
