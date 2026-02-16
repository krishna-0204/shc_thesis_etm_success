from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .effect_sizes import compute_effect_sizes
from .risk_ratios import compute_rr
from .bucketing import bucket_summary
from .utils import load_frames

def bucket_spreads(bucket_tbl: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    if bucket_tbl is None or bucket_tbl.empty:
        return pd.DataFrame()
    agg = bucket_tbl.groupby("feature", as_index=False).agg(
        n_total=("n", "sum"),
        buckets=("bucket", "nunique"),
        min_rate=("y_rate", "min"),
        max_rate=("y_rate", "max"),
    )
    agg["spread"] = agg["max_rate"] - agg["min_rate"]
    return agg.sort_values(["spread", "n_total"], ascending=[False, False]).head(top_k)

def summarize(df: pd.DataFrame, outcome_col: str, *, num_k: int = 15, cat_k: int = 15, bucket_k: int = 10, topn: int = 60):
    d_tbl = compute_effect_sizes(df, outcome_col=outcome_col).head(num_k).copy()
    d_tbl["kind"] = "numeric_effect"

    rr_tbl = compute_rr(df, outcome_col=outcome_col).head(cat_k).copy()
    rr_tbl["kind"] = "categorical_rr"

    b_tbl = bucket_summary(df, outcome_col=outcome_col)
    b_top = bucket_spreads(b_tbl, top_k=bucket_k).copy()
    if not b_top.empty:
        b_top["kind"] = "bucket_spread"

    bullets: list[str] = []

    for _, r in d_tbl.iterrows():
        direction = "higher" if r["cohens_d"] > 0 else "lower"
        bullets.append(
            f"Outcome=1 has {direction} **{r['feature']}** on average "
            f"(Cohen’s d={r['cohens_d']:.2f}, n1={int(r.get('n1', 0))}, n0={int(r.get('n0', 0))})."
        )

    for _, r in rr_tbl.iterrows():
        more = "higher" if r["risk_ratio"] > 1 else "lower"
        bullets.append(
            f"Being **{r['feature']}={r['level']}** is associated with {more} outcome=1 odds "
            f"(RR={r['risk_ratio']:.2f}, 95% CI [{r['rr_low']:.2f}, {r['rr_high']:.2f}], n={int(r['n_level'])})."
        )

    for _, r in b_top.iterrows():
        bullets.append(
            f"**{r['feature']}** separates across buckets: rate ranges "
            f"{r['min_rate']:.1%}–{r['max_rate']:.1%} (spread {r['spread']:.1%}, total n={int(r['n_total'])})."
        )

    bullets = bullets[:topn]
    md = "\n".join(f"- {b}" for b in bullets) if bullets else "_No salient trends found._"

    combo = pd.concat([d_tbl, rr_tbl, b_top], ignore_index=True, sort=False)
    return combo, md

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--outcome", required=True)
    ap.add_argument("--num_k", type=int, default=15)
    ap.add_argument("--cat_k", type=int, default=15)
    ap.add_argument("--bucket_k", type=int, default=10)
    ap.add_argument("--topn", type=int, default=60)
    args = ap.parse_args()

    df, _ = load_frames(args.clean, None, outcome_col=args.outcome)
    if args.outcome not in df.columns:
        raise SystemExit(f"Outcome '{args.outcome}' not found.")
    combo, md = summarize(df, args.outcome, num_k=args.num_k, cat_k=args.cat_k, bucket_k=args.bucket_k, topn=args.topn)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    combo.to_csv(args.out_csv, index=False)
    Path(args.out_md).write_text(md, encoding="utf-8")
    print(f"[trend_finder] wrote {args.out_csv} and {args.out_md}")

if __name__ == "__main__":
    main()
