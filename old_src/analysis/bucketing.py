from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from .utils import (load_frames, numeric_features, OUTCOME_COL, quantile_buckets,
                    INSIGHTS_DIR, ensure_binary, DEFAULT_CLEAN)

def bucket_summary(df: pd.DataFrame, outcome_col: str = OUTCOME_COL,
                   cols: list[str] | None = None, q=(0, .25, .5, .75, 1.0)) -> pd.DataFrame:
    y = ensure_binary(df[outcome_col])
    cols = cols or numeric_features(df, exclude={outcome_col, "random_id", "merge_id"})
    rows = []
    for c in cols:
        if c == outcome_col:
            continue
        b, edges = quantile_buckets(df[c], q=q)
        if b.isna().all():
            continue
        tmp = pd.DataFrame({"bucket": b, "y": y}).dropna(subset=["bucket"])
        tab = (tmp.groupby("bucket", observed=True, sort=False)["y"]
                .agg(["count","mean"])
                .reset_index())

        for _, r in tab.iterrows():
            rows.append({
                "feature": c, "bucket": str(r["bucket"]),
                "n": int(r["count"]), "grad_rate": float(r["mean"]),
                "edges": "|".join(map(str, edges))
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(["feature", "bucket"], inplace=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(DEFAULT_CLEAN))
    ap.add_argument("--out",   default=str(INSIGHTS_DIR / "ins_bucketed_grad_rates.csv"))
    ap.add_argument("--outcome", default=OUTCOME_COL)
    args = ap.parse_args()

    df, _ = load_frames(args.clean, None)
    if args.outcome not in df.columns:
        raise SystemExit(f"Outcome '{args.outcome}' not in columns.")
    tbl = bucket_summary(df, outcome_col=args.outcome)
    tbl.to_csv(args.out, index=False)
    print(f"[bucketing] wrote {args.out} ({len(tbl)} rows)")

if __name__ == "__main__":
    main()
