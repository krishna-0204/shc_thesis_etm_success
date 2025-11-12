from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from .utils import (load_frames, OUTCOME_COL, ensure_binary, INSIGHTS_DIR,
                    DEFAULT_CLEAN)

def slice_stability(df: pd.DataFrame, outcome_col: str,
                    feature: str, confounders: list[str], min_n: int = 50) -> pd.DataFrame:
    """Check whether the direction of association holds across confounder slices."""
    y = ensure_binary(df[outcome_col])
    rows = []
    for conf in confounders:
        if conf not in df.columns:
            continue
        levels = (df[conf].astype("string").value_counts().index.tolist())[:12]
        for lev in levels:
            dsub = df[df[conf].astype("string") == lev]
            if len(dsub) < min_n:
                continue
            ysub = ensure_binary(dsub[outcome_col])
            if pd.api.types.is_numeric_dtype(dsub[feature]):
                a = dsub.loc[ysub == 1, feature].dropna()
                b = dsub.loc[ysub == 0, feature].dropna()
                if len(a) >= 20 and len(b) >= 20:
                    gap = (a.mean() - b.mean())
                    rows.append({"confounder": conf, "level": lev, "metric": "mean_diff", "value": gap, "n": len(dsub)})
            else:
                s = dsub[feature].astype("string")
                top = s.value_counts()
                if top.empty:
                    continue
                top_lev = top.index[0]
                mask = (s == top_lev)
                n1 = int((ysub[mask] == 1).sum())
                n0 = int((ysub[~mask] == 1).sum())
                rr = ( (n1+0.5) / (mask.sum()+1.0) ) / ( (n0+0.5) / ((~mask).sum()+1.0) )
                rows.append({"confounder": conf, "level": lev, "metric": f"RR({top_lev})", "value": float(rr), "n": len(dsub)})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(DEFAULT_CLEAN))
    ap.add_argument("--out", default=str(INSIGHTS_DIR / "slice_stability.csv"))
    ap.add_argument("--outcome", default=OUTCOME_COL)
    ap.add_argument("--feature", required=True, help="feature to stress-test")
    ap.add_argument("--confounders", nargs="+", default=["with_math_ap","sat_verb_grouping","summer_start","1st_term"])
    args = ap.parse_args()

    df, _ = load_frames(args.clean, None)
    if args.outcome not in df.columns:
        raise SystemExit(f"Outcome '{args.outcome}' not found.")
    if args.feature not in df.columns:
        raise SystemExit(f"Feature '{args.feature}' not found.")
    tab = slice_stability(df, args.outcome, args.feature, args.confounders)
    tab.to_csv(args.out, index=False)
    print(f"[slices] wrote {args.out} ({len(tab)} rows)")

if __name__ == "__main__":
    main()
