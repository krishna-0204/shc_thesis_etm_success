from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from .utils import load_frames, categorical_features, rr_ci, score_rr, ensure_binary, MIN_N

def compute_rr(df: pd.DataFrame, outcome_col: str, exclude: set[str] | None = None, min_level_n: int = MIN_N) -> pd.DataFrame:
    exclude = set(exclude or []) | {"random_id", "merge_id", outcome_col}
    y = ensure_binary(df[outcome_col])
    rows = []
    for c in categorical_features(df, exclude):
        s = df[c].astype("string")
        vc = s.value_counts(dropna=True)
        levels = [lev for lev, cnt in vc.items() if int(cnt) >= int(min_level_n)]
        if not levels:
            continue
        for lev in levels:
            mask = (s == lev)
            n_level = int(mask.sum())
            n_rest = int((~mask).sum())
            a1 = int((y[mask] == 1).sum())
            b1 = int((y[~mask] == 1).sum())
            rr, lo, hi, p1, p0 = rr_ci(a1, n_level, b1, n_rest, smooth=0.5)
            rows.append({
                "feature": c,
                "level": lev,
                "n_level": n_level,
                "n_rest": n_rest,
                "y_rate_level": p1,
                "y_rate_rest": p0,
                "risk_ratio": rr,
                "rr_low": lo,
                "rr_high": hi,
                "abs_log_rr": abs(float(np.log(rr))),
                "score": score_rr(abs(float(np.log(rr))), n_level, n_rest),
            })
    out = pd.DataFrame(rows)
    return out.sort_values("score", ascending=False) if not out.empty else out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outcome", required=True)
    ap.add_argument("--min_level_n", type=int, default=MIN_N)
    args = ap.parse_args()

    df, _ = load_frames(args.clean, None, outcome_col=args.outcome)
    if args.outcome not in df.columns:
        raise SystemExit(f"Outcome '{args.outcome}' not in columns.")
    tbl = compute_rr(df, outcome_col=args.outcome, min_level_n=args.min_level_n)
    tbl.to_csv(args.out, index=False)
    print(f"[risk_ratios] wrote {args.out} ({len(tbl)} rows)")

if __name__ == "__main__":
    main()
