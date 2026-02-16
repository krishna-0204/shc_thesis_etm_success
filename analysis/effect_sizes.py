from __future__ import annotations

import argparse
import pandas as pd

from .utils import load_frames, numeric_features, cohen_d, score_numeric, ensure_binary, MIN_N

def compute_effect_sizes(df: pd.DataFrame, outcome_col: str, exclude: set[str] | None = None) -> pd.DataFrame:
    exclude = set(exclude or []) | {"random_id", "merge_id", outcome_col}
    y = ensure_binary(df[outcome_col])
    rows = []
    for c in numeric_features(df, exclude):
        try:
            res = cohen_d(pd.to_numeric(df[c], errors="coerce"), y, min_n=MIN_N)
            if not res:
                continue
            d, n1, n0 = res["d"], res["n1"], res["n0"]
            rows.append({
                "feature": c,
                "cohens_d": d,
                "abs_d": abs(d),
                "mean_y1": res["mean1"],
                "mean_y0": res["mean0"],
                "n1": n1,
                "n0": n0,
                "score": score_numeric(abs(d), n1, n0),
            })
        except Exception:
            continue
    out = pd.DataFrame(rows)
    return out.sort_values("score", ascending=False) if not out.empty else out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outcome", required=True)
    args = ap.parse_args()

    df, _ = load_frames(args.clean, None, outcome_col=args.outcome)
    if args.outcome not in df.columns:
        raise SystemExit(f"Outcome '{args.outcome}' not in columns.")
    tbl = compute_effect_sizes(df, outcome_col=args.outcome)
    tbl.to_csv(args.out, index=False)
    print(f"[effect_sizes] wrote {args.out} ({len(tbl)} rows)")

if __name__ == "__main__":
    main()
