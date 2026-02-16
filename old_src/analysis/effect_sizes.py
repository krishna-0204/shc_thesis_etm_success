from __future__ import annotations
import argparse
import pandas as pd
from .utils import (load_frames, numeric_features, OUTCOME_COL, MIN_N,
                    cohen_d, score_numeric, INSIGHTS_DIR, ensure_binary, DEFAULT_CLEAN)

def compute_effect_sizes(df: pd.DataFrame,
                         outcome_col: str = OUTCOME_COL,
                         exclude: set[str] = None) -> pd.DataFrame:
    exclude = set(exclude or []) | {"random_id", "merge_id", outcome_col}
    y = ensure_binary(df[outcome_col])
    rows = []
    for c in numeric_features(df, exclude):
        try:
            res = cohen_d(df[c], y)
            if not res:
                continue
            d, n1, n0 = res["d"], res["n1"], res["n0"]
            rows.append({
                "feature": c, "cohens_d": d, "abs_d": abs(d),
                "mean_grad1": res["mean1"], "mean_grad0": res["mean0"],
                "n1": n1, "n0": n0, "score": score_numeric(abs(d), n1, n0)
            })
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("score", ascending=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(DEFAULT_CLEAN))
    ap.add_argument("--out",   default=str(INSIGHTS_DIR / "ins_numeric_cohens_d.csv"))
    ap.add_argument("--outcome", default=OUTCOME_COL)
    args = ap.parse_args()

    df, _ = load_frames(args.clean, None)
    if args.outcome not in df.columns:
        raise SystemExit(f"Outcome '{args.outcome}' not in columns.")
    tbl = compute_effect_sizes(df, outcome_col=args.outcome)
    tbl.to_csv(args.out, index=False)
    print(f"[effect_sizes] wrote {args.out} ({len(tbl)} rows)")

if __name__ == "__main__":
    main()
