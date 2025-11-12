from __future__ import annotations
import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from .utils import INSIGHTS_DIR, OUTCOME_COL

PLOTS_DIR = INSIGHTS_DIR / "plots"

def _safe_bins(x: np.ndarray) -> int:
    n = x.size
    if n <= 1:
        return 10
    return max(10, min(60, int(np.sqrt(n))))

def _safe_name(name: str) -> str:
    # keep letters, numbers, dot, dash, underscore; everything else -> underscore
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))[:120]

def plot_numeric(feature: str, df: pd.DataFrame, yname: str):
    s = pd.to_numeric(df.get(feature, pd.Series(dtype=float)), errors="coerce").dropna()
    if s.empty:
        return []
    y = (pd.to_numeric(df.loc[s.index, yname], errors="coerce") == 1).astype(int)

    files = []

    # Overall histogram
    plt.figure()
    plt.hist(s.values, bins=_safe_bins(s.values))
    plt.xlabel(feature); plt.ylabel("Count"); plt.title(f"{feature} — overall")
    fn1 = PLOTS_DIR / f"{_safe_name(feature)}__hist_overall.png"
    plt.tight_layout(); plt.savefig(fn1); plt.close()
    files.append(fn1)

    # By outcome overlays (only if both groups non-empty)
    a = s[y == 1].values
    b = s[y == 0].values
    if a.size > 0 and b.size > 0:
        plt.figure()
        plt.hist(a, bins=_safe_bins(a), alpha=0.6, label="graduated=1")
        plt.hist(b, bins=_safe_bins(b), alpha=0.6, label="graduated=0")
        plt.xlabel(feature); plt.ylabel("Count"); plt.title(f"{feature} — by outcome")
        plt.legend()
        fn2 = PLOTS_DIR / f"{_safe_name(feature)}__hist_by_outcome.png"
        plt.tight_layout(); plt.savefig(fn2); plt.close()
        files.append(fn2)

    return files

def plot_categorical(feature: str, df: pd.DataFrame, yname: str, max_levels=25):
    if feature not in df.columns:
        return []
    s = df[feature].astype("string")
    vc = s.value_counts(dropna=True)
    if vc.empty:
        return []

    # focus on top-N levels by frequency
    top_levels = vc.head(max_levels).index.tolist()
    sub = df[s.isin(top_levels)].copy()
    if sub.empty or yname not in sub.columns:
        return []

    files = []

    # Overall counts bar
    cnt = sub[feature].astype("string").value_counts()
    plt.figure()
    plt.bar(cnt.index, cnt.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count"); plt.title(f"{feature} — overall (top {len(cnt)})")
    fn1 = PLOTS_DIR / f"{_safe_name(feature)}__bar_overall.png"
    plt.tight_layout(); plt.savefig(fn1); plt.close()
    files.append(fn1)

    # Graduation rate per level
    sub_y = (pd.to_numeric(sub[yname], errors="coerce") == 1).astype(int)
    grad_rate = sub.groupby(sub[feature].astype("string"))[sub_y.name].mean().sort_values(ascending=False)
    plt.figure()
    plt.bar(grad_rate.index, grad_rate.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Graduation rate"); plt.title(f"{feature} — grad rate by level")
    fn2 = PLOTS_DIR / f"{_safe_name(feature)}__bar_grad_rate.png"
    plt.tight_layout(); plt.savefig(fn2); plt.close()
    files.append(fn2)

    return files

def select_top_features(numeric_csv: Path, categorical_csv: Path, k_num=10, k_cat=10):
    top_num, top_cat = [], []
    if numeric_csv.exists():
        d = pd.read_csv(numeric_csv)
        if not d.empty and "feature" in d.columns and "score" in d.columns:
            d = d.sort_values("score", ascending=False)
            top_num = d["feature"].dropna().drop_duplicates().head(k_num).tolist()
    if categorical_csv.exists():
        c = pd.read_csv(categorical_csv)
        if not c.empty and "feature" in c.columns and "score" in c.columns:
            c = c.sort_values("score", ascending=False)
            top_cat = c["feature"].dropna().drop_duplicates().head(k_cat).tolist()
    return top_num, top_cat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(INSIGHTS_DIR / "_sanitized_clean.csv"))
    ap.add_argument("--numeric_ins", default=str(INSIGHTS_DIR / "ins_numeric_cohens_d.csv"))
    ap.add_argument("--cat_ins", default=str(INSIGHTS_DIR / "ins_categorical_rr.csv"))
    ap.add_argument("--outcome", default=OUTCOME_COL)
    ap.add_argument("--k_num", type=int, default=10)
    ap.add_argument("--k_cat", type=int, default=10)
    args = ap.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.clean)
    if args.outcome not in df.columns:
        raise SystemExit(f"Outcome '{args.outcome}' not found in {args.clean}")

    top_num, top_cat = select_top_features(Path(args.numeric_ins), Path(args.cat_ins), args.k_num, args.k_cat)

    lines = ["# Feature distributions (auto)\n", f"_Outcome_: **{args.outcome}**\n"]

    # Plot numerics
    for f in top_num:
        try:
            files = plot_numeric(f, df, args.outcome)
            if files:
                rel = [f"![](plots/{Path(x).name})" for x in files]
                lines.append(f"- **{f}**: " + ", ".join(rel))
        except Exception:
            # keep going even if one feature fails to plot
            continue

    # Plot categoricals (avoid duplicates)
    seen = set(top_num)
    for f in top_cat:
        if f in seen:
            continue
        try:
            files = plot_categorical(f, df, args.outcome)
            if files:
                rel = [f"![](plots/{Path(x).name})" for x in files]
                lines.append(f"- **{f}**: " + ", ".join(rel))
        except Exception:
            continue

    (INSIGHTS_DIR / "feature_distributions.md").write_text("\n".join(lines), encoding="utf-8")

    # FIX: Path.glob returns an iterator — cast to list before len
    png_files = list(PLOTS_DIR.glob("*.png"))
    print(f"[distributions] wrote {len(png_files)} PNGs and feature_distributions.md")

if __name__ == "__main__":
    main()
