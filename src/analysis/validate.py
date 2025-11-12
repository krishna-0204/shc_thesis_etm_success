from __future__ import annotations
import argparse
import pandas as pd
from .utils import DEFAULT_CLEAN, DEFAULT_LONG, INSIGHTS_DIR, OUTCOME_COL

RECOMMENDED_COLS = [
    OUTCOME_COL, "cgpa_at_etm_to_any_campus","graduating_cgpa",
    "first_two_regular_mean_gpa","mean_term_gpa","warnings_per_term",
    "grade_forgiveness_used","multi_repeat_flag","terms_with_gpa"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(DEFAULT_CLEAN))
    ap.add_argument("--long",  default=str(DEFAULT_LONG))
    ap.add_argument("--out",   default=str(INSIGHTS_DIR / "_data_report.md"))
    args = ap.parse_args()

    df = pd.read_csv(args.clean)
    lines = []
    lines.append(f"# Data report\n\n**Rows**: {len(df)}  **Cols**: {df.shape[1]}\n")
    lines.append("## Missingness (top 20)\n")
    miss = df.isna().mean().sort_values(ascending=False).head(20)
    lines.extend([f"- {c}: {v:.1%}" for c, v in miss.items()])
    lines.append("\n## Recommended columns present\n")
    for c in RECOMMENDED_COLS:
        lines.append(f"- {c}: {'✅' if c in df.columns else '⚠️ missing'}")
    try:
        dfl = pd.read_csv(args.long)
        lines.append(f"\n**Long table**: {len(dfl)} rows, cols={list(dfl.columns)}")
    except Exception:
        lines.append("\n**Long table**: not found")

    out = "\n".join(lines)
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"[validate] wrote {args.out}")

if __name__ == "__main__":
    main()
