from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outcome", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.clean)
    lines = []
    lines.append(f"# Data report\n\n**Rows**: {len(df)}  **Cols**: {df.shape[1]}\n")

    if args.outcome not in df.columns:
        lines.append(f"\n⚠️ Outcome '{args.outcome}' not found.\n")
    else:
        vc = df[args.outcome].value_counts(dropna=False)
        lines.append("\n## Outcome counts\n")
        for k, v in vc.items():
            lines.append(f"- {k}: {int(v)}")

    lines.append("\n## Missingness (top 25)\n")
    miss = df.isna().mean().sort_values(ascending=False).head(25)
    lines.extend([f"- {c}: {v:.1%}" for c, v in miss.items()])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"[validate] wrote {args.out}")

if __name__ == "__main__":
    main()
