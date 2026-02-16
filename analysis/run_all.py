from __future__ import annotations

import argparse
import subprocess

from .utils import INSIGHTS_DIR, outcome_set

def run(cmd: list[str]):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/processed/clean_features.csv")
    ap.add_argument("--outcomes", default="graduated_me,not_graduate_me,transferred_out_of_me")
    ap.add_argument("--min_level_n", type=int, default=25)
    ap.add_argument("--num_k", type=int, default=15)
    ap.add_argument("--cat_k", type=int, default=15)
    ap.add_argument("--bucket_k", type=int, default=10)
    ap.add_argument("--topn", type=int, default=60)
    args = ap.parse_args()

    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    outcomes = outcome_set(args.outcomes)

    for outcome in outcomes:
        out_dir = INSIGHTS_DIR / f"by_{outcome}"
        out_dir.mkdir(parents=True, exist_ok=True)

        sanitized = out_dir / "_sanitized_clean.csv"
        report = out_dir / "_sanitizer_report.json"

        run(["python", "-m", "analysis.sanitizer",
             "--clean", args.clean, "--out", str(sanitized), "--report", str(report), "--outcome", outcome])

        run(["python", "-m", "analysis.validate",
             "--clean", str(sanitized), "--out", str(out_dir / "_data_report.md"), "--outcome", outcome])

        run(["python", "-m", "analysis.effect_sizes",
             "--clean", str(sanitized), "--out", str(out_dir / "ins_numeric_cohens_d.csv"), "--outcome", outcome])

        run(["python", "-m", "analysis.risk_ratios",
             "--clean", str(sanitized), "--out", str(out_dir / "ins_categorical_rr.csv"),
             "--outcome", outcome, "--min_level_n", str(args.min_level_n)])

        run(["python", "-m", "analysis.bucketing",
             "--clean", str(sanitized), "--out", str(out_dir / "ins_bucketed_rates.csv"), "--outcome", outcome])

        run(["python", "-m", "analysis.trend_finder",
             "--clean", str(sanitized),
             "--out_csv", str(out_dir / "top_trends_table.csv"),
             "--out_md", str(out_dir / "top_trends.md"),
             "--outcome", outcome,
             "--num_k", str(args.num_k),
             "--cat_k", str(args.cat_k),
             "--bucket_k", str(args.bucket_k),
             "--topn", str(args.topn)])

    print("[run_all] Complete. Outputs in", INSIGHTS_DIR)

if __name__ == "__main__":
    main()
