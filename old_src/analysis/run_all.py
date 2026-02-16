from __future__ import annotations
import subprocess
from .utils import INSIGHTS_DIR, OUTCOME_COL

SANITIZED = str(INSIGHTS_DIR / "_sanitized_clean.csv")

CMDS = [
    ["python", "-m", "src.analysis.sanitizer"],

    ["python", "-m", "src.analysis.validate", "--clean", SANITIZED],
    ["python", "-m", "src.analysis.effect_sizes", "--clean", SANITIZED, "--outcome", OUTCOME_COL],
    ["python", "-m", "src.analysis.risk_ratios", "--clean", SANITIZED, "--outcome", OUTCOME_COL],
    ["python", "-m", "src.analysis.bucketing",   "--clean", SANITIZED, "--outcome", OUTCOME_COL],
    ["python", "-m", "src.analysis.gpa_trajectory_trends", "--outcome", OUTCOME_COL],
    ["python", "-m", "src.analysis.trend_finder",
        "--clean", SANITIZED,
        "--outcome", OUTCOME_COL,
        "--topn", "200",
        "--num_k", "60",        # was 16
        "--cat_k", "60",        # was 16
        "--bucket_k", "40",     # was 8
        ],
    ["python", "-m", "src.analysis.distributions", "--clean", SANITIZED, "--outcome", OUTCOME_COL],
]

def main():
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for cmd in CMDS:
        print(">>", " ".join(cmd))
        subprocess.run(cmd, check=True)
    print("[run_all] Complete.")

if __name__ == "__main__":
    main()
