from __future__ import annotations
import pandas as pd

def basic_sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of row-level flags for common issues. O(N)."""
    checks = pd.DataFrame(index=df.index)
    if "graduating_cgpa" in df and "cgpa_at_etm_to_any_campus" in df:
        checks["cgpa_negative_gap"] = (df["graduating_cgpa"] < df["cgpa_at_etm_to_any_campus"]).astype("Int64")
    if "no_enrolled_terms_su_included" in df:
        checks["enrolled_terms_zero_or_neg"] = (df["no_enrolled_terms_su_included"] <= 0).astype("Int64")
    # Add more as needed
    return checks
