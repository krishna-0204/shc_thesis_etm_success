
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .cleaning import standardize_columns, to_numeric, clean_grades  # noqa: F401


# ---------------------------
# Constants & configuration
# ---------------------------

# Passing and failing grade sets (for first-grade flags only)
PASS_GRADES = {"A", "A-", "B+", "B", "B-", "C+", "C"}
FAIL_GRADES = {"D", "F"}
WITHDRAW = {"W", "WF", "WN", "LD"}

# Mapping of canonical per-course columns after standardization
COURSE_KEYS: Dict[str, Dict[str, str]] = {
    "chem_110": {
        "first_grade": "chem_110_grade_code_1st_fa_or_sp",
        "outcome_1st_fa": "chem_110_outcome_after_1st_fa",
        "outcome_1st_ay": "chem_110_outcome_after_1st_ay",
        "outcome_2nd_ay": "chem_110_outcome_after_2nd_ay",
        "final_outcome": "chem_110_outcome",
    },
    "edsgn_100": {
        "first_grade": "edsgn_100_grade_code_1st_fa_or_sp",
        "outcome_1st_fa": "edsgn_100_outcome_after_1st_fa",
        "outcome_1st_ay": "edsgn_100_outcome_after_1st_ay",
        "outcome_2nd_ay": "edsgn_100_outcome_after_2nd_ay",
        "final_outcome": "edsgn_100_outcome",
    },
    "math_140": {
        "first_grade": "math_140_grade_code_1st_fa_or_sp",
        "outcome_1st_fa": "math_140_outcome_after_1st_fa",
        "outcome_1st_ay": "math_140_outcome_after_1st_ay",
        "outcome_2nd_ay": "math_140_outcome_after_2nd_ay",
        "final_outcome": "math_140_outcome",
    },
    "math_141": {
        "first_grade": "math_141_grade_code_1st_fa_or_sp",
        "outcome_1st_fa": "math_141_outcome_after_1st_fa",
        "outcome_1st_ay": "math_141_outcome_after_1st_ay",
        "outcome_2nd_ay": "math_141_outcome_after_2nd_ay",
        "final_outcome": "math_141_outcome",
    },
    "phys_211": {
        "first_grade": "phys_211_grade_code_1st_fa_or_sp",
        "outcome_1st_fa": "phys_211_outcome_after_1st_fa",
        "outcome_1st_ay": "phys_211_outcome_after_1st_ay",
        "outcome_2nd_ay": "phys_211_outcome_after_2nd_ay",
        "final_outcome": "phys_211_outcome",
    },
}


# ---------------------------
# Debug helpers
# ---------------------------

def print_available_columns(df: pd.DataFrame, n: int = 999) -> None:
    cols = list(df.columns)[:n]
    print(f"[debug] standardized columns ({len(cols)}): {cols}")


TERM_DIGIT_TO_NAME = {1: "Spring", 5: "Summer", 8: "Fall"}
TERM_DIGIT_TO_MONTH = {1: 1, 5: 6, 8: 8}

def decode_psu_term(term_code: int | float | str):
    import math, pandas as pd
    if pd.isna(term_code): 
        return None, None, None  # (label, year, pandas_timestamp)
    n = int(term_code)
    sem_digit = n % 10
    base = n // 10
    year = 2000 + (base - 200)
    sem_name = TERM_DIGIT_TO_NAME.get(sem_digit, f"Term{sem_digit}")
    month = TERM_DIGIT_TO_MONTH.get(sem_digit, 1)
    ts = pd.Timestamp(year=year, month=month, day=1)
    return f"{sem_name} {year}", year, ts


# ---------------------------
# Outcome utilities
# ---------------------------

def outcome_is_pass(x: object) -> bool:
    """
    Heuristic: treat anything containing PASS/A/B/C as pass, but not C-.
    Used only for the late 'final_outcome' free-text signal.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    xs = str(x).upper()
    if "C-" in xs:
        return False
    return any(tok in xs for tok in ["PASS", "A", "B", "C"])


# Patterns to parse AY outcomes (AY1 already includes FA)
ATT_PATTERNS: Dict[str, re.Pattern] = {
    "never": re.compile(r"^\s*Never\s+Enrolled\s*$", re.I),
    "first_pass": re.compile(r"^\s*1(?:st)?\s*Attempt\s*=\s*ABC\s*$", re.I),
    "only1_dfw": re.compile(r"^\s*Only\s+1\s*Attempt\s*=\s*DFW\s*$", re.I),
    "two_dfw_then_abc": re.compile(r"^\s*2\s*Attempts\s*\(\s*1st\s*=\s*DFW\s*,\s*2nd\s*=\s*ABC\s*\)\s*$", re.I),
    "two_both_dfw": re.compile(r"^\s*2\s*Attempts\s*\(\s*Both\s*=\s*DFW\s*\)\s*$", re.I),
}


def parse_period_outcome(s: object) -> Tuple[int, bool, int | None]:
    """
    Parse an AY-period outcome string to a compact representation.

    Returns:
      (attempts_in_period, pass_in_period, pass_index_in_period or None)

    Examples:
      '1st Attempt=ABC'                        -> (1, True, 1)
      'Only 1 Attempt=DFW'                     -> (1, False, None)
      '2 Attempts (1st=DFW, 2nd=ABC)'         -> (2, True, 2)
      '2 Attempts (Both=DFW)'                  -> (2, False, None)
      'Never Enrolled' / blank / NaN           -> (0, False, None)
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return 0, False, None
    txt = str(s).strip()
    if not txt or ATT_PATTERNS["never"].match(txt):
        return 0, False, None
    if ATT_PATTERNS["first_pass"].match(txt):
        return 1, True, 1
    if ATT_PATTERNS["only1_dfw"].match(txt):
        return 1, False, None
    if ATT_PATTERNS["two_dfw_then_abc"].match(txt):
        return 2, True, 2
    if ATT_PATTERNS["two_both_dfw"].match(txt):
        return 2, False, None

    # Fallbacks for unexpected phrasing
    m_pass = re.search(r"(\d+)\s*Attempts?.*ABC", txt, re.I)
    if m_pass:
        k = int(m_pass.group(1))
        return k, True, k
    m_any = re.search(r"(\d+)\s*Attempts?", txt, re.I)
    if m_any:
        return int(m_any.group(1)), False, None
    return 0, False, None  # safe default


def _label_course_outcome(total_attempts: int, pass_on: int | None) -> str:
    """
    Build the human label. If never passed, say 'No pass after N attempts' (or 'Never enrolled' if N=0).
    If passed, map attempt index to the pass-on-N wording (>=4 => 'Requires more than three attempts').
    """
    if pass_on is not None:
        if pass_on == 1: return "Pass on 1st attempt"
        if pass_on == 2: return "Pass on 2nd attempt"
        if pass_on == 3: return "Pass on 3rd attempt"
        return "Requires more than three attempts"  # passed on 4th+
    if total_attempts == 0:
        return "Never enrolled"
    return f"No pass after {total_attempts} attempt{'s' if total_attempts != 1 else ''}"



def compute_attempts_ay_model(ay1: object, ay2: object, final_outcome: object) -> Tuple[float, str, int]:
    """
    CUMULATIVE model:
      - AY1 value already includes FA (Fall).
      - AY2 value is cumulative across FA + AY1 + AY2.
      - We use the latest non-empty cumulative snapshot (prefer AY2 unless it's 'Never Enrolled'/blank).
      - If no pass in snapshots but final_outcome hints a pass, add +1 attempt to the latest attempts-so-far.
      - If still no pass, label 'No pass after N attempts' (or 'Never enrolled' if N=0).

    Returns:
      attempts_to_abc: float (1..N) or NaN if never passed
      label: human-readable outcome
      total_attempts: int (cumulative attempts tried up to the latest snapshot)
    """
    ay1_attempts, ay1_pass, ay1_pass_idx = parse_period_outcome(ay1)
    ay2_attempts, ay2_pass, ay2_pass_idx = parse_period_outcome(ay2)

    # choose the latest cumulative snapshot
    # note: parse_period_outcome returns (0, False, None) for Never Enrolled/blank
    has_ay2_snapshot = ay2_attempts > 0 or ay2_pass
    if has_ay2_snapshot:
        attempts_so_far = ay2_attempts
        pass_idx = ay2_pass_idx
        passed = ay2_pass
    else:
        attempts_so_far = ay1_attempts
        pass_idx = ay1_pass_idx
        passed = ay1_pass

    # if the chosen snapshot already has a pass, that's the final answer
    if passed:
        total_attempts = pass_idx
        return float(pass_idx), _label_course_outcome(total_attempts, pass_idx), total_attempts

    # no pass in snapshots — check late free-text final_outcome
    if outcome_is_pass(final_outcome):
        pass_on = attempts_so_far + 1
        total_attempts = pass_on
        return float(pass_on), _label_course_outcome(total_attempts, pass_on), total_attempts

    # never passed; report attempts tried so far from the latest snapshot
    total_attempts = attempts_so_far
    if total_attempts == 0:
        return np.nan, "Never enrolled", 0
    return np.nan, _label_course_outcome(total_attempts, None), total_attempts



# ---------------------------
# Per-course features
# ---------------------------

def build_attempt_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ETM course mastery features per course, including human-readable outcomes
    that respect the "AY1 includes FA" rule.
    """
    n = len(df)
    feat = pd.DataFrame({"random_id": df.get("random_id", pd.Series(range(n)))})

    def series_or_na(colname: str | None, _n: int) -> pd.Series:
        if colname and colname in df.columns:
            return df[colname]
        return pd.Series([np.nan] * _n, index=df.index)

    for cname, mapping in COURSE_KEYS.items():
        # Raw columns
        fg = series_or_na(mapping.get("first_grade"), n)
        o_fa = series_or_na(mapping.get("outcome_1st_fa"), n)     # not used for counting (AY1 already includes FA)
        o_ay1 = series_or_na(mapping.get("outcome_1st_ay"), n)
        o_ay2 = series_or_na(mapping.get("outcome_2nd_ay"), n)
        o_fin = series_or_na(mapping.get("final_outcome"), n)

                # Compute attempts & outcome label using the AY model
        attempts_to_abc_list: List[float] = []
        label_text: List[str] = []
        total_attempts_list: List[int] = []

        for i in range(n):
            att_to_abc, lab, tot = compute_attempts_ay_model(
                o_ay1.iat[i],   # AY1 (includes FA)
                o_ay2.iat[i],   # AY2
                o_fin.iat[i],   # final free-text signal
            )
            attempts_to_abc_list.append(att_to_abc)
            label_text.append(lab)
            total_attempts_list.append(tot)

        s_attempts = pd.Series(attempts_to_abc_list, index=df.index)

        # Numeric features
        feat[f"{cname}_attempts_to_abc"] = s_attempts                 # NaN if never passed
        feat[f"{cname}_pass_by_first_attempt"] = s_attempts.eq(1.0).astype("Int64")
        feat[f"{cname}_ever_passed"] = s_attempts.notna().astype("Int64")
        feat[f"{cname}_first_grade_dfw"] = (
            fg.astype("string").str.upper().isin(FAIL_GRADES.union(WITHDRAW)).astype("Int64")
        )

        # NEW: attempts tried even if never passed (helps audit 'No pass after N attempts')
        feat[f"{cname}_total_attempts"] = pd.Series(total_attempts_list, index=df.index, dtype="Int64")

        # Human-readable outcome label
        feat[f"{cname}_outcome_label"] = pd.Series(label_text, index=df.index, dtype="string")



    # Rollups across the five ETM courses
    attempt_cols = [c for c in feat.columns if c.endswith("_attempts_to_abc")]
    dfw_cols = [c for c in feat.columns if c.endswith("_first_grade_dfw")]
    first_cols = [c for c in feat.columns if c.endswith("_pass_by_first_attempt")]
    ever_cols = [c for c in feat.columns if c.endswith("_ever_passed")]

    feat["etm_total_attempts_to_abc"] = feat[attempt_cols].sum(axis=1, skipna=True)
    feat["etm_first_attempt_pass_count"] = feat[first_cols].sum(axis=1)
    feat["etm_never_passed_count"] = (feat[ever_cols] == 0).sum(axis=1)
    feat["etm_first_grade_dfw_count"] = feat[dfw_cols].sum(axis=1)

    return feat


# ---------------------------
# Information-sheet features
# ---------------------------

def engineer_info_features(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Select & engineer info-sheet-only features with robust column fallbacks.
    """
    df = summary  # already standardized

    def resolve_col(df: pd.DataFrame, aliases: List[str], contains_all: tuple[str, ...] = ()) -> str | None:
        # exact alias first
        for a in aliases:
            if a in df.columns:
                return a
        # fuzzy fallback: choose first column containing ALL tokens
        for c in df.columns:
            if all(tok in c for tok in contains_all):
                return c
        return None

    # Resolve flexible columns (exact aliases + fuzzy tokens)
    col_enrolled = resolve_col(
        df,
        ["no_enrolled_terms_su_included", "enrolled_terms_su_included", "no_enrolled_terms", "enrolled_terms"],
        contains_all=("enroll", "term"),
    )
    col_warn = resolve_col(
        df,
        ["no_of_warning", "no_of_warnings", "warnings", "warning", "num_warnings", "n_warnings"],
        contains_all=("warn",),
    )
    col_lt25 = resolve_col(
        df,
        ["no_term_gpa_2_5", "term_gpa25", "no_term_gpa_2_50"],
        contains_all=("gpa", "2", "5"),
    )
    col_lt3 = resolve_col(
        df,
        ["no_term_gpa_3", "term_gpa3", "no_term_gpa_3_0"],
        contains_all=("gpa", "3"),
    )
    col_repeat3 = resolve_col(
        df,
        ["no_of_courses_taken_ge3rd_time", "courses_taken_ge3rd_time", "courses_taken_3rd_time"],
        contains_all=("3rd", "time"),
    )

    # Base keepers
    keep = [
        "random_id",
        "credit_window",
        "cgpa_at_etm_to_any_campus",
        "highest_cgpa_during_credit_window",
        "academic_suspenion_itwo",
        "me_bs_degree_status",
        "graduating_cgpa",
        "dif_btw_max_and_min_term_gpa",
        "total_grade_forgiveness_credits_approved_before_etm_to_any_campus",
        "sat_verb_grouping",
        "1st_aleks_math_score_grouping",
        "with_math_ap",
        "1st_math_course",
        "1st_math_course_campus",
    ]
    for opt in [col_enrolled, col_warn, col_lt25, col_lt3, col_repeat3]:
        if opt:
            keep.append(opt)

    base = df[[c for c in keep if c in df.columns]].copy()

    # Numeric coercions
    num_cols = [
        "cgpa_at_etm_to_any_campus",
        "highest_cgpa_during_credit_window",
        "graduating_cgpa",
        "dif_btw_max_and_min_term_gpa",
        "total_grade_forgiveness_credits_approved_before_etm_to_any_campus",
    ]
    for c in [col_enrolled, col_warn, col_lt25, col_lt3, col_repeat3]:
        if c:
            num_cols.append(c)
    to_numeric(base, [c for c in num_cols if c in base.columns])

    # Engineered features
    base["cgpa_gap"] = base.get("graduating_cgpa") - base.get("cgpa_at_etm_to_any_campus")

    # Graduated ME? (treat "GRAD..." or "DEGREE ..." as awarded; exclude negatives)
    status = base.get("me_bs_degree_status", pd.Series(index=base.index)).astype("string").str.upper().fillna("")
    positive = status.str.contains(r"\bGRAD", na=False) | status.str.contains(r"\bDEGREE\b", na=False)
    negative = status.str.contains(
    r"\b(?:NO|NOT|DENIED|PEND|PENDING|SEEK|SEEKING|IN\s*PROGRESS|PLANN|PLAN|INTENT|ADMIT|ENROLL)\b",
    regex=True, na=False
    )
    base["graduated_me"] = (positive & ~negative).astype("Int64")

    # Denominators & rates
    if col_enrolled and col_enrolled in base.columns:
        denom = base[col_enrolled].replace({0: np.nan})
    else:
        denom = pd.Series(np.nan, index=base.index)

    if col_warn and col_warn in base.columns:
        # treat missing warnings as 0 so students with no warnings get 0.0 instead of NaN
        num_warn = base[col_warn].fillna(0)
        base["warnings_per_term"] = num_warn / denom
    else:
        base["warnings_per_term"] = np.nan

    base["low_gpa_term_rate_2_5"] = (
        base[col_lt25] / denom if (col_lt25 and col_lt25 in base.columns) else np.nan
    )
    base["low_gpa_term_rate_3_0"] = (
        base[col_lt3] / denom if (col_lt3 and col_lt3 in base.columns) else np.nan
    )

    base["peak_minus_etm"] = base.get("highest_cgpa_during_credit_window") - base.get("cgpa_at_etm_to_any_campus")

    # Policy flags
    if "total_grade_forgiveness_credits_approved_before_etm_to_any_campus" in base.columns:
        base["grade_forgiveness_used"] = (
            base["total_grade_forgiveness_credits_approved_before_etm_to_any_campus"] > 0
        ).astype("Int64")
    else:
        base["grade_forgiveness_used"] = pd.Series(pd.NA, dtype="Int64", index=base.index)

    if col_repeat3 and col_repeat3 in base.columns:
        base["multi_repeat_flag"] = (base[col_repeat3] > 0).astype("Int64")
    else:
        base["multi_repeat_flag"] = pd.Series(pd.NA, dtype="Int64", index=base.index)

    # Missingness indicators
    for c in ["cgpa_at_etm_to_any_campus", "graduating_cgpa", col_enrolled]:
        if c and c in base.columns:
            base[f"{c}_is_missing"] = base[c].isna().astype("Int64")

    return base



# ---------------------------
# Public entry point
# ---------------------------

def prepare_information_features(summary_sheet: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline for the information sheet only + ETM course mastery features.
    Standardizes columns, builds engineered info features, then per-course mastery,
    and merges by random_id.
    """
    df = standardize_columns(summary_sheet)
    info = engineer_info_features(df)
    attempts = build_attempt_features(df)
    return info.merge(attempts, on="random_id", how="left")
