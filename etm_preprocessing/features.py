from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .cleaning import standardize_columns, to_numeric

PASS_GRADES = {"A", "A-", "B+", "B", "B-", "C+", "C"}
FAIL_GRADES = {"D", "F"}
WITHDRAW = {"W", "WF", "WN", "LD"}

COURSE_KEYS: Dict[str, Dict[str, str]] = {
    "chem_110": {
        "first_grade": "chem_110_grade_code_1st_fa_or_sp",
        "outcome_1st_ay": "chem_110_outcome_after_1st_ay",
        "outcome_2nd_ay": "chem_110_outcome_after_2nd_ay",
        "final_outcome": "chem_110_outcome",
    },
    "edsgn_100": {
        "first_grade": "edsgn_100_grade_code_1st_fa_or_sp",
        "outcome_1st_ay": "edsgn_100_outcome_after_1st_ay",
        "outcome_2nd_ay": "edsgn_100_outcome_after_2nd_ay",
        "final_outcome": "edsgn_100_outcome",
    },
    "math_140": {
        "first_grade": "math_140_grade_code_1st_fa_or_sp",
        "outcome_1st_ay": "math_140_outcome_after_1st_ay",
        "outcome_2nd_ay": "math_140_outcome_after_2nd_ay",
        "final_outcome": "math_140_outcome",
    },
    "math_141": {
        "first_grade": "math_141_grade_code_1st_fa_or_sp",
        "outcome_1st_ay": "math_141_outcome_after_1st_ay",
        "outcome_2nd_ay": "math_141_outcome_after_2nd_ay",
        "final_outcome": "math_141_outcome",
    },
    "phys_211": {
        "first_grade": "phys_211_grade_code_1st_fa_or_sp",
        "outcome_1st_ay": "phys_211_outcome_after_1st_ay",
        "outcome_2nd_ay": "phys_211_outcome_after_2nd_ay",
        "final_outcome": "phys_211_outcome",
    },
}

ATT_PATTERNS: Dict[str, re.Pattern] = {
    "never": re.compile(r"^\s*Never\s+Enrolled\s*$", re.I),
    "first_pass": re.compile(r"^\s*1(?:st)?\s*Attempt\s*=\s*ABC\s*$", re.I),
    "only1_dfw": re.compile(r"^\s*Only\s+1\s*Attempt\s*=\s*DFW\s*$", re.I),
    "two_dfw_then_abc": re.compile(r"^\s*2\s*Attempts\s*\(\s*1st\s*=\s*DFW\s*,\s*2nd\s*=\s*ABC\s*\)\s*$", re.I),
    "two_both_dfw": re.compile(r"^\s*2\s*Attempts\s*\(\s*Both\s*=\s*DFW\s*\)\s*$", re.I),
}

TERM_DIGIT_TO_NAME = {1: "Spring", 5: "Summer", 8: "Fall"}
TERM_DIGIT_TO_MONTH = {1: 1, 5: 6, 8: 8}


def decode_psu_term(term_code: int | float | str):
    if pd.isna(term_code):
        return None, None, None
    n = int(term_code)
    sem_digit = n % 10
    base = n // 10
    year = 2000 + (base - 200)
    sem_name = TERM_DIGIT_TO_NAME.get(sem_digit, f"Term{sem_digit}")
    month = TERM_DIGIT_TO_MONTH.get(sem_digit, 1)
    ts = pd.Timestamp(year=year, month=month, day=1)
    return f"{sem_name} {year}", year, ts


def outcome_is_pass(x: object) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    xs = str(x).strip().upper()
    # Treat explicit ABC/PASS as pass; exclude C-
    if "C-" in xs:
        return False
    return ("ABC" in xs) or ("PASS" in xs)


def parse_period_outcome(s: object) -> Tuple[int, bool, int | None]:
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

    m_pass = re.search(r"(\d+)\s*Attempts?.*ABC", txt, re.I)
    if m_pass:
        k = int(m_pass.group(1))
        return k, True, k
    m_any = re.search(r"(\d+)\s*Attempts?", txt, re.I)
    if m_any:
        return int(m_any.group(1)), False, None
    return 0, False, None


def compute_attempts_ay_model(ay1: object, ay2: object, final_outcome: object) -> float:
    """
    Returns attempts_to_abc (1..N) or NaN if never passed.
    AY rule: AY1 already includes FA.
    """
    ay1_attempts, ay1_pass, ay1_pass_idx = parse_period_outcome(ay1)
    ay2_attempts, ay2_pass, ay2_pass_idx = parse_period_outcome(ay2)

    has_ay2_snapshot = ay2_attempts > 0 or ay2_pass
    if has_ay2_snapshot:
        attempts_so_far = ay2_attempts
        pass_idx = ay2_pass_idx
        passed = ay2_pass
    else:
        attempts_so_far = ay1_attempts
        pass_idx = ay1_pass_idx
        passed = ay1_pass

    if passed:
        return float(pass_idx)

    if outcome_is_pass(final_outcome):
        return float(attempts_so_far + 1)

    return np.nan


def build_attempt_features(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    feat = pd.DataFrame({"random_id": df.get("random_id", pd.Series(range(n), index=df.index))})

    def series_or_na(colname: str | None, _n: int) -> pd.Series:
        if colname and colname in df.columns:
            return df[colname]
        return pd.Series([np.nan] * _n, index=df.index)

    for cname, mapping in COURSE_KEYS.items():
        fg = series_or_na(mapping.get("first_grade"), n)
        o_ay1 = series_or_na(mapping.get("outcome_1st_ay"), n)
        o_ay2 = series_or_na(mapping.get("outcome_2nd_ay"), n)
        o_fin = series_or_na(mapping.get("final_outcome"), n)

        attempts_to_abc = [
            compute_attempts_ay_model(o_ay1.iat[i], o_ay2.iat[i], o_fin.iat[i]) for i in range(n)
        ]
        s_attempts = pd.Series(attempts_to_abc, index=df.index)

        feat[f"{cname}_attempts_to_abc"] = s_attempts
        feat[f"{cname}_pass_by_first_attempt"] = s_attempts.eq(1.0).astype("Int64")
        feat[f"{cname}_ever_passed"] = s_attempts.notna().astype("Int64")
        feat[f"{cname}_first_grade_dfw"] = (
            fg.astype("string").str.upper().isin(FAIL_GRADES.union(WITHDRAW)).astype("Int64")
        )

    attempt_cols = [c for c in feat.columns if c.endswith("_attempts_to_abc")]
    dfw_cols = [c for c in feat.columns if c.endswith("_first_grade_dfw")]
    first_cols = [c for c in feat.columns if c.endswith("_pass_by_first_attempt")]
    ever_cols = [c for c in feat.columns if c.endswith("_ever_passed")]

    feat["etm_total_attempts_to_abc"] = feat[attempt_cols].sum(axis=1, skipna=True)
    feat["etm_first_attempt_pass_count"] = feat[first_cols].sum(axis=1)
    feat["etm_never_passed_count"] = (feat[ever_cols] == 0).sum(axis=1)
    feat["etm_first_grade_dfw_count"] = feat[dfw_cols].sum(axis=1)

    return feat


def engineer_info_features(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary

    def resolve_col(df_: pd.DataFrame, aliases: List[str], contains_all: tuple[str, ...] = ()) -> str | None:
        for a in aliases:
            if a in df_.columns:
                return a
        for c in df_.columns:
            if all(tok in c for tok in contains_all):
                return c
        return None

    # --- resolve potentially-renamed columns ---
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

    # --- pass-through columns (only if present) ---
    keep = [
        "random_id",
        "credit_window",
        "cgpa_at_etm_to_any_campus",
        "highest_cgpa_during_credit_window",
        "me_bs_degree_status",
        "graduating_cgpa",
        "academic_suspenion_itwo",
        "no_of_warning",
        "no_of_courses_taken_ge3rd_time",
        "total_grade_forgiveness_credits_approved_before_etm_to_any_campus",
        "no_enrolled_terms_su_included",
        "dif_btw_max_and_min_term_gpa",
        "no_term_gpa_2_5",
        "no_term_gpa_3",
        "sat_verb_grouping",
        "1st_aleks_math_score_grouping",
        "with_math_ap",
        "1st_math_course",
        "1st_math_course_campus",
    ]
    for opt in [col_enrolled, col_warn, col_lt25, col_lt3, col_repeat3]:
        if opt and opt not in keep:
            keep.append(opt)

    base = df[[c for c in keep if c in df.columns]].copy()

    # --- numeric cleanup ---
    num_cols = [
        "cgpa_at_etm_to_any_campus",
        "highest_cgpa_during_credit_window",
        "graduating_cgpa",
        "dif_btw_max_and_min_term_gpa",
        "total_grade_forgiveness_credits_approved_before_etm_to_any_campus",
        col_enrolled,
        col_warn,
        col_lt25,
        col_lt3,
        col_repeat3,
    ]
    to_numeric(base, [c for c in num_cols if c and c in base.columns])

    # --- derived GPA deltas ---
    base["cgpa_gap"] = base.get("graduating_cgpa") - base.get("cgpa_at_etm_to_any_campus")
    base["peak_minus_etm"] = base.get("highest_cgpa_during_credit_window") - base.get("cgpa_at_etm_to_any_campus")

    # --- ME degree status bucket + outcomes ---
    status_raw = base.get("me_bs_degree_status", pd.Series(index=base.index)).astype("string")
    status = (
        status_raw.fillna("")
        .str.replace("\u00A0", " ", regex=False)
        .str.strip()
        .str.upper()
    )

    def bucketize_me_status(s: pd.Series) -> pd.Series:
        out = pd.Series(pd.NA, index=s.index, dtype="string")

        is_blank = s.eq("") | s.isna()
        out.loc[is_blank] = "missing_or_blank"

        # ME degree earned (single or double major)
        out.loc[s.str.contains(r"\bME_BS\b.*\bDEGREE\b.*\bONLY\b", regex=True, na=False)] = "me_bs_only"
        out.loc[s.str.contains(r"\bME_BS\b.*\bWITH\b.*\bANOTHER\b", regex=True, na=False)] = "me_bs_plus_other_bach"

        # No ME degree but has another bach degree
        out.loc[s.str.contains(r"\bNO\b.*\bME_BS\b.*\bDEGREE\b.*\bOTHER\b", regex=True, na=False)] = "other_bach_no_me"

        return out.fillna("other_unknown")

    bucket = bucketize_me_status(status)
    base["me_degree_status_bucket"] = bucket

    base["graduated_me"] = bucket.isin(["me_bs_only", "me_bs_plus_other_bach"]).astype("Int64")
    base["not_graduate_me"] = bucket.eq("missing_or_blank").astype("Int64")
    base["transferred_out_of_me"] = bucket.eq("other_bach_no_me").astype("Int64")

    # --- rate-style features (divide by enrolled terms) ---
    denom = (
        base.get(col_enrolled, pd.Series(np.nan, index=base.index)).replace({0: np.nan})
        if col_enrolled
        else pd.Series(np.nan, index=base.index)
    )

    if col_warn and col_warn in base.columns:
        base["warnings_per_term"] = base[col_warn].fillna(0) / denom
    else:
        base["warnings_per_term"] = np.nan

    base["low_gpa_term_rate_2_5"] = (
        base[col_lt25] / denom if (col_lt25 and col_lt25 in base.columns) else np.nan
    )
    base["low_gpa_term_rate_3_0"] = (
        base[col_lt3] / denom if (col_lt3 and col_lt3 in base.columns) else np.nan
    )

    # --- binary flags ---
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

    # --- missingness indicators ---
    for c in ["cgpa_at_etm_to_any_campus", "graduating_cgpa", col_enrolled]:
        if c and c in base.columns:
            base[f"{c}_is_missing"] = base[c].isna().astype("Int64")

    return base


def prepare_information_features(summary_sheet: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(summary_sheet)
    info = engineer_info_features(df)
    attempts = build_attempt_features(df)
    return info.merge(attempts, on="random_id", how="left")
