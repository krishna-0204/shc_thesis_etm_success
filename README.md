# shc_thesis_etm_success
Krishna Pagrut's SHC thesis (ETM Success for MechE Students) Repository



# ETM — Preprocessing

This repository prepares clean, analysis-ready features from the SHC ETM dataset.  
It focuses on **reproducible preprocessing**: column normalization, typing, policy-aware course outcomes, and engineered features suitable for clustering or supervised models later.

---


## Setup

```bash
python -m venv shc_thesis_venv
source shc_thesis_venv/bin/activate  # Windows: shc_thesis_venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Place your workbook at: `data/raw/ETM_Study_Data_to_Researcher.xlsx`.

---

## Build the features

```bash
export PYTHONPATH=src
python -m etm_preprocessing.cli build \
  --excel data/raw/ETM_Study_Data_to_Researcher.xlsx \
  --out   data/processed/clean_features.csv
```

The output CSV contains **only information-sheet based features** (see glossary below). Term-GPA melt/trajectory features will be added as a separate subcommand.

---

## Pass policy assumptions

The ETM courses consider an **ABC pass** (i.e., **C- is not a pass**).

- Pass grades: `A, A-, B+, B, B-, C+, C`
- Fail grades: `D, F`
- Withdraw-like codes mapped to W: `W, WF, WN, LD`
- Any “Outcome …” field containing “PASS” or any of `A/B/C` (but **not** `C-`) is treated as pass.

> If your department treats `C-` as pass in any special case, this single rule can be switched centrally and the features will regenerate consistently.

---

## Feature glossary (columns emitted)

Below are **all columns the current pipeline writes** to `clean_features.csv`.

### A. Core identifiers & anchors (from information sheet)
- `random_id` — anonymized student ID. *(string)*
- `credit_window` — credit window label or range as given. *(string)*
- `cgpa_at_etm_to_any_campus` — CGPA at ETM milestone. *(float)*
- `highest_cgpa_during_credit_window` — peak CGPA within the window. *(float)*
- `me_bs_degree_status` — status label for ME BS degree. *(string)*
- `graduating_cgpa` — terminal CGPA. *(float)*

### B. Standing / policy / enrollment summaries (from information sheet)
- `academic_suspenion_itwo` — indicator of iTwo suspension. *(0/1 or Int64)*
- `no_of_warning` — count of academic warnings. *(int)*
- `no_of_courses_taken_ge3rd_time` — count of courses taken 3+ times. *(int)*
- `total_grade_forgiveness_credits_approved_before_etm_to_any_campus` — forgiveness credits pre-ETM. *(float)*
- `no_enrolled_terms_su_included` — enrolled term count (incl. SU). *(int)*
- `dif_btw_max_and_min_term_gpa` — term GPA range summary. *(float)*
- `no_term_gpa_2_5` — count of terms with GPA < 2.5. *(int)*
- `no_term_gpa_3` — count of terms with GPA < 3.0. *(int)*

### C. Preparation / placement (from information sheet)
- `sat_verb_grouping` — verbal SAT group (ordinal/categorical). *(string)*
- `1st_aleks_math_score_grouping` — first ALEKS group (ordinal). *(string)*
- `with_math_ap` — AP math credit present. *(0/1 or Int64)*
- `1st_math_course` — code for first math course (e.g., MATH 22/110/140). *(string)*
- `1st_math_course_campus` — campus for the first math course. *(string)*

### D. Engineered outcome/trajectory features (derived)
- `cgpa_gap` = `graduating_cgpa` − `cgpa_at_etm_to_any_campus`. *(float)*
- `graduated_me` = 1 if `me_bs_degree_status` contains “GRAD”. *(Int64)*
- `warnings_per_term` = `no_of_warning` ÷ `no_enrolled_terms_su_included`. *(float)*
- `low_gpa_term_rate_2_5` = `no_term_gpa_2_5` ÷ `no_enrolled_terms_su_included`. *(float)*
- `low_gpa_term_rate_3_0` = `no_term_gpa_3` ÷ `no_enrolled_terms_su_included`. *(float)*
- `peak_minus_etm` = `highest_cgpa_during_credit_window` − `cgpa_at_etm_to_any_campus`. *(float)*
- `grade_forgiveness_used` = 1 if forgiveness credits > 0. *(Int64)*
- `multi_repeat_flag` = 1 if any course count ≥ 3. *(Int64)*

### E. Missingness indicators (useful predictors)
- `cgpa_at_etm_to_any_campus_is_missing` *(Int64)*
- `graduating_cgpa_is_missing` *(Int64)*
- `no_enrolled_terms_su_included_is_missing` *(Int64)*

### F. ETM course mastery — per-course features

For each: **CHEM 110, EDSGN 100, MATH 140, MATH 141, PHYS 211**  
(the column name uses the keys `chem_110`, `edsgn_100`, `math_140`, `math_141`, `phys_211`)

- `{course}_attempts_to_abc` — minimal attempts (1..4) inferred from first grade and phased outcomes; NaN if never passed. *(float; values 1–4 or NaN)*
- `{course}_pass_by_first_attempt` — 1 if first grade was ABC. *(Int64)*
- `{course}_ever_passed` — 1 if any outcome shows ABC pass. *(Int64)*
- `{course}_first_grade_dfw` — 1 if first grade ∈ {D, F, W}. *(Int64)*

### G. ETM course mastery — rollups across all five courses
- `etm_total_attempts_to_abc` — sum of `{course}_attempts_to_abc` across courses (ignores NaNs). *(float)*
- `etm_first_attempt_pass_count` — number of courses passed on first attempt. *(int)*
- `etm_never_passed_count` — number of ETM courses never passed (based on `{course}_ever_passed == 0`). *(int)*
- `etm_first_grade_dfw_count` — number of courses with first grade D/F/W. *(int)*

> **Note on raw provenance**: This pipeline **uses** the per-course raw outcomes (`(Grade Code) (1st FA or SP)`, `Outcome after 1st AY`, `Outcome after 2nd AY`, `Outcome`) to compute the features above, but does **not** emit those raw columns in the final CSV by default. If you want to keep them for auditing/future research, we can add a `--emit-raw-etm` flag to pass those columns through with names like `raw_chem_110_grade_first` / `raw_chem_110_outcome_1st_ay`, etc.

---

## Data quality checks (initial)

- `cgpa_negative_gap`: flags if `graduating_cgpa` < `cgpa_at_etm`.  
- `enrolled_terms_zero_or_neg`: flags rows with nonpositive enrolled terms.

(We’ll expand checks when we integrate the term-GPA sheet, e.g., recomputing low-GPA counts from melted terms and comparing to provided summaries.)

---

## Complexity

- Column normalization & typing: **O(N · M)**  
- Feature engineering (info sheet): **O(N)**  
- Per-course attempt computation (5 courses): **O(5N) = O(N)**  
- Memory footprint is linear in the number of rows/columns.

---

## Roadmap (next additions)

1. **Term-GPA melt & trajectory features**  
   - Long format (`term_index`, `term_gpa`), early 4 FA/SP terms, volatility, slopes.  
2. **Raw provenance passthrough** for auditing (optional flag).  
3. **Config-driven pass policy** (`conf/pass_policy.yaml`).  
4. **Unit tests** for edge cases (ambiguous outcomes, C- handling, missing fields).  
5. **Cluster-ready matrix** (scaling/encoding) once features are finalized.

---

## Reproducibility & versioning

- The CLI captures a stable read → transform → write flow.  
- Put the exact workbook under `data/raw/` and tag releases when you change pass policy or feature definitions.

