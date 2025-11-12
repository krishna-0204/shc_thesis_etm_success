# Feature inventory (from current `clean_features.csv`)

## Raw summary/info columns (standardized)

* `random_id, credit_window, cgpa_at_etm_to_any_campus, highest_cgpa_during_credit_window, academic_suspenion__itwo, _warning, _courses_taken_ge3rd_time, total_grade_forgiveness_credits_approved_before_etm_to_any_campus, me_bs_degree_status, graduating_cgpa, enrolled_terms__su_included, dif_btw_max_and_min_term_gpa, _term_gpa25, _term_gpa3, sat_verb_grouping, 1st_aleks_math_score_grouping, with_math_ap, 1st_math_course, 1st_math_course_campus`
* Per-course raw outcomes/grades (as provided):
  `chem_110_outcome_after_1st_fa, …_after_1st_ay, …_after_2nd_ay, chem_110_outcome` (and same pattern for `edsgn_100, math_140, math_141, phys_211`)
  `chem_110__grade_code__1st_fa_or_sp, edsgn_100__grade_code__1st_fa_or_sp, math_140__grade_code__1st_fa_or_sp, math_141__grade_code__1st_fa_or_sp, phys_211__grade_code__1st_fa_or_sp`

## Engineered info-sheet features

* `cgpa_gap, graduated_me, warnings_per_term, low_gpa_term_rate_2_5, low_gpa_term_rate_3_0, peak_minus_etm, grade_forgiveness_used, multi_repeat_flag`
* Missingness indicators:
  `cgpa_at_etm_to_any_campus_is_missing, graduating_cgpa_is_missing, enrolled_terms__su_included_is_missing`

## Per-course mastery features (each of CHEM 110, EDSGN 100, MATH 140, MATH 141, PHYS 211)

* `<course>_attempts_to_abc, <course>_pass_by_first_attempt, <course>_ever_passed, <course>_first_grade_dfw, <course>_total_attempts, <course>_outcome`

## Cross-course rollups

* `etm_total_attempts_to_abc, etm_first_attempt_pass_count, etm_never_passed_count, etm_first_grade_dfw_count`

## Raw GPA grid from the GPA sheet

* `1st_term` (numeric code like 2168), plus term slots:
  `1st_fall, 1st_spring, 1st_summer, …, 8th_fall, 8th_spring, 8th_summer, 9th_fall`

## Engineered GPA-trajectory features

* `terms_with_gpa, mean_term_gpa, median_term_gpa, std_term_gpa, min_term_gpa, max_term_gpa, first_term_gpa, last_term_gpa, n_summer_terms, n_regular_terms, low_gpa_terms_2_5, low_gpa_terms_3_0, gpa_trend_slope, first_two_regular_mean_gpa, summer_term_ratio`

---

## (Optional) Column tidy-ups for readability

You’ve got a few artifacts from standardization (leading underscores, doubled underscores, a misspelling). If you want cleaner names in the CSV and README, apply this rename map right before `save_csv` in `cli.py`:

```python
CANONICAL_RENAMES = {
    "academic_suspenion__itwo": "academic_suspension_itwo",
    "_warning": "no_of_warnings",
    "_courses_taken_ge3rd_time": "no_of_courses_taken_ge3rd_time",
    "enrolled_terms__su_included": "no_enrolled_terms_su_included",
    "_term_gpa25": "no_term_gpa_2_5",
    "_term_gpa3": "no_term_gpa_3_0",
    # grade-code cols: collapse double underscores pattern-wise
    "chem_110__grade_code__1st_fa_or_sp": "chem_110_grade_code_1st_fa_or_sp",
    "edsgn_100__grade_code__1st_fa_or_sp": "edsgn_100_grade_code_1st_fa_or_sp",
    "math_140__grade_code__1st_fa_or_sp": "math_140_grade_code_1st_fa_or_sp",
    "math_141__grade_code__1st_fa_or_sp": "math_141_grade_code_1st_fa_or_sp",
    "phys_211__grade_code__1st_fa_or_sp": "phys_211_grade_code_1st_fa_or_sp",
}
master = master.rename(columns=CANONICAL_RENAMES)
```

Feature Descriptions

## A) Identifiers & baseline academics (raw, standardized)

* **`random_id`** — De-identified student identifier (string).
* **`credit_window`** — Credit window used to compute ETM metrics (e.g., first N credits) (int).
* **`cgpa_at_etm_to_any_campus`** — Cumulative GPA at the time of ETM (float).
* **`highest_cgpa_during_credit_window`** — Peak cumulative GPA observed within the credit window (float).
* **`academic_suspenion__itwo`** — Academic suspension flag/status from iTwo (categorical string).
* **`_warning`** — Count of academic warnings on record (int).
* **`_courses_taken_ge3rd_time`** — Count of courses attempted three or more times (int).
* **`total_grade_forgiveness_credits_approved_before_etm_to_any_campus`** — Total credits approved under grade forgiveness before ETM (float/int).
* **`me_bs_degree_status`** — Text status regarding Mechanical Engineering B.S. degree (string).
* **`graduating_cgpa`** — Cumulative GPA at graduation (float).
* **`enrolled_terms__su_included`** — Number of enrolled terms including Summer (int).
* **`dif_btw_max_and_min_term_gpa`** — Max term GPA minus min term GPA (term-to-term volatility span) (float).
* **`_term_gpa25`** — Number of terms with term GPA < 2.5 (int).
* **`_term_gpa3`** — Number of terms with term GPA < 3.0 (int).
* **`sat_verb_grouping`** — SAT verbal grouping/bucket (categorical).
* **`1st_aleks_math_score_grouping`** — First ALEKS math score bucket (categorical).
* **`with_math_ap`** — Indicator student had math AP credit (0/1 or Y/N; categorical).
* **`1st_math_course`** — Code of first math course attempted (e.g., MATH 140) (string).
* **`1st_math_course_campus`** — Campus where the first math course was taken (string).

## B) Per-course outcomes (raw snapshots as provided)

(For each of: `chem_110`, `edsgn_100`, `math_140`, `math_141`, `phys_211`)

* **`<course>_outcome_after_1st_fa`** — Status after the **first Fall** term only (e.g., “1st Attempt=ABC”, “Only 1 Attempt=DFW”, “Never Enrolled”) (string).
* **`<course>_outcome_after_1st_ay`** — Cumulative status across **AY1 (Fall + Spring + Summer)** (string).
* **`<course>_outcome_after_2nd_ay`** — Cumulative status across **AY1 + AY2** (string).
* **`<course>_outcome`** — Final/free-text course outcome captured later (string).
* **`<course>__grade_code__1st_fa_or_sp`** — **First recorded letter grade** in **first Fall or Spring** attempt (e.g., A, B+, C, D, F, W…) (string).

> Notes: AY1 already includes Fall; AY2 is cumulative over AY1 + second academic year.

## C) Engineered info-sheet features

* **`cgpa_gap`** — `graduating_cgpa − cgpa_at_etm_to_any_campus` (float).
* **`graduated_me`** — 1 if `me_bs_degree_status` indicates degree conferred (nullable Int64).
* **`warnings_per_term`** — `_warning ÷ enrolled_terms__su_included` (float; denom 0 → NaN).
* **`low_gpa_term_rate_2_5`** — `_term_gpa25 / enrolled_terms__su_included` (float).
* **`low_gpa_term_rate_3_0`** — `_term_gpa3 / enrolled_terms__su_included` (float).
* **`peak_minus_etm`** — `highest_cgpa_during_credit_window − cgpa_at_etm_to_any_campus` (float).
* **`grade_forgiveness_used`** — 1 if `total_grade_forgiveness_credits_approved_before_etm_to_any_campus > 0` (nullable Int64).
* **`multi_repeat_flag`** — 1 if `_courses_taken_ge3rd_time > 0` (nullable Int64).

### Missingness indicators (nullable Int64; 1 = missing)

* **`cgpa_at_etm_to_any_campus_is_missing`**
* **`graduating_cgpa_is_missing`**
* **`enrolled_terms__su_included_is_missing`**

## D) Per-course mastery features (engineered)

(For each of: `chem_110`, `edsgn_100`, `math_140`, `math_141`, `phys_211`)

* **`<course>_attempts_to_abc`** — Attempt number on which the course was first passed with grade A/B/C under the AY rules (float; NaN if never passed).
* **`<course>_pass_by_first_attempt`** — 1 if passed on first attempt (nullable Int64).
* **`<course>_ever_passed`** — 1 if ever achieved A/B/C (nullable Int64).
* **`<course>_first_grade_dfw`** — 1 if the **first** grade code was D/F/W (nullable Int64).
* **`<course>_total_attempts`** — Total attempts tried (even if never passed) inferred from AY snapshots (nullable Int64).
* **`<course>_outcome`** — Human label summarizing the path (e.g., “Pass on 2nd attempt”, “No pass after 2 attempts”, “Never enrolled”) (string).
* **`<course>_outcome_label`** -  (engineered) = the normalized, human-readable summary we compute from AY1/AY2 snapshots.

### Cross-course rollups (across the five ETM courses)

* **`etm_total_attempts_to_abc`** — Sum of `<course>_attempts_to_abc` (skips NaN) (float).
* **`etm_first_attempt_pass_count`** — Count of courses with `<course>_pass_by_first_attempt == 1` (int).
* **`etm_never_passed_count`** — Number of courses with `<course>_ever_passed == 0` (int).
* **`etm_first_grade_dfw_count`** — Number of courses with `<course>_first_grade_dfw == 1` (int).

## E) Term GPA grid (raw from GPA sheet)

* **`1st_term`** — PeopleSoft-style term code (e.g., `2168` = **Fall 2016**; last digit 1/5/8 → Spring/Summer/Fall) (int).
* **`1st_fall, 1st_spring, 1st_summer, …, 8th_fall, 8th_spring, 8th_summer, 9th_fall`** — Term GPAs for each slot (float; NaN if not enrolled that term).

## F) GPA-trajectory features (engineered from the grid)

* **`terms_with_gpa`** — Number of non-missing term GPAs (int).
* **`mean_term_gpa`** — Mean of available term GPAs (float).
* **`median_term_gpa`** — Median of available term GPAs (float).
* **`std_term_gpa`** — Standard deviation of term GPAs (float).
* **`min_term_gpa`**, **`max_term_gpa`** — Min/Max term GPA (float).
* **`first_term_gpa`** — GPA at earliest available slot for the student (float).
* **`last_term_gpa`** — GPA at latest available slot for the student (float).
* **`n_summer_terms`** — Count of Summer terms with GPA (int).
* **`n_regular_terms`** — Count of regular (Fall/Spring) terms with GPA (int).
* **`low_gpa_terms_2_5`**, **`low_gpa_terms_3_0`** — Counts of terms with GPA < 2.5 / < 3.0 (int).
* **`gpa_trend_slope`** — OLS slope of GPA vs. term index (↑ positive = improving trend; ↓ negative = declining) (float).
* **`first_two_regular_mean_gpa`** — Mean GPA across the first two regular (Fall/Spring) terms (float).
* **`summer_term_ratio`** — `n_summer_terms / terms_with_gpa` (float; NaN if denom = 0).

---

### dtype / NA conventions

* **Nullable Ints** use pandas `Int64` (can store 0/1 and `NaN` cleanly).
* Rates divide by `enrolled_terms__su_included`; if 0 or missing, the rate is `NaN`.
* `<course>_attempts_to_abc = NaN` means “never passed,” but `<course>_total_attempts` still records how many tries occurred.

If you want, I can also generate a compact markdown table (name • description • dtype) you can drop straight into the README.
