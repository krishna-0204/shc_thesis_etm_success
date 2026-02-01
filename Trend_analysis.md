# Numeric effects (Cohen’s d)

* **low_gpa_term_rate_3_0 ↓**: Graduates spent a smaller share of semesters below 3.0 (fewer mediocre terms).
* **_term_gpa3 ↓**: Graduates had fewer sub-3.0 semesters in raw count.
* **low_gpa_term_rate_2_5 ↓**: Graduates spent a smaller share of semesters under 2.5 (fewer weak terms).
* **_term_gpa25 ↓**: Graduates had fewer sub-2.5 semesters in raw count.
* **dif_btw_max_and_min_term_gpa ↓**: Graduates’ term GPAs swung less (steadier performance).
* **cgpa_at_etm_to_any_campus ↑**: Graduates entered ETM with higher cumulative GPAs.
* **peak_minus_etm ↓**: Graduates’ peak CGPA was closer to their ETM CGPA (less drop before ETM).
* **grade-forgiveness credits ↓**: Graduates used fewer grade-forgiveness credits before ETM.
* **grade_forgiveness_used ↓**: Graduates were less likely to use any grade forgiveness.
* **highest_cgpa_during_credit_window ↑**: Graduates hit higher peak CGPAs early on.
* **_courses_taken_ge3rd_time ↓**: Graduates rarely pushed courses to a 3rd+ attempt.
* **cgpa_gap ↑**: Graduates improved more from ETM CGPA to graduating CGPA.
* **multi_repeat_flag ↓**: Graduates were less likely to have multiple repeats on record.
* **warnings_per_term ↓**: Graduates accumulated fewer academic warnings per term.
* **chem_110_attempts_to_abc ↓**: Graduates needed fewer tries to first pass CHEM 110.
* **phys_211_attempts_to_abc ↓**: Fewer tries to first pass PHYS 211 among graduates.
* **etm_total_attempts_to_abc ↓**: Across ETM courses, graduates reached first passes in fewer total attempts.
* **math_141_attempts_to_abc ↓**: Fewer tries to first pass MATH 141 among graduates.
* **chem_110_total_attempts ↓**: Graduates attempted CHEM 110 fewer times overall.
* **phys_211_total_attempts ↓**: Fewer total PHYS 211 attempts among graduates.
* **enrolled_terms__su_included ↑**: Graduates persisted across more total terms.
* **math_141_total_attempts ↓**: Fewer total MATH 141 attempts among graduates.
* **math_140_attempts_to_abc ↓**: Fewer tries to first pass MATH 140 among graduates.
* **summer_term_ratio ↓**: Graduates took a smaller fraction of summer terms.
* **math_140_total_attempts ↓**: Fewer total MATH 140 attempts among graduates.
* **2nd_fall ↑**: Second-fall GPAs trend a bit higher among graduates.
* **n_regular_terms ↑**: Graduates completed more fall/spring terms.
* **n_summer_terms ↓**: Graduates completed fewer summer terms.
* **etm_never_passed_count ↑**: Slight uptick in “never passed” count among graduates (tiny; likely a labeling quirk—flag for review).
* **phys_211_ever_passed ↓**: Slightly less often flagged “ever passed PHYS 211” among graduates (very small; likely coding artifact).
* **chem_110_ever_passed ↓**: Same oddity for CHEM 110 (very small; likely coding artifact).
* **1st_fall ↓**: First-fall GPA is marginally lower among graduates (tiny; cohort/timing effect possible).
* **edsgn_100_total_attempts ↑**: Slightly more total EDSGN 100 attempts among graduates (tiny; mixed with other positive EDSGN signals).
* **math_140_ever_passed ↓**: Slightly less often flagged “ever passed MATH 140” among graduates (tiny; check coding).
* **gpa_trend_slope ↑**: Graduates’ GPAs improved more over time (positive slope).
* **math_141_ever_passed ↓**: Same tiny “ever passed” oddity for MATH 141 (coding check).
* **edsgn_100_ever_passed ↑**: Graduates more often ultimately passed EDSGN 100.
* **chem_110_pass_by_first_attempt ↑**: First-try pass in CHEM 110 is more common among graduates.
* **first_two_regular_mean_gpa ↓**: Very small lower early mean among graduates (counter-intuitive; watch binning/censoring).
* **1st_spring ↓**: First-spring GPA a hair lower among graduates (tiny; cohort effect possible).
* **std_term_gpa ↓**: Graduates’ term GPAs were less volatile.
* **edsgn_100_attempts_to_abc ↑**: Slightly more tries before first A/B/C in EDSGN 100 (tiny; mixed with stronger “pass on first try” positives elsewhere).
* **low_gpa_terms_2_5 ↑**: Slightly more raw <2.5 terms (tiny; denominator effects conflict with the rate version).
* **etm_first_attempt_pass_count ↑**: Graduates passed more ETM courses on the first try (aggregate).
* **edsgn_100_pass_by_first_attempt ↑**: First-try pass in EDSGN 100 is more common among graduates.
* **min_term_gpa ↑**: Graduates’ worst term was a bit better.
* **phys_211_pass_by_first_attempt ↑**: First-try pass in PHYS 211 is more common among graduates.
* **4th_fall ↑**: Fourth-fall GPAs trend slightly higher among graduates.
* **first_term_gpa ↓**: Very first term GPA is marginally lower among graduates (tiny; timing/cohort).
* **math_141_pass_by_first_attempt ↑**: First-try pass in MATH 141 is more common among graduates.
* **low_gpa_terms_3_0 ↓**: Slightly fewer raw sub-3.0 terms among graduates.
* **last_term_gpa ↑**: Graduates finish with slightly higher last-term GPAs.
* **4th_spring ↑**: Fourth-spring GPAs trend slightly higher among graduates.
* **3rd_fall ↑**: Third-fall GPAs trend slightly higher among graduates.
* **median_term_gpa ↓**: Median term GPA a touch lower among graduates (tiny; noise).
* **1st_term ↓**: First term code is a bit earlier among graduates (cohort timing).
* **terms_with_gpa ↑**: More terms with recorded GPAs for graduates (more continuous enrollment).
* **mean_term_gpa ↑**: Average term GPA is slightly higher among graduates.
* **math_140_pass_by_first_attempt ↓**: Slightly less often flagged first-try pass in MATH 140 among graduates (tiny; coding check).
* **3rd_spring ↑**: Third-spring GPAs trend slightly higher among graduates.

# Categorical effects (Risk Ratios)

* **credit_window 40–59 (↑)**: Computing ETM metrics over 40–59 credits aligns with higher graduation odds.
* **credit_window 29–55 (↓)**: This overlapping band aligns with lower odds (bins should be disjointed later).
* **academic_suspension = No (↑)**: Not being suspended is strongly tied to graduating (rare “Yes” → wide CI).
* **MATH 140 early grade = D (↓)**: Early D in MATH 140 hurts graduation odds.
* **CHEM 110 early grade = D (↓)**: Early D in CHEM 110 hurts odds.
* **ALEKS 30–45 (↓)**: Low ALEKS placement corresponds to lower odds.
* **1st math campus = AB (↓)**: Starting math at AB campus aligns with lower odds vs peers.
* **With math AP = No (↓)**: Lacking math AP credit corresponds to lower odds.
* **With math AP = Yes (↑)**: Having math AP credit corresponds to higher odds.
* **1st math campus = BK (↓)**: Starting math at BK campus aligns with lower odds.
* **1st math campus = UP (↑)**: Starting math at University Park aligns with higher odds.
* **MATH 140 early grade = A (↑)**: Early A in MATH 140 is favorable.
* **MATH 140 early grade = C (↓)**: Early C in MATH 140 is unfavorable.
* **CHEM 110 early grade = A (↑)**: Early A in CHEM 110 is favorable.
* **MATH 140 early grade = SAT (↓)**: “SAT” (placement/transfer) here tracks to lower odds in this data.
* **CHEM 110 early grade = LD (↓)**: Late Drop in CHEM 110 tracks to lower odds.
* **1st math course = MATH 26 (↓)**: Starting in MATH 26 aligns with lower odds.
* **CHEM 110 early grade = B (↑)**: Early B in CHEM 110 is favorable.
* **ALEKS 61–75 (↓)**: Mid ALEKS tier is slightly unfavorable vs peers.
* **PHYS 211 early grade = C+ (↓)**: Early C+ in PHYS 211 is unfavorable.
* **CHEM 110 early grade = B− (↓)**: Early B− in CHEM 110 is somewhat unfavorable.
* **1st math course = MATH 22 (↓)**: Starting in MATH 22 aligns with lower odds.
* **CHEM 110 early grade = C (↓)**: Early C in CHEM 110 is unfavorable.
* **ALEKS 46–60 (↓)**: Lower-mid ALEKS tier is unfavorable.
* **First term = Summer 2017 (↓)**: This start cohort tracks to lower odds (cohort effect).
* **EDSGN 100 early grade = B− (↓)**: Early B− in EDSGN 100 leans unfavorable (small n).
* **1st math course = MATH 141H (↑)**: Starting in honors calculus aligns with higher odds.
* **First term = Spring 2021 (↑)**: This start cohort tracks to higher odds (cohort effect).
* **ALEKS 76–82 (↑)**: Upper-mid ALEKS tier is favorable.
* **CHEM 110 early = Not Enrolled (↓)**: Skipping CHEM 110 in that first window leans unfavorable.
* **MATH 141 early grade = B− (≈↓)**: B− leans lower odds (borderline).
* **1st math campus = BW (≈↓)**: BW start leans lower odds (borderline).
* **PHYS 211 early grade = B+ (↑)**: Early B+ in PHYS 211 is favorable.
* **1st math course = MATH 230 (↑)**: Starting in vector calc aligns with higher odds (small n).
* **EDSGN 100 early grade = B (≈↓)**: B trends lower (borderline; small n).
* **ALEKS 90–100 (↑)**: Top ALEKS tier is favorable (slightly).
* **MATH 141 early grade = B (↑)**: Early B in MATH 141 is favorable.
* **1st math campus = LV (↑)**: Starting math at LV aligns with higher odds.
* **MATH 140 early grade = LD (↑)**: Late Drop in MATH 140 shows slightly higher odds here (idiosyncratic; interpret cautiously).
* **PHYS 211 early grade = A (↑)**: Early A in PHYS 211 is favorable.
* **ALEKS 83–89 (↑)**: High ALEKS tier is favorable.
* **PHYS 211 early grade = A− (↑)**: Early A− in PHYS 211 is favorable.
* **1st math campus = NK (↑)**: NK start aligns with higher odds (small n).
* **CHEM 110 early grade = B+ (↑)**: Early B+ in CHEM 110 is slightly favorable.
* **1st math campus = WB (↑)**: WB start aligns with higher odds (small n).
* **PHYS 211 early grade = B− (≈↓)**: B− leans lower (borderline; small n).
* **1st math campus = HB (≈↓)**: HB start leans lower (borderline).
* **CHEM 110 early grade = C+ (≈↓)**: C+ leans lower (borderline).
* **1st math course = MATH 231 (≈↓)**: Starting in MATH 231 leans lower (borderline; small n).
* **1st math campus = HN (↑)**: HN start aligns with higher odds (small n).
* **CHEM 110 early grade = SAT (≈↓)**: SAT code leans lower (borderline).
* **PHYS 211 early = Not Enrolled (≈↓)**: Not enrolled leans lower odds (borderline).
* **MATH 141 early grade = LD (↑)**: LD in MATH 141 shows slightly higher odds (borderline; unusual—treat cautiously).
* **1st math course = MATH 140B (≈↓)**: 140B start leans lower (small n).
* **CHEM 110 early grade = A− (≈=)**: A− looks neutral to slightly lower (borderline).
* **1st math course = MATH 251 (≈↑)**: 251 start looks slightly favorable.
* **EDSGN 100 early grade = B+ (≈↑)**: B+ is mildly favorable (borderline).
* **First term = Summer 2021 (≈↓)**: This cohort leans lower (borderline).
* **MATH 140 early grade = A− (≈↑)**: A− is mildly favorable.
* **MATH 140 early grade = B (≈↓)**: B trends slightly lower.

# Bucketed separations (grad-rate spread across bins)

* **CHEM 110 total attempts**: More attempts → clearly worse graduation rates; strong separator.
* **Share of <3.0 terms**: Higher share of sub-3.0 terms → worse outcomes; strong separator.
* **PHYS 211 total attempts**: More attempts → worse outcomes; strong separator.
* **Count of <3.0 terms**: More sub-3.0 terms → worse outcomes; strong separator.
* **ETM CGPA**: Higher ETM CGPA → higher grad rates; robust separation.
* **Peak CGPA in window**: Higher peak early CGPA → higher grad rates.
* **Term-GPA swing (max–min)**: Larger swings → worse grad rates; stability helps.
* **ETM total attempts to first passes**: More total attempts to first A/B/C → worse outcomes.
* **Peak−ETM gap**: Bigger drop from peak to ETM → worse outcomes.
* **MATH 141 total attempts**: More attempts → worse outcomes (milder but present).
* **MATH 140 total attempts**: More attempts → worse outcomes (milder but present).
* **EDSGN 100 total attempts**: Slightly higher attempts relate to slightly better rates here (weak, but separates).
* **Total enrolled terms**: Moderate separation; more enrollment time modestly helps.
* **Std dev of term GPA**: More volatility → slightly worse outcomes.
* **First two regular terms mean GPA**: Higher early mean → modestly better outcomes.
* **2nd-spring GPA**: Higher 2nd-spring → modestly better outcomes.
* **1st-spring GPA**: Higher 1st-spring → modestly better outcomes.
* **First-term GPA**: Higher first-term → modestly better outcomes.
* **1st-fall GPA**: Higher first fall → modestly better outcomes.
* **Terms with GPA**: More recorded terms → slightly better outcomes (persistence).
* **Summer term ratio**: More summers → slightly worse outcomes.
* **Minimum term GPA**: Higher floor → slightly better outcomes.
* **CGPA gap (grad − ETM)**: Bigger improvement → slightly better outcomes (nearly ceiling).
* **2nd-fall GPA**: Higher 2nd-fall → modestly better outcomes.
* **4th-fall GPA**: Higher 4th-fall → modestly better outcomes.
* **ETM never-passed count**: More “never passed” → slightly worse rates (small separation).
* **4th-spring GPA**: Higher 4th-spring → modestly better outcomes.
* **3rd-spring GPA**: Higher 3rd-spring → modestly better outcomes.
* **# summer terms**: More summers → slightly worse outcomes.
* **3rd-fall GPA**: Higher 3rd-fall → modestly better outcomes.
* **# regular terms**: More fall/spring terms → slightly better outcomes.
* **1st term code**: Later/earlier cohorts show small differences (cohort timing).
* **Last-term GPA**: Higher finishing GPA → slightly better outcomes.
* **ETM first-attempt pass count**: Passing more ETM courses on first try → slightly better outcomes.
* **Mean term GPA**: Higher mean → slightly better outcomes.
* **GPA trend slope**: Upward trend → slightly better outcomes.
* **Count of <3.0 terms**: More sub-3.0 terms → slightly worse outcomes (small separation).
* **Median term GPA**: Higher median → tiny improvement.
* **Max term GPA**: Max alone barely separates outcomes.
* **# courses taken ≥3rd time**: This binning showed no real separation (flat).

# Early GPA ladder

* **First two regular terms**: Early GPAs do differentiate, but the ladder is not strictly monotone here (likely cohort/noise); broadly, stronger early terms help.

