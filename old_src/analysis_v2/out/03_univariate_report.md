# Univariate screens (known outcomes only)

- input: `/Users/krishnapagrut/Developer/shc_thesis_etm_success/src/analysis_v2/out/analysis_view.csv`
- rows total: 1236
- rows with known outcome_status_known==1: 956
- predictors screened: 72
- scipy available (chi-square p-values): False

## Notes (important)

- This report excludes degree-status-derived columns (leakage).
- Unknown outcomes (status_known==0) are excluded from outcome comparisons.

## Top numeric signals vs graduated_me (by |Cohen's d|)

| feature                           |   n |   missing_rate |      mean_y0 |    mean_y1 |    std_y0 |    std_y1 |   cohens_d |   pearson_r |   auc_rank |
|:----------------------------------|----:|---------------:|-------------:|-----------:|----------:|----------:|-----------:|------------:|-----------:|
| low_gpa_term_rate_3_0             | 956 |     0          |  0.179113    | 0.0879531  | 0.201298  | 0.138237  | -0.646661  |  -0.120684  |   0.374779 |
| low_gpa_term_rate_2_5             | 956 |     0          |  0.0630701   | 0.0231142  | 0.125467  | 0.0624476 | -0.607786  |  -0.113526  |   0.414007 |
| dif_btw_max_and_min_term_gpa      | 956 |     0          |  1.21943     | 0.887286   | 0.956104  | 0.565109  | -0.569163  |  -0.106396  |   0.398232 |
| graduating_cgpa                   | 943 |     0.0135983  |  3.45273     | 3.57252    | 0.303694  | 0.268764  |  0.444342  |   0.0669933 |   0.611761 |
| cgpa_gap                          | 943 |     0.0135983  | -0.0563636   | 0.0158849  | 0.224681  | 0.171665  |  0.417559  |   0.0629717 |   0.622199 |
| cgpa_at_etm_to_any_campus         | 955 |     0.00104603 |  3.48471     | 3.55663    | 0.294333  | 0.271598  |  0.264037  |   0.0489177 |   0.574663 |
| highest_cgpa_during_credit_window | 956 |     0          |  3.50657     | 3.57365    | 0.286325  | 0.262445  |  0.254722  |   0.0478333 |   0.57557  |
| peak_minus_etm                    | 955 |     0.00104603 |  0.0252941   | 0.0170141  | 0.061901  | 0.0481606 | -0.170016  |  -0.0315207 |   0.487035 |
| gpa_trend_slope                   | 956 |     0          |  0.000541446 | 0.00830739 | 0.0421342 | 0.0502338 |  0.155419  |   0.0292065 |   0.556104 |
| max_term_gpa                      | 956 |     0          |  3.95114     | 3.93313    | 0.0867305 | 0.129749  | -0.140241  |  -0.0263563 |   0.488584 |
| std_term_gpa                      | 956 |     0          |  0.384809    | 0.364358   | 0.222392  | 0.227272  | -0.0900509 |  -0.0169272 |   0.472654 |
| min_term_gpa                      | 956 |     0          |  2.73486     | 2.7996     | 0.768108  | 0.756951  |  0.0854836 |   0.0160689 |   0.519497 |
| first_two_regular_mean_gpa        | 956 |     0          |  3.54257     | 3.51568    | 0.241251  | 0.318652  | -0.0850449 |  -0.0159865 |   0.483589 |
| last_term_gpa                     | 956 |     0          |  3.65543     | 3.61907    | 0.307182  | 0.523876  | -0.0702363 |  -0.0132034 |   0.537723 |
| summer_term_ratio                 | 956 |     0          |  0.150609    | 0.144391   | 0.0831235 | 0.0948203 | -0.0658465 |  -0.0123783 |   0.493004 |

## Top categorical signals vs graduated_me (by Cramer's V, then lift)

| feature                            | level        |   n |   rate_y1 |   overall_rate_y1 |   lift_vs_overall |   cramers_v |   chi2 |   p_value |   n_levels_used |
|:-----------------------------------|:-------------|----:|----------:|------------------:|------------------:|------------:|-------:|----------:|----------------:|
| 1st_math_course_campus             | BK           |  20 |  0.85     |          0.963389 |          0.882302 |    0.139784 |    nan |       nan |               9 |
| 1st_math_course_campus             | HB           |  52 |  0.923077 |          0.963389 |          0.958156 |    0.139784 |    nan |       nan |               9 |
| 1st_math_course_campus             | ER           |  27 |  0.925926 |          0.963389 |          0.961113 |    0.139784 |    nan |       nan |               9 |
| 1st_math_course_campus             | other        |  84 |  0.928571 |          0.963389 |          0.963859 |    0.139784 |    nan |       nan |               9 |
| 1st_math_course_campus             | AB           |  31 |  0.935484 |          0.963389 |          0.971034 |    0.139784 |    nan |       nan |               9 |
| 1st_math_course_campus             | BW           |  16 |  0.9375   |          0.963389 |          0.973127 |    0.139784 |    nan |       nan |               9 |
| 1st_math_course_campus             | UP           | 677 |  0.977843 |          0.963389 |          1.015    |    0.139784 |    nan |       nan |               9 |
| 1st_math_course_campus             | LV           |  23 |  0.956522 |          0.963389 |          0.992872 |    0.139784 |    nan |       nan |               9 |
| 1st_math_course_campus             | AL           |  26 |  0.961538 |          0.963389 |          0.998079 |    0.139784 |    nan |       nan |               9 |
| n_regular_terms                    | 9            | 138 |  0.913043 |          0.963389 |          0.947741 |    0.139281 |    nan |       nan |               9 |
| n_regular_terms                    | 11           |  35 |  0.914286 |          0.963389 |          0.949031 |    0.139281 |    nan |       nan |               9 |
| n_regular_terms                    | 7            |  31 |  1        |          0.963389 |          1.038    |    0.139281 |    nan |       nan |               9 |
| n_regular_terms                    | 6            |  12 |  1        |          0.963389 |          1.038    |    0.139281 |    nan |       nan |               9 |
| n_regular_terms                    | other        |  12 |  1        |          0.963389 |          1.038    |    0.139281 |    nan |       nan |               9 |
| n_regular_terms                    | 4            |   7 |  1        |          0.963389 |          1.038    |    0.139281 |    nan |       nan |               9 |
| n_regular_terms                    | 12           |  14 |  0.928571 |          0.963389 |          0.963859 |    0.139281 |    nan |       nan |               9 |
| n_regular_terms                    | 10           | 172 |  0.988372 |          0.963389 |          1.02593  |    0.139281 |    nan |       nan |               9 |
| n_regular_terms                    | 8            | 535 |  0.968224 |          0.963389 |          1.00502  |    0.139281 |    nan |       nan |               9 |
| chem_110__grade_code__1st_fa_or_sp | C            |  64 |  0.90625  |          0.963389 |          0.940689 |    0.129406 |    nan |       nan |               9 |
| chem_110__grade_code__1st_fa_or_sp | C+           |  36 |  1        |          0.963389 |          1.038    |    0.129406 |    nan |       nan |               9 |
| chem_110__grade_code__1st_fa_or_sp | other        |  31 |  1        |          0.963389 |          1.038    |    0.129406 |    nan |       nan |               9 |
| chem_110__grade_code__1st_fa_or_sp | Not Enrolled | 186 |  0.935484 |          0.963389 |          0.971034 |    0.129406 |    nan |       nan |               9 |
| chem_110__grade_code__1st_fa_or_sp | B            | 198 |  0.984848 |          0.963389 |          1.02227  |    0.129406 |    nan |       nan |               9 |
| chem_110__grade_code__1st_fa_or_sp | B-           |  42 |  0.97619  |          0.963389 |          1.01329  |    0.129406 |    nan |       nan |               9 |
| chem_110__grade_code__1st_fa_or_sp | A            | 251 |  0.972112 |          0.963389 |          1.00905  |    0.129406 |    nan |       nan |               9 |

## Top numeric signals vs any_bachelor (by |Cohen's d|)

| feature                           |   n |   missing_rate |   mean_y0 |   mean_y1 |   std_y0 |    std_y1 |   cohens_d |   pearson_r |   auc_rank |
|:----------------------------------|----:|---------------:|----------:|----------:|---------:|----------:|-----------:|------------:|-----------:|
| cgpa_at_etm_to_any_campus         | 955 |     0.00104603 |       nan | 3.55407   |      nan | 0.2726    |        nan |         nan |        nan |
| highest_cgpa_during_credit_window | 956 |     0          |       nan | 3.57119   |      nan | 0.263497  |        nan |         nan |        nan |
| peak_minus_etm                    | 955 |     0.00104603 |       nan | 0.0173089 |      nan | 0.0486999 |        nan |         nan |        nan |
| graduating_cgpa                   | 943 |     0.0135983  |       nan | 3.56972   |      nan | 0.270057  |        nan |         nan |        nan |
| cgpa_gap                          | 943 |     0.0135983  |       nan | 0.0141994 |      nan | 0.173278  |        nan |         nan |        nan |
| dif_btw_max_and_min_term_gpa      | 956 |     0          |       nan | 0.899446  |      nan | 0.586588  |        nan |         nan |        nan |
| low_gpa_term_rate_2_5             | 956 |     0          |       nan | 0.024577  |      nan | 0.0661332 |        nan |         nan |        nan |
| low_gpa_term_rate_3_0             | 956 |     0          |       nan | 0.0912905 |      nan | 0.141934  |        nan |         nan |        nan |
| mean_term_gpa                     | 956 |     0          |       nan | 3.49833   |      nan | 0.316545  |        nan |         nan |        nan |
| median_term_gpa                   | 956 |     0          |       nan | 3.55903   |      nan | 0.310378  |        nan |         nan |        nan |
| std_term_gpa                      | 956 |     0          |       nan | 0.365107  |      nan | 0.227013  |        nan |         nan |        nan |
| min_term_gpa                      | 956 |     0          |       nan | 2.79723   |      nan | 0.757052  |        nan |         nan |        nan |
| max_term_gpa                      | 956 |     0          |       nan | 3.93379   |      nan | 0.128441  |        nan |         nan |        nan |
| first_term_gpa                    | 956 |     0          |       nan | 3.4965    |      nan | 0.403023  |        nan |         nan |        nan |
| last_term_gpa                     | 956 |     0          |       nan | 3.6204    |      nan | 0.517489  |        nan |         nan |        nan |

## Top categorical signals vs any_bachelor (by Cramer's V, then lift)

| feature                            | level               |   n |   rate_y1 |   overall_rate_y1 |   lift_vs_overall |   cramers_v |   chi2 |   p_value |   n_levels_used |
|:-----------------------------------|:--------------------|----:|----------:|------------------:|------------------:|------------:|-------:|----------:|----------------:|
| warnings_per_term                  | 0.0                 | 953 |         1 |                 1 |                 1 |           0 |    nan |       nan |               3 |
| edsgn_100_ever_passed              | 1                   | 937 |         1 |                 1 |                 1 |           0 |    nan |       nan |               2 |
| edsgn_100_attempts_to_abc          | 1.0                 | 932 |         1 |                 1 |                 1 |           0 |    nan |       nan |               4 |
| edsgn_100_outcome                  | Pass on 1st attempt | 932 |         1 |                 1 |                 1 |           0 |    nan |       nan |               5 |
| edsgn_100_pass_by_first_attempt    | 1                   | 932 |         1 |                 1 |                 1 |           0 |    nan |       nan |               2 |
| edsgn_100_total_attempts           | 1                   | 932 |         1 |                 1 |                 1 |           0 |    nan |       nan |               4 |
| multi_repeat_flag                  | 0                   | 919 |         1 |                 1 |                 1 |           0 |    nan |       nan |               2 |
| grade_forgiveness_used             | 0                   | 911 |         1 |                 1 |                 1 |           0 |    nan |       nan |               2 |
| chem_110_ever_passed               | 1                   | 851 |         1 |                 1 |                 1 |           0 |    nan |       nan |               2 |
| chem_110_attempts_to_abc           | 1.0                 | 836 |         1 |                 1 |                 1 |           0 |    nan |       nan |               3 |
| chem_110_outcome                   | Pass on 1st attempt | 836 |         1 |                 1 |                 1 |           0 |    nan |       nan |               4 |
| chem_110_pass_by_first_attempt     | 1                   | 836 |         1 |                 1 |                 1 |           0 |    nan |       nan |               2 |
| chem_110_total_attempts            | 1                   | 836 |         1 |                 1 |                 1 |           0 |    nan |       nan |               3 |
| phys_211_ever_passed               | 1                   | 813 |         1 |                 1 |                 1 |           0 |    nan |       nan |               2 |
| phys_211__grade_code__1st_fa_or_sp | Not Enrolled        | 801 |         1 |                 1 |                 1 |           0 |    nan |       nan |               9 |
| credit_window_label                | 40-59               | 794 |         1 |                 1 |                 1 |           0 |    nan |       nan |               2 |
| math_141_ever_passed               | 1                   | 790 |         1 |                 1 |                 1 |           0 |    nan |       nan |               2 |
| phys_211_attempts_to_abc           | 1.0                 | 787 |         1 |                 1 |                 1 |           0 |    nan |       nan |               3 |
| phys_211_outcome                   | Pass on 1st attempt | 787 |         1 |                 1 |                 1 |           0 |    nan |       nan |               4 |
| phys_211_pass_by_first_attempt     | 1                   | 787 |         1 |                 1 |                 1 |           0 |    nan |       nan |               2 |
| phys_211_total_attempts            | 1                   | 787 |         1 |                 1 |                 1 |           0 |    nan |       nan |               3 |
| math_141__grade_code__1st_fa_or_sp | Not Enrolled        | 779 |         1 |                 1 |                 1 |           0 |    nan |       nan |               9 |
| math_141_attempts_to_abc           | 1.0                 | 734 |         1 |                 1 |                 1 |           0 |    nan |       nan |               4 |
| math_141_outcome                   | Pass on 1st attempt | 734 |         1 |                 1 |                 1 |           0 |    nan |       nan |               6 |
| math_141_pass_by_first_attempt     | 1                   | 734 |         1 |                 1 |                 1 |           0 |    nan |       nan |               2 |

## Files written

- `/Users/krishnapagrut/Developer/shc_thesis_etm_success/src/analysis_v2/out/03_univariate_report.md`
- `/Users/krishnapagrut/Developer/shc_thesis_etm_success/src/analysis_v2/out/03_numeric_screen_graduated_me.csv`
- `/Users/krishnapagrut/Developer/shc_thesis_etm_success/src/analysis_v2/out/03_numeric_screen_any_bachelor.csv`
- `/Users/krishnapagrut/Developer/shc_thesis_etm_success/src/analysis_v2/out/03_categorical_screen_graduated_me.csv`
- `/Users/krishnapagrut/Developer/shc_thesis_etm_success/src/analysis_v2/out/03_categorical_screen_any_bachelor.csv`