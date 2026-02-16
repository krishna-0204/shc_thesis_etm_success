# Outcomes EDA (analysis_v2)

- input: `/Users/krishnapagrut/Developer/shc_thesis_etm_success/src/analysis_v2/out/analysis_view.csv`
- rows: 1236
- unique random_id: 1236

## Overall base rates

- graduated_me: 0.7451
- any_bachelor: 0.7735
- status_known: 0.7735

## Base rates by credit_window_label

| credit_window_label   |   n |   rate_graduated_me |   rate_any_bachelor |   rate_status_known |
|:----------------------|----:|--------------------:|--------------------:|--------------------:|
| 40-59                 | 821 |            0.934227 |            0.967113 |            0.967113 |
| 29-55                 | 415 |            0.371084 |            0.390361 |            0.390361 |

## Base rates by me_degree_status_bucket

| me_degree_status_bucket   |   n |   rate_graduated_me |   rate_any_bachelor |   rate_status_known |
|:--------------------------|----:|--------------------:|--------------------:|--------------------:|
| me_bs_only                | 860 |                   1 |                   1 |                   1 |
| missing_or_blank          | 280 |                   0 |                   0 |                   0 |
| me_bs_plus_other_bach     |  61 |                   1 |                   1 |                   1 |
| other_bach_no_me          |  35 |                   0 |                   1 |                   1 |

## Files written

- `/Users/krishnapagrut/Developer/shc_thesis_etm_success/src/analysis_v2/out/02_outcomes_eda.md`
- `/Users/krishnapagrut/Developer/shc_thesis_etm_success/src/analysis_v2/out/02_base_rates_overall.csv`
- `/Users/krishnapagrut/Developer/shc_thesis_etm_success/src/analysis_v2/out/02_base_rates_by_credit_window.csv`
- `/Users/krishnapagrut/Developer/shc_thesis_etm_success/src/analysis_v2/out/02_base_rates_by_bucket.csv`