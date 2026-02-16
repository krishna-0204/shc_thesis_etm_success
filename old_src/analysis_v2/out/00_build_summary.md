# analysis_v2 build summary

- input: `/Users/krishnapagrut/Developer/shc_thesis_etm_success/data/processed/clean_features.csv`
- rows (before dedupe): 1236
- rows (after dedupe by random_id): 1236

## Files written

- `/Users/krishnapagrut/Developer/shc_thesis_etm_success/src/analysis_v2/out/analysis_view_full.csv` (debug)
- `/Users/krishnapagrut/Developer/shc_thesis_etm_success/src/analysis_v2/out/analysis_view.csv` (tidy)
- `/Users/krishnapagrut/Developer/shc_thesis_etm_success/src/analysis_v2/out/analysis_view.parquet` (tidy parquet)

## Key columns sanity check (tidy view)

- credit_window_label (top 10):
  - 40-59: 821
  - 29-55: 415

- me_degree_status_bucket (top 10):
  - me_bs_only: 860
  - missing_or_blank: 280
  - me_bs_plus_other_bach: 61
  - other_bach_no_me: 35

- outcome_graduated_me (top 5):
  - 1: 921
  - 0: 315

- outcome_any_bachelor (top 5):
  - 1: 956
  - 0: 280

- outcome_status_known (top 5):
  - 1: 956
  - 0: 280

## Missingness (key columns)

- credit_window_label: 0.0000
- me_degree_status_bucket: 0.0000
- outcome_graduated_me: 0.0000
- outcome_any_bachelor: 0.0000
- outcome_status_known: 0.0000