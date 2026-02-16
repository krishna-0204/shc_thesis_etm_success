# ETM Preprocessing (rebuild)

This rebuild:
1) reads the raw ETM Excel workbook
2) standardizes messy column names
3) deduplicates per-student rows (via merge_id)
4) keeps ALL raw columns and adds ONLY the engineered features (A–G)
5) filters to a target credit_window (default 40-59)
6) writes:
   - clean_processed_data.csv
   - clean_processed_data.xlsx
   - (if term-GPA grid exists) clean_features_terms_long.csv

## Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run
python -m etm_preprocessing.cli build \
  --excel /path/to/Populated_ETM_Study_Data_to_Researcher.xlsx \
  --out-dir data/processed \
  --credit-window 40-59

## Outputs
data/processed/clean_processed_data.csv
data/processed/clean_processed_data.xlsx
data/processed/clean_features_terms_long.csv (optional)
