from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path
from rich import print

from .io import load_excel, read_sheet, save_csv
from .cleaning import standardize_columns
from .features import prepare_information_features

def find_sheet_by_cols(xls: pd.ExcelFile, must_have: list[str]) -> str | None:
    for s in xls.sheet_names:
        df0 = standardize_columns(read_sheet(xls, s).head(2))
        if all(any(m in c for c in df0.columns) for m in must_have):
            return s
    return None

def build_information_only(excel_path: str, out_csv: str) -> None:
    xls = load_excel(excel_path)
    # Heuristic sheet detection (tweak if needed)
    summary_sheet = find_sheet_by_cols(xls, ["random id","cgpa","graduating"]) or xls.sheet_names[0]
    df_summary = read_sheet(xls, summary_sheet)
    features = prepare_information_features(df_summary)
    save_csv(features, out_csv)
    print(f"[green]Saved:[/green] {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build information-sheet features")
    b.add_argument("--excel", required=True, help="Path to ETM Excel workbook")
    b.add_argument("--out", required=True, help="Path to write clean feature CSV")

    args = ap.parse_args()
    if args.cmd == "build":
        build_information_only(args.excel, args.out)

if __name__ == "__main__":
    main()
