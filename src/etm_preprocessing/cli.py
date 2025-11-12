from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd
from rich import print

from .io import load_excel, read_sheet, save_csv
from .cleaning import standardize_columns
from .features import prepare_information_features, decode_psu_term  # engineered info + per-course features
from .gpa_features import build_term_gpa_features   # engineered GPA features + long table


def _normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def find_sheet_by_cols(xls: pd.ExcelFile, must_have: list[str]) -> str | None:
    for s in xls.sheet_names:
        df0 = standardize_columns(read_sheet(xls, s).head(2))
        if all(any(m in c for c in df0.columns) for m in must_have):
            return s
    return None

# def find_gpa_sheet(xls: pd.ExcelFile) -> str | None:
#     for s in xls.sheet_names:
#         df0 = standardize_columns(read_sheet(xls, s).head(2))
#         hallmarks = {"random_id", "1st_term", "1st_fall", "1st_spring"}
#         if hallmarks.issubset(set(df0.columns)):
#             return s
#     return None
def find_gpa_sheet(xls: pd.ExcelFile) -> str | None:
    for s in xls.sheet_names:
        df0 = standardize_columns(read_sheet(xls, s).head(2))
        cols = set(df0.columns)
        if ("random_id" in cols and "1st_term" in cols and
            any(x in cols for x in ("1st_fall","1st_spring","1st_summer"))):
            return s
    return None

def merge_nondup(left: pd.DataFrame, right: pd.DataFrame | None, on: str = "random_id") -> pd.DataFrame:
    """Left-join while dropping any right-side columns that already exist on left (except the key)."""
    if right is None or right.empty:
        return left
    keep = [c for c in right.columns if (c == on) or (c not in left.columns)]
    return left.merge(right[keep], on=on, how="left")

def _make_merge_id(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace("\u00A0","", regex=False).str.replace("\u200B","", regex=False)  # strip NBSP / zero-width
    s = s.str.strip().str.upper().str.replace(r"[\u2010-\u2015\-]", "", regex=True)   # normalize dashes
    s = s.str.replace(r"[^A-Z0-9]", "", regex=True)                                   # keep A–Z0–9
    core = s.str.extract(r"(\d{5,})", expand=False)                                   # prefer trailing digits
    return core.fillna(s)


def build_information_only(excel_path: str, out_csv: str) -> None:
    xls = load_excel(excel_path)
    print(f"[cyan]Workbook sheets:[/cyan] {xls.sheet_names}")

    # ----- SUMMARY / INFO SHEET -----
    summary_sheet = find_sheet_by_cols(xls, ["random id", "cgpa", "graduating"]) or xls.sheet_names[0]
    df_summary_raw = standardize_columns(read_sheet(xls, summary_sheet))
    print(f"[cyan]Summary sheet:[/cyan] {summary_sheet}")
    print(f"[cyan]Summary cols (std, first 30):[/cyan] {list(df_summary_raw.columns)[:30]}")

    # Normalize IDs and build merge_id (maps S-00998 -> 00998, B-00998 -> 00998)
    df_summary_raw["random_id"] = _normalize_id(df_summary_raw["random_id"])
    df_summary_raw["merge_id"]  = _make_merge_id(df_summary_raw["random_id"])

    # If you had dup students, dedup by merge_id (can't sort by 1st_term here; it's not on this sheet)
    df_summary_raw = df_summary_raw.drop_duplicates(subset="merge_id", keep="first")

    # Engineered info/per-course
    engineered_info = prepare_information_features(df_summary_raw)

    # Start master
    master = df_summary_raw.copy()  # already has merge_id

    # Add engineered info on summary IDs
    extra_cols = [c for c in engineered_info.columns if c not in master.columns or c == "random_id"]
    master = merge_nondup(master, engineered_info[extra_cols], on="random_id")

    # ----- GPA SHEET (raw grid + engineered) -----
    gpa_sheet = find_gpa_sheet(xls)
    print(f"[cyan]GPA sheet:[/cyan] {gpa_sheet if gpa_sheet else 'NOT FOUND'}")

    gpa_long = None
    if gpa_sheet:
        df_gpa_raw = standardize_columns(read_sheet(xls, gpa_sheet))
        print(f"[cyan]GPA cols (std, first 30):[/cyan] {list(df_gpa_raw.columns)[:30]}")

        # Normalize & merge_id
        df_gpa_raw["random_id"] = _normalize_id(df_gpa_raw["random_id"])
        df_gpa_raw["merge_id"]  = _make_merge_id(df_gpa_raw["random_id"])

        # GPA grid columns in fixed order
        SEM_COLS_ORDERED = [
            "1st_fall","1st_spring","1st_summer",
            "2nd_fall","2nd_spring","2nd_summer",
            "3rd_fall","3rd_spring","3rd_summer",
            "4th_fall","4th_spring","4th_summer",
            "5th_fall","5th_spring","5th_summer",
            "6th_fall","6th_spring","6th_summer",
            "7th_fall","7th_spring","7th_summer",
            "8th_fall","8th_spring","8th_summer",
            "9th_fall",
        ]
        present = [c for c in SEM_COLS_ORDERED if c in df_gpa_raw.columns]
        for c in present:
            df_gpa_raw[c] = pd.to_numeric(df_gpa_raw[c], errors="coerce")

        keep = ["merge_id", "1st_term", "random_id"]
        if "summer_start" in df_gpa_raw.columns:
            keep.append("summer_start")
        keep += present
        raw_gpa_grid = df_gpa_raw[keep]

        # Engineered GPA features (+ long table)
        gpa_wide, gpa_long = build_term_gpa_features(df_gpa_raw)

        # Add merge_id onto gpa_wide via key_map
        key_map = df_gpa_raw[["random_id","merge_id"]].drop_duplicates()
        gpa_wide = gpa_wide.merge(key_map, on="random_id", how="left")

        # Merge both to master ON merge_id
        master = merge_nondup(master, raw_gpa_grid, on="merge_id")
        master = merge_nondup(master, gpa_wide,     on="merge_id")

        # Human-readable first term (from raw grid column 1st_term)
        if "1st_term" in master.columns:
            master["first_term_label"] = master["1st_term"].apply(
                lambda v: decode_psu_term(v)[0] if pd.notna(v) else None
            )

        # ---- Diagnostics
        overlap = master["merge_id"].isin(df_gpa_raw["merge_id"]).sum()
        print(f"[yellow]GPA merge overlap:[/yellow] {overlap} / {master['merge_id'].nunique()} students")
        for c in ["1st_fall","1st_spring","2nd_fall"]:
            if c in master.columns:
                print(f"[yellow]{c} non-null:[/yellow] {int(master[c].notna().sum())}")
        if present and master[present].isna().all(axis=None):
            print("[red]All GPA grid fields are NA after merge. Check ID normalization and column names.[/red]")

    # ----- SAVE
    save_csv(master, out_csv)
    print(f"[green]Saved:[/green] {out_csv}")

    if gpa_long is not None and not gpa_long.empty:
        sidecar = Path(out_csv).with_name("clean_features_terms_long.csv")
        save_csv(gpa_long, sidecar)
        print(f"[green]Saved:[/green] {sidecar}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build ALL features (raw + engineered)")
    b.add_argument("--excel", required=True, help="Path to ETM Excel workbook")
    b.add_argument("--out", required=True, help="Path to write clean feature CSV")

    args = ap.parse_args()
    if args.cmd == "build":
        build_information_only(args.excel, args.out)

if __name__ == "__main__":
    main()