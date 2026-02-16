from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import standardize_columns
from .features import prepare_information_features, decode_psu_term
from .gpa_features import build_term_gpa_features, SEM_COLS_ORDERED
from .io import load_excel, read_sheet, save_csv, save_xlsx
from .validation import basic_sanity_checks


def _normalize_id(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    s = s.str.replace("\u00A0", "", regex=False).str.replace("\u200B", "", regex=False)
    s = s.str.strip().str.upper()
    s = s.str.replace(r"[\u2010-\u2015]", "-", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)

    m = s.str.extract(r"^([A-Z])[-_]?0*([0-9]+)$", expand=True)
    pref, digits = m[0], m[1]
    norm = (pref + "-" + digits.str.zfill(5)).astype("string")
    return norm.where(pref.notna() & digits.notna(), s)


def _make_merge_id(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    s = s.str.replace("\u00A0", "", regex=False).str.replace("\u200B", "", regex=False)
    s = s.str.strip().str.upper().str.replace(r"[\u2010-\u2015\-]", "", regex=True)
    s = s.str.replace(r"[^A-Z0-9]", "", regex=True)
    core = s.str.extract(r"(\d{5,})", expand=False)
    return core.fillna(s)


def find_sheet_by_cols(xls: pd.ExcelFile, must_have: list[str]) -> str | None:
    for s in xls.sheet_names:
        df0 = standardize_columns(read_sheet(xls, s).head(2))
        cols = list(df0.columns)
        if all(any(tok in c for c in cols) for tok in must_have):
            return s
    return None


def find_gpa_sheet(xls: pd.ExcelFile) -> str | None:
    for s in xls.sheet_names:
        df0 = standardize_columns(read_sheet(xls, s).head(2))
        cols = set(df0.columns)
        if (
            "random_id" in cols
            and "1st_term" in cols
            and any(x in cols for x in ("1st_fall", "1st_spring", "1st_summer"))
        ):
            return s
    return None


def merge_nondup(left: pd.DataFrame, right: pd.DataFrame | None, on: str) -> pd.DataFrame:
    if right is None or right.empty:
        return left
    keep = [c for c in right.columns if (c == on) or (c not in left.columns)]
    return left.merge(right[keep], on=on, how="left")


def _first_nonnull(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[0] if len(s2) else np.nan


def collapse_rows_by_merge_id(df_raw: pd.DataFrame, completeness_cols: list[str] | None = None) -> pd.DataFrame:
    df = df_raw.copy()
    key_cols = {"merge_id", "random_id"}

    if completeness_cols is None:
        non_key = [c for c in df.columns if c not in key_cols]
    else:
        non_key = [c for c in completeness_cols if c in df.columns and c not in key_cols]
        if not non_key:
            non_key = [c for c in df.columns if c not in key_cols]

    df["_nonnull_count"] = df[non_key].notna().sum(axis=1) if non_key else 0
    df["_has_dash"] = (
        df["random_id"].astype("string").str.contains("-", na=False).astype(int)
        if "random_id" in df.columns
        else 0
    )
    df = df.sort_values(["merge_id", "_nonnull_count", "_has_dash"], ascending=[True, False, False])

    drop_tmp = {"_nonnull_count", "_has_dash"}
    agg_cols = [c for c in df.columns if c not in drop_tmp]
    return df.groupby("merge_id", as_index=False)[agg_cols].agg(_first_nonnull)


def _parse_window(s: object) -> tuple[int | None, int | None]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None, None
    txt = str(s).replace("–", "-").replace("—", "-")
    m = re.search(r"(\d{1,3})\s*(?:-|to)\s*(\d{1,3})", txt, re.I)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def filter_credit_window(df: pd.DataFrame, target: str = "40-59") -> pd.DataFrame:
    if "credit_window" not in df.columns:
        return df

    lo_t, hi_t = _parse_window(target)
    if lo_t is None or hi_t is None:
        mask = df["credit_window"].astype("string").str.contains(str(target), na=False)
        return df.loc[mask].copy()

    lo_hi = df["credit_window"].apply(_parse_window)
    lo = lo_hi.apply(lambda t: t[0])
    hi = lo_hi.apply(lambda t: t[1])
    mask = (lo == lo_t) & (hi == hi_t)

    if mask.sum() == 0:
        mask = df["credit_window"].astype("string").str.contains(str(target), na=False)

    return df.loc[mask].copy()


def build_clean_processed(excel_path: str, out_dir: str, credit_window: str = "40-59") -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    xls = load_excel(excel_path)

    # Summary / info sheet
    summary_sheet = find_sheet_by_cols(xls, ["random", "cgpa"]) or xls.sheet_names[0]
    df_summary_raw = standardize_columns(read_sheet(xls, summary_sheet))
    if "random_id" not in df_summary_raw.columns:
        raise ValueError(f"Summary sheet '{summary_sheet}' is missing required column: random_id")

    df_summary_raw["random_id"] = _normalize_id(df_summary_raw["random_id"])
    df_summary_raw["merge_id"] = _make_merge_id(df_summary_raw["random_id"])

    df_summary_raw = filter_credit_window(df_summary_raw, credit_window)

    df_summary = collapse_rows_by_merge_id(df_summary_raw)


    engineered = prepare_information_features(df_summary)

    master = df_summary.copy()
    master = merge_nondup(master, engineered, on="random_id")

    # GPA sheet (optional)
    gpa_sheet = find_gpa_sheet(xls)
    gpa_long = None

    if gpa_sheet:
        df_gpa_raw = standardize_columns(read_sheet(xls, gpa_sheet))
        if "random_id" not in df_gpa_raw.columns:
            raise ValueError(f"GPA sheet '{gpa_sheet}' is missing required column: random_id")

        df_gpa_raw["random_id"] = _normalize_id(df_gpa_raw["random_id"])
        df_gpa_raw["merge_id"] = _make_merge_id(df_gpa_raw["random_id"])

        for c in [x for x in SEM_COLS_ORDERED if x in df_gpa_raw.columns]:
            df_gpa_raw[c] = pd.to_numeric(df_gpa_raw[c], errors="coerce")

        df_gpa = collapse_rows_by_merge_id(
            df_gpa_raw, completeness_cols=SEM_COLS_ORDERED + ["1st_term", "summer_start"]
        )

        keep = [
            c
            for c in (["merge_id", "random_id", "1st_term", "summer_start"] + [x for x in SEM_COLS_ORDERED if x in df_gpa.columns])
            if c in df_gpa.columns
        ]
        raw_gpa_grid = df_gpa[keep].copy()

        gpa_wide, gpa_long = build_term_gpa_features(df_gpa)

        if "merge_id" not in gpa_wide.columns:
            key_map = df_gpa[["random_id", "merge_id"]].drop_duplicates()
            gpa_wide = gpa_wide.merge(key_map, on="random_id", how="left")

        master = merge_nondup(master, raw_gpa_grid, on="merge_id")
        master = merge_nondup(master, gpa_wide, on="merge_id")

        if "1st_term" in master.columns:
            master["first_term_label"] = master["1st_term"].apply(lambda v: decode_psu_term(v)[0] if pd.notna(v) else None)

    # QC flags
    qc = basic_sanity_checks(master).add_prefix("qc_")
    master = pd.concat([master, qc], axis=1)

    # Save outputs
    save_csv(master, outp / "clean_processed_data.csv")

    sheets = {"clean_processed_data": master}
    if gpa_long is not None and not gpa_long.empty:
        sheets["terms_long"] = gpa_long
        save_csv(gpa_long, outp / "clean_features_terms_long.csv")

    save_xlsx(sheets, outp / "clean_processed_data.xlsx")


def main() -> None:
    ap = argparse.ArgumentParser(prog="etm-preprocess")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build clean_processed_data.csv/xlsx (default credit window 40-59)")
    b.add_argument("--excel", required=True, help="Path to ETM Excel workbook")
    b.add_argument("--out-dir", required=True, help="Output directory")
    b.add_argument("--credit-window", default="40-59", help="Target credit window (default: 40-59)")

    args = ap.parse_args()
    if args.cmd == "build":
        build_clean_processed(args.excel, args.out_dir, args.credit_window)


if __name__ == "__main__":
    main()
