from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print

from .io import load_excel, read_sheet, save_csv
from .cleaning import standardize_columns
from .features import prepare_information_features, decode_psu_term
from .gpa_features import build_term_gpa_features


# -----------------------------
# Constants
# -----------------------------
SEM_COLS_ORDERED = [
    "1st_fall", "1st_spring", "1st_summer",
    "2nd_fall", "2nd_spring", "2nd_summer",
    "3rd_fall", "3rd_spring", "3rd_summer",
    "4th_fall", "4th_spring", "4th_summer",
    "5th_fall", "5th_spring", "5th_summer",
    "6th_fall", "6th_spring", "6th_summer",
    "7th_fall", "7th_spring", "7th_summer",
    "8th_fall", "8th_spring", "8th_summer",
    "9th_fall",
]


# -----------------------------
# ID utilities
# -----------------------------
def _normalize_id(series: pd.Series) -> pd.Series:
    """Canonicalize Random ID formatting.

    Normalizes whitespace / unicode dashes and formats IDs like:
      S00001, S-00001, s 00001  -> S-00001
      B00001, B-00001           -> B-00001

    This prevents many-to-many merge explosions when merge_id is the digit core.
    """
    s = series.astype("string")
    s = s.str.replace("\u00A0", "", regex=False).str.replace("\u200B", "", regex=False)  # NBSP / zero-width
    s = s.str.strip().str.upper()
    s = s.str.replace(r"[\u2010-\u2015]", "-", regex=True)  # unicode hyphen variants
    s = s.str.replace(r"\s+", "", regex=True)

    # Try to parse a leading letter + digits (optionally separated by dash/underscore)
    m = s.str.extract(r"^([A-Z])[-_]?0*([0-9]+)$", expand=True)
    pref, digits = m[0], m[1]
    norm = (pref + "-" + digits.str.zfill(5)).astype("string")

    return norm.where(pref.notna() & digits.notna(), s)


def _make_merge_id(series: pd.Series) -> pd.Series:
    """
    Create a merge-safe ID by extracting the numeric core from random_id.

    Examples:
      S-00998 -> 00998
      B-00998 -> 00998
      "  s 00998 " -> 00998

    If digits are not found, falls back to an alnum-cleaned version.
    """
    s = series.astype(str)
    s = s.str.replace("\u00A0", "", regex=False).str.replace("\u200B", "", regex=False)  # NBSP / zero-width
    s = s.str.strip().str.upper().str.replace(r"[\u2010-\u2015\-]", "", regex=True)     # normalize dashes
    s = s.str.replace(r"[^A-Z0-9]", "", regex=True)                                      # keep A–Z0–9
    core = s.str.extract(r"(\d{5,})", expand=False)                                      # prefer digit core
    return core.fillna(s)


# -----------------------------
# Sheet finding utilities
# -----------------------------
def find_sheet_by_cols(xls: pd.ExcelFile, must_have: list[str]) -> str | None:
    """
    Find a sheet where standardized column names contain each of the must_have tokens.
    Token matching is substring-based.
    """
    for s in xls.sheet_names:
        df0 = standardize_columns(read_sheet(xls, s).head(2))
        if all(any(m in c for c in df0.columns) for m in must_have):
            return s
    return None


def find_gpa_sheet(xls: pd.ExcelFile) -> str | None:
    """
    Find the GPA grid sheet: must contain random_id, 1st_term, and at least one term GPA slot.
    """
    for s in xls.sheet_names:
        df0 = standardize_columns(read_sheet(xls, s).head(2))
        cols = set(df0.columns)
        if ("random_id" in cols and "1st_term" in cols and
                any(x in cols for x in ("1st_fall", "1st_spring", "1st_summer"))):
            return s
    return None


# -----------------------------
# Merge helpers
# -----------------------------
def merge_nondup(left: pd.DataFrame, right: pd.DataFrame | None, on: str) -> pd.DataFrame:
    """
    Left-join while dropping any right-side columns that already exist on left (except the key).
    Prevents accidental overwrites and keeps merges readable.
    """
    if right is None or right.empty:
        return left
    keep = [c for c in right.columns if (c == on) or (c not in left.columns)]
    return left.merge(right[keep], on=on, how="left")


def collapse_gpa_rows_by_merge_id(df_gpa_raw: pd.DataFrame, term_cols: list[str]) -> pd.DataFrame:
    """
    Ensure 1 row per merge_id for GPA data (prevents many-to-many merge explosions).

    Strategy:
      1) Compute completeness score = number of non-null term slots per row
      2) Sort so most complete row per merge_id comes first
      3) Group by merge_id, take first non-null per column

    Time complexity:
      - O(n * t) to compute completeness (t = #term cols)
      - O(n log n) for sorting
      - O(n * c) for group aggregation
    """
    df = df_gpa_raw.copy()

    present_cols = [c for c in term_cols if c in df.columns]
    if present_cols:
        df["_term_nonnull_count"] = df[present_cols].notna().sum(axis=1)
    else:
        df["_term_nonnull_count"] = 0

    sort_cols = ["merge_id", "_term_nonnull_count"]
    asc = [True, False]

    if "1st_term" in df.columns:
        df["_has_1st_term"] = df["1st_term"].notna().astype(int)
        sort_cols += ["_has_1st_term", "1st_term"]
        asc += [False, True]

    df = df.sort_values(sort_cols, ascending=asc)

    def first_nonnull(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan

    drop_tmp = {"_term_nonnull_count", "_has_1st_term"}
    agg_cols = [c for c in df.columns if c not in drop_tmp]

    out = (
        df.groupby("merge_id", as_index=False)[agg_cols]
        .agg(first_nonnull)
    )
    return out


def collapse_summary_rows_by_merge_id(df_summary_raw: pd.DataFrame) -> pd.DataFrame:
    """Ensure 1 row per merge_id for the summary/info sheet.

    The raw workbook sometimes includes multiple rows per student (e.g., S00001 and S-00001).
    If we dedupe by `keep="first"` we may keep an incomplete row and discard a richer one.
    This function chooses the most complete row per merge_id and then takes first non-null per column.

    Complexity:
      - O(n * c) to compute completeness (c=#columns)
      - O(n log n) to sort
      - O(n * c) to aggregate
    """
    df = df_summary_raw.copy()

    # score completeness across all non-key columns
    key_cols = {"merge_id", "random_id"}
    non_key = [c for c in df.columns if c not in key_cols]
    if non_key:
        df["_nonnull_count"] = df[non_key].notna().sum(axis=1)
    else:
        df["_nonnull_count"] = 0

    # prefer rows with canonical dashed IDs if present
    if "random_id" in df.columns:
        df["_has_dash"] = df["random_id"].astype("string").str.contains("-", na=False).astype(int)
    else:
        df["_has_dash"] = 0

    df = df.sort_values(["merge_id", "_nonnull_count", "_has_dash"], ascending=[True, False, False])

    def first_nonnull(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan

    drop_tmp = {"_nonnull_count", "_has_dash"}
    agg_cols = [c for c in df.columns if c not in drop_tmp]

    out = df.groupby("merge_id", as_index=False)[agg_cols].agg(first_nonnull)
    return out



# -----------------------------
# Pipeline
# -----------------------------
def build_information_only(excel_path: str, out_csv: str) -> None:
    xls = load_excel(excel_path)
    print(f"[cyan]Workbook sheets:[/cyan] {xls.sheet_names}")

    # ----- SUMMARY / INFO SHEET -----
    summary_sheet = find_sheet_by_cols(xls, ["random id", "cgpa", "graduating"]) or xls.sheet_names[0]
    df_summary_raw = standardize_columns(read_sheet(xls, summary_sheet))
    print(f"[cyan]Summary sheet:[/cyan] {summary_sheet}")
    print(f"[cyan]Summary cols (std, first 30):[/cyan] {list(df_summary_raw.columns)[:30]}")

    if "random_id" not in df_summary_raw.columns:
        raise ValueError("Summary sheet is missing required column: random_id")

    # Normalize IDs and build merge_id
    df_summary_raw["random_id"] = _normalize_id(df_summary_raw["random_id"])
    df_summary_raw["merge_id"] = _make_merge_id(df_summary_raw["random_id"])

    # ---- CRITICAL FIX: Collapse duplicates BEFORE any feature building/merging
    # Ensures 1 row per merge_id and keeps the most complete record.
    before_sum = len(df_summary_raw)
    df_summary_raw = collapse_summary_rows_by_merge_id(df_summary_raw)
    after_sum = len(df_summary_raw)
    print(f"[yellow]Summary collapse by merge_id:[/yellow] {before_sum} -> {after_sum}")
    if df_summary_raw["merge_id"].duplicated().any():
        raise RuntimeError("Summary collapse failed: merge_id still has duplicates")

    # Engineered info + per-course mastery features (uses summary sheet only)
    engineered_info = prepare_information_features(df_summary_raw)

    # Start master (summary raw)
    master = df_summary_raw.copy()

    # Add engineered info on random_id
    extra_cols = [c for c in engineered_info.columns if c not in master.columns or c == "random_id"]
    master = merge_nondup(master, engineered_info[extra_cols], on="random_id")

    # ----- GPA SHEET (raw grid + engineered) -----
    gpa_sheet = find_gpa_sheet(xls)
    print(f"[cyan]GPA sheet:[/cyan] {gpa_sheet if gpa_sheet else 'NOT FOUND'}")

    gpa_long = None
    if gpa_sheet:
        df_gpa_raw = standardize_columns(read_sheet(xls, gpa_sheet))
        print(f"[cyan]GPA cols (std, first 30):[/cyan] {list(df_gpa_raw.columns)[:30]}")

        if "random_id" not in df_gpa_raw.columns:
            raise ValueError("GPA sheet is missing required column: random_id")

        # Normalize & merge_id
        df_gpa_raw["random_id"] = _normalize_id(df_gpa_raw["random_id"])
        df_gpa_raw["merge_id"] = _make_merge_id(df_gpa_raw["random_id"])

        # Numeric coercion for term slots
        present = [c for c in SEM_COLS_ORDERED if c in df_gpa_raw.columns]
        for c in present:
            df_gpa_raw[c] = pd.to_numeric(df_gpa_raw[c], errors="coerce")

        # ---- CRITICAL FIX: Collapse duplicates BEFORE any merging or feature building
        before_gpa = len(df_gpa_raw)
        df_gpa_raw = collapse_gpa_rows_by_merge_id(df_gpa_raw, SEM_COLS_ORDERED)
        after_gpa = len(df_gpa_raw)
        print(f"[yellow]GPA collapse by merge_id:[/yellow] {before_gpa} -> {after_gpa}")
        if df_gpa_raw["merge_id"].duplicated().any():
            raise RuntimeError("GPA collapse failed: merge_id still has duplicates")

        # Build raw GPA grid (for transparency)
        keep = ["merge_id"]
        if "random_id" in df_gpa_raw.columns:
            keep.append("random_id")
        if "1st_term" in df_gpa_raw.columns:
            keep.append("1st_term")
        if "summer_start" in df_gpa_raw.columns:
            keep.append("summer_start")
        keep += present

        raw_gpa_grid = df_gpa_raw[[c for c in keep if c in df_gpa_raw.columns]].copy()

        # Engineered GPA features (+ long table)
        # IMPORTANT: build on the collapsed df so gpa_wide is 1 row per student
        gpa_wide, gpa_long = build_term_gpa_features(df_gpa_raw)

        # Attach merge_id to gpa_wide
        # If gpa_wide already includes merge_id, this is a no-op
        if "merge_id" not in gpa_wide.columns:
            key_map = df_gpa_raw[["random_id", "merge_id"]].drop_duplicates()
            gpa_wide = gpa_wide.merge(key_map, on="random_id", how="left")

        # Merge to master ON merge_id
        master = merge_nondup(master, raw_gpa_grid, on="merge_id")
        master = merge_nondup(master, gpa_wide, on="merge_id")

        # Human-readable first term label
        if "1st_term" in master.columns:
            master["first_term_label"] = master["1st_term"].apply(
                lambda v: decode_psu_term(v)[0] if pd.notna(v) else None
            )

        # Diagnostics: overlap count
        overlap = master["merge_id"].isin(df_gpa_raw["merge_id"]).sum()
        print(f"[yellow]GPA merge overlap:[/yellow] {overlap} / {master['merge_id'].nunique()} students")
        for c in ["1st_fall", "1st_spring", "2nd_fall"]:
            if c in master.columns:
                print(f"[yellow]{c} non-null:[/yellow] {int(master[c].notna().sum())}")

    # ----- FINAL DEDUPE CHECK -----
    # At this point master SHOULD be 1 row per merge_id.
    before_master = len(master)
    master = master.drop_duplicates(subset="merge_id", keep="first").copy()
    after_master = len(master)
    if before_master != after_master:
        print(f"[red]WARNING:[/red] master had duplicate merge_id rows: {before_master} -> {after_master}")

    # ----- SAVE
    save_csv(master, out_csv)
    print(f"[green]Saved:[/green] {out_csv}")

    if gpa_long is not None and not gpa_long.empty:
        sidecar = Path(out_csv).with_name("clean_features_terms_long.csv")
        save_csv(gpa_long, sidecar)
        print(f"[green]Saved:[/green] {sidecar}")


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
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
