#!/usr/bin/env python3
"""
Kaggle Real-Estate: dataset builder
----------------------------------

Purpose
- Read the provided CSVs in a directory (e.g., `train/`),
- Build a single modeling table with index (month × sector),
- Fill missing target for unseen (month, sector) with 0 (per host rules),
- Merge auxiliary tables by their natural keys:
  * month+sector → join on [month, sector]
  * month only  → join on [month]
  * year+sector → join on [year, sector]
  * year only   → join on [year] (broadcast to all months)
  * sector only → join on [sector] (broadcast to all months)
  * city only   → requires sector↔city mapping seen in any file; otherwise skip (with warning)

Usage
------
python kaggle_real_estate_dataset_builder.py \
  --data-dir ./train \
  --main-file new_house_transactions.csv \
  --output ./merged_training_table.csv \
  --verbose

Notes
- Designed for the user's 9 CSVs list, but will work generically with any similar schema.
- The tool attempts robust detection of key columns and common date formats.
- If your `month` format is known (e.g. "%Y-%m"), set `--month-format` to avoid parsing warnings.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# ---------------------
# Parsing & Normalizers
# ---------------------

def normalize_month(series: pd.Series, explicit_format: Optional[str] = None) -> pd.Series:
    """Normalize a month-like column to pandas Timestamp at month-start.
    Tries several common formats unless an explicit format is provided.
    """
    if explicit_format:
        return pd.to_datetime(series, format=explicit_format, errors="coerce").dt.to_period("M").dt.to_timestamp()

    # Try common patterns to reduce warnings; fall back to auto
    candidates = ("%Y-%m", "%Y/%m", "%Y%m", "%Y %b", "%Y-%m-%d", "%Y/%m/%d")
    for fmt in candidates:
        dt = pd.to_datetime(series, format=fmt, errors="ignore")
        if getattr(dt, "dtype", None) is not None and getattr(dt.dtype, "kind", "") == "M":
            return pd.to_datetime(dt, errors="coerce").dt.to_period("M").dt.to_timestamp()
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def normalize_sector(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


# ---------------------
# Column helpers
# ---------------------

def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.map(lambda x: str(x).strip())
    return df


def maybe_get_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    return None


def detect_target_col(cols: Sequence[str]) -> str:
    """Find target column in main file; raise if not found.
    Prefers exact known names, then fuzzy matching.
    """
    lower_map = {c.lower(): c for c in cols}
    exacts = [
        "amount_new_house_transactions",  # user's actual column
        "amount_new_house_transaction",
        "amount_of_new_house_transactions",
        "amount_of_new_house_transaction",
    ]
    for nm in exacts:
        if nm in lower_map:
            return lower_map[nm]

    # Fuzzy: must include amount+new+house+trans
    for c in cols:
        s = c.lower().replace(" ", "")
        if ("amount" in s) and ("new" in s) and ("house" in s) and ("trans" in s):
            return c
    raise KeyError(f"Could not detect target column among: {list(cols)}")


# ---------------------
# Core builder
# ---------------------

def build_main_index(
    df_main: pd.DataFrame,
    month_col: str = "month",
    sector_col: str = "sector",
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Create the X skeleton (month×sector), merge target and fill missing target with 0.
    Returns (X, target_col_name).
    """
    df_main = strip_columns(df_main.copy())

    # Normalize keys
    if month_col not in df_main.columns:
        # Try alternative names
        alt = maybe_get_col(df_main, {"month", "date"})
        if not alt:
            raise KeyError("Main file: missing month/date column")
        month_col = alt
    if sector_col not in df_main.columns:
        alt = maybe_get_col(df_main, {"sector", "sector_id"})
        if not alt:
            raise KeyError("Main file: missing sector column")
        sector_col = alt

    df_main[month_col] = normalize_month(df_main[month_col])
    df_main[sector_col] = normalize_sector(df_main[sector_col])

    # Detect target
    if target_col is None:
        target_col = detect_target_col(df_main.columns)

    # Build full grid
    months = sorted(df_main[month_col].dropna().unique())
    sectors = sorted(df_main[sector_col].dropna().unique())
    X = (
        pd.DataFrame(index=pd.MultiIndex.from_product([months, sectors], names=["month", "sector"]))
        .reset_index()
    )
    X["year"] = X["month"].dt.year

    # Merge target
    X = X.merge(
        df_main[[month_col, sector_col, target_col]].rename(columns={month_col: "month", sector_col: "sector"}),
        on=["month", "sector"],
        how="left",
    )
    X[target_col] = pd.to_numeric(X[target_col], errors="coerce").fillna(0.0)
    return X, target_col


def collect_sector_city_mappings(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    pairs = []
    for d in dfs:
        if "sector" in d.columns and "city" in d.columns:
            pairs.append(d[["sector", "city"]].dropna().drop_duplicates())
    if not pairs:
        return None
    mapping = pd.concat(pairs, ignore_index=True).drop_duplicates()
    return mapping


def merge_feature_table(
    X: pd.DataFrame,
    df: pd.DataFrame,
    file_stem: str,
    month_format: Optional[str] = None,
    sector_city_map: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge one feature table into X using detected key pattern."""
    df = strip_columns(df.copy())

    # Detect key candidates
    mcol = maybe_get_col(df, {"month", "date"})
    ycol = maybe_get_col(df, {"year", "data_year", "city_indicator_data_year", "stat_year"})
    scol = maybe_get_col(df, {"sector", "sector_id"})
    ccol = maybe_get_col(df, {"city", "city_id", "city_name"})

    if mcol is not None:
        df[mcol] = normalize_month(df[mcol], explicit_format=month_format)
        df = df.rename(columns={mcol: "month"})
    if ycol is not None:
        df[ycol] = pd.to_datetime(df[ycol].astype(str), errors="coerce").dt.year
        df = df.rename(columns={ycol: "year"})
    if scol is not None:
        df[scol] = normalize_sector(df[scol])
        df = df.rename(columns={scol: "sector"})
    if ccol is not None:
        df = df.rename(columns={ccol: "city"})

    # Decide merge key precedence
    if {"month", "sector"}.issubset(df.columns):
        key = ["month", "sector"]
    elif "month" in df.columns:
        key = ["month"]
    elif {"year", "sector"}.issubset(df.columns):
        key = ["year", "sector"]
    elif "year" in df.columns:
        key = ["year"]
    elif "sector" in df.columns:
        key = ["sector"]
    elif "city" in df.columns:
        key = ["city"]
    else:
        raise ValueError(f"[{file_stem}] No usable join key (month/date/year/sector/city)")

    logging.info("Merging %-35s on %s", file_stem, key)

    # Prepare X for needed keys
    if "year" in key and "year" not in X.columns:
        X["year"] = X["month"].dt.year
    if key == ["city"]:
        if sector_city_map is None:
            raise ValueError(
                f"[{file_stem}] Only 'city' key available but no sector↔city mapping found in any file."
            )
        # Ensure X has city via mapping
        if "city" not in X.columns:
            X = X.merge(sector_city_map, on="sector", how="left")

    # Merge
    if key == ["month", "sector"]:
        out = X.merge(df, on=["month", "sector"], how="left", suffixes=("", f"_{file_stem}"))
    elif key == ["month"]:
        out = X.merge(df, on=["month"], how="left", suffixes=("", f"_{file_stem}"))
    elif key == ["year", "sector"]:
        out = X.merge(df, on=["year", "sector"], how="left", suffixes=("", f"_{file_stem}"))
    elif key == ["year"]:
        out = X.merge(df, on=["year"], how="left", suffixes=("", f"_{file_stem}"))
    elif key == ["sector"]:
        out = X.merge(df, on=["sector"], how="left", suffixes=("", f"_{file_stem}"))
    elif key == ["city"]:
        out = X.merge(df, on=["city"], how="left", suffixes=("", f"_{file_stem}"))
    else:
        raise RuntimeError(f"Unexpected key: {key}")

    return out


def build_dataset(
    data_dir: Path,
    main_file: str = "new_house_transactions.csv",
    month_format: Optional[str] = None,
    exclude_files: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Build the full modeling table from a directory of CSVs.

    Returns (X, target_col_name)
    """
    exclude = set(exclude_files or [])
    main_path = (data_dir / main_file).resolve()
    if not main_path.exists():
        raise FileNotFoundError(f"Main file not found: {main_path}")

    logging.info("Reading main file: %s", main_path)
    df_main = pd.read_csv(main_path)

    X, target_col = build_main_index(df_main)

    # Gather feature files (all CSVs except main and excludes)
    all_csvs = sorted(p for p in data_dir.glob("*.csv"))
    feature_paths = [p for p in all_csvs if p.resolve() != main_path and p.name not in exclude]

    # First pass: read and standardize, also collect sector↔city map
    parsed_feature_dfs: List[Tuple[pd.DataFrame, str]] = []
    prelim_maps: List[pd.DataFrame] = []
    for p in feature_paths:
        try:
            d = pd.read_csv(p)
            d = strip_columns(d)
            # Normalize possible keys (light touch here; final normalization in merge_feature_table)
            mcol = maybe_get_col(d, {"month", "date"})
            ycol = maybe_get_col(d, {"year", "data_year", "city_indicator_data_year", "stat_year"})
            scol = maybe_get_col(d, {"sector", "sector_id"})
            ccol = maybe_get_col(d, {"city", "city_id", "city_name"})
            if mcol:
                d[mcol] = normalize_month(d[mcol], explicit_format=month_format)
                d = d.rename(columns={mcol: "month"})
            if ycol:
                d[ycol] = pd.to_datetime(d[ycol].astype(str), errors="coerce").dt.year
                d = d.rename(columns={ycol: "year"})
            if scol:
                d[scol] = normalize_sector(d[scol])
                d = d.rename(columns={scol: "sector"})
            if ccol:
                d = d.rename(columns={ccol: "city"})
            # collect potential mapping
            if {"sector", "city"}.issubset(d.columns):
                prelim_maps.append(d[["sector", "city"]].dropna().drop_duplicates())
            parsed_feature_dfs.append((d, p.stem))
        except Exception as e:
            logging.warning("Failed to pre-parse %s: %s", p.name, e)

    sector_city_map = collect_sector_city_mappings([d for d, _ in parsed_feature_dfs] + prelim_maps) if parsed_feature_dfs else None

    # Merge features one by one
    for d, stem in parsed_feature_dfs:
        try:
            X = merge_feature_table(X, d, stem, month_format=month_format, sector_city_map=sector_city_map)
        except Exception as e:
            logging.warning("Skip %s due to: %s", stem, e)

    return X, target_col


# ---------------------
# CLI
# ---------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build merged training table (month×sector)")
    ap.add_argument("--data-dir", type=Path, required=True, help="Directory containing CSVs (e.g., ./train)")
    ap.add_argument("--main-file", type=str, default="new_house_transactions.csv", help="Main CSV filename")
    ap.add_argument("--output", type=Path, default=None, help="Optional path to save merged CSV")
    ap.add_argument("--month-format", type=str, default=None, help="Explicit month date format (e.g., %Y-%m)")
    ap.add_argument("--exclude", type=str, nargs="*", default=[], help="Filenames to exclude from merge")
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    X, target_col = build_dataset(
        data_dir=args.data_dir,
        main_file=args.main_file,
        month_format=args.month_format,
        exclude_files=args.exclude,
    )

    logging.info("Merged shape: %s", X.shape)
    logging.info("Target column: %s", target_col)

    # quick health checks
    if target_col in X.columns:
        nan_count = int(X[target_col].isna().sum())
        logging.info("Target NaNs (should be 0 after fill): %d", nan_count)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        X.to_csv(args.output, index=False)
        logging.info("Saved merged table to %s", args.output)


if __name__ == "__main__":
    main()
