"""
1.3 Data Pre-processing for IBM Telco Customer Churn.

What this script does (matches the assignment list):
  1. Load the raw dataset
  2. Display data types
  3. Display summary statistics
  4. Check missing values
  5. Impute missing numeric data
  6. Normalize numeric columns
  7. Save cleaned data + a JSON report for the dashboard
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

LOGGER = logging.getLogger(__name__)

# Columns that are IDs, constants, or geography we do not use in ML
DROP_FOR_ML = [
    "CustomerID",
    "Count",          # always 1
    "Country",        # always United States
    "State",          # always California
    "Lat Long",
    "Zip Code",
    "Latitude",
    "Longitude",
]


def setup_logger(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def load_raw(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading raw file: %s", path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    LOGGER.info("Loaded shape: %s rows x %s columns", df.shape[0], df.shape[1])
    return df


def display_dtypes(df: pd.DataFrame) -> dict:
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    LOGGER.info("Data types:")
    for col, dtype in dtypes.items():
        LOGGER.info("  %-22s %s", col, dtype)
    return dtypes


def summary_statistics(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=[np.number])
    stats = numeric.describe().T
    stats["missing"] = numeric.isna().sum()
    LOGGER.info("Summary statistics (numeric columns):\n%s", stats.to_string())
    return json.loads(stats.to_json(orient="index"))


def missing_report(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    report = {
        col: {"count": int(missing[col]), "percent": float(pct[col])}
        for col in df.columns
        if missing[col] > 0
    }
    LOGGER.info("Missing-value report: %s", report if report else "no pandas-null values")
    return report


def impute_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Total Charges is stored as text. Brand-new customers (Tenure Months = 0)
    have a blank Total Charges. That is the main numeric missing-value case.
    """
    df = df.copy()
    notes: dict = {}

    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    n_missing = int(df["Total Charges"].isna().sum())
    LOGGER.info("Total Charges missing after numeric conversion: %s", n_missing)

    # New customers have not been billed yet -> total charges should be 0
    mask_new = df["Total Charges"].isna() & (df["Tenure Months"] == 0)
    df.loc[mask_new, "Total Charges"] = 0.0

    # Any leftover numeric holes: fill with the column median
    leftover = int(df["Total Charges"].isna().sum())
    median_tc = float(df["Total Charges"].median())
    if leftover:
        df["Total Charges"] = df["Total Charges"].fillna(median_tc)

    notes["total_charges_converted"] = True
    notes["total_charges_blank_originally"] = n_missing
    notes["filled_with_zero_new_customers"] = int(mask_new.sum())
    notes["filled_with_median"] = leftover
    notes["median_total_charges"] = median_tc
    LOGGER.info("Imputation notes: %s", notes)
    return df, notes


def normalize_numeric(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """Min-Max scale selected numeric columns into 0-1 range."""
    df = df.copy()
    scaler = MinMaxScaler()
    present = [c for c in columns if c in df.columns]
    scaled_names = [f"{c}_scaled" for c in present]
    df[scaled_names] = scaler.fit_transform(df[present])
    scale_info = {
        col: {"min": float(mn), "max": float(mx)}
        for col, mn, mx in zip(present, scaler.data_min_, scaler.data_max_)
    }
    LOGGER.info("Normalized columns: %s", present)
    return df, scale_info


def run_preprocess(raw_path: Path, processed_dir: Path, reports_dir: Path) -> dict:
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(raw_path)
    dtypes_before = display_dtypes(df)
    stats_before = summary_statistics(df)
    missing_before = missing_report(df)

    df, impute_notes = impute_numeric(df)

    # Keep a modelling-friendly copy (IDs dropped later in EDA)
    scale_cols = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]
    df, scale_info = normalize_numeric(df, scale_cols)

    dtypes_after = display_dtypes(df)
    missing_after = missing_report(df)

    out_csv = processed_dir / "telco_preprocessed.csv"
    df.to_csv(out_csv, index=False)
    LOGGER.info("Wrote processed dataset: %s", out_csv)

    report = {
        "step": "1.3_preprocess",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(raw_path),
        "n_rows": int(df.shape[0]),
        "n_cols_after": int(df.shape[1]),
        "dtypes_before": dtypes_before,
        "dtypes_after": dtypes_after,
        "summary_statistics_before": stats_before,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "imputation": impute_notes,
        "normalization": scale_info,
        "processed_csv": str(out_csv),
        "churn_counts": df["Churn Label"].value_counts().to_dict(),
    }
    report_path = reports_dir / "preprocess_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    LOGGER.info("Wrote preprocess report: %s", report_path)
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    setup_logger(root / "reports" / "preprocess.log")
    run_preprocess(
        raw_path=root / "data" / "raw" / "Telco_customer_churn.xlsx",
        processed_dir=root / "data" / "processed",
        reports_dir=root / "reports",
    )
