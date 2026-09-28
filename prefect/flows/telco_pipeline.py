"""
Prefect flow for Assignment 1.3 + 1.4 + 1.5.

Runs on YOUR laptop via a Prefect worker.
Prefect Cloud only stores the schedule, run history, logs, and dashboard.

Usage (after prefect cloud login):
    python -m prefect.flows.telco_pipeline          # one local test
    prefect deploy ...                              # register + schedule
    prefect worker start --pool local-process       # pick up scheduled runs
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, task
from prefect.logging import get_run_logger

# Project root = telco_project/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eda import run_eda  # noqa: E402
from preprocess import run_preprocess, setup_logger  # noqa: E402


def _paths() -> dict[str, Path]:
    raw_xlsx = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn.xlsx"
    raw_csv = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn.csv"
    raw = raw_xlsx if raw_xlsx.exists() else raw_csv
    return {
        "raw": raw,
        "processed_dir": PROJECT_ROOT / "data" / "processed",
        "reports_dir": PROJECT_ROOT / "reports",
        "plots_dir": PROJECT_ROOT / "reports" / "plots",
    }


@task(name="1.3-preprocess", retries=1, retry_delay_seconds=10)
def preprocess_task() -> dict:
    """Assignment 1.3 — stats, missing values, impute, dtypes, normalize."""
    logger = get_run_logger()
    p = _paths()
    p["reports_dir"].mkdir(parents=True, exist_ok=True)
    setup_logger(p["reports_dir"] / "pipeline.log")

    if not p["raw"].exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {p['raw']}. "
            "Put Telco_customer_churn.xlsx or .csv in data/raw/"
        )

    logger.info("Raw file: %s", p["raw"])
    report = run_preprocess(p["raw"], p["processed_dir"], p["reports_dir"])
    logger.info(
        "Preprocess done | rows=%s | blank TotalCharges fixed=%s",
        report.get("n_rows"),
        report.get("imputation", {}).get("total_charges_blank_originally"),
    )
    return report


@task(name="1.4-eda")
def eda_task() -> dict:
    """Assignment 1.4 — correlation, binning, encoding, importance, charts."""
    logger = get_run_logger()
    p = _paths()
    processed = p["processed_dir"] / "telco_preprocessed.csv"
    if not processed.exists():
        raise FileNotFoundError(f"Processed file missing: {processed}")

    report = run_eda(processed, p["reports_dir"], p["plots_dir"])
    logger.info(
        "EDA done | churn_rate=%s%% | plots=%s | top_feature=%s",
        report.get("overall_churn_rate_pct"),
        len(report.get("plots", [])),
        next(iter(report.get("feature_importance", {})), None),
    )
    return report


@task(name="write-metrics")
def write_metrics_task(preprocess_report: dict, eda_report: dict, duration: float) -> dict:
    """Write a small JSON file the report / dashboard can quote."""
    logger = get_run_logger()
    p = _paths()
    metrics = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "status": "SUCCEEDED",
        "duration_seconds": round(duration, 2),
        "n_rows": preprocess_report.get("n_rows"),
        "missing_total_charges_originally": preprocess_report.get("imputation", {}).get(
            "total_charges_blank_originally"
        ),
        "overall_churn_rate_pct": eda_report.get("overall_churn_rate_pct"),
        "top_feature": next(iter(eda_report.get("feature_importance", {})), None),
        "n_plots": len(eda_report.get("plots", [])),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "orchestrator": "prefect-cloud",
        "compute": "local-worker",
    }
    path = p["reports_dir"] / "pipeline_metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote metrics: %s", path)
    logger.info("Metrics payload: %s", metrics)
    return metrics


@flow(name="telco-churn-dataops", log_prints=True)
def telco_pipeline() -> dict:
    """
    Assignment 1.5 workflow.

    Task 1 = pre-processing (1.3)
    Task 2 = EDA (1.4)
    Task 3 = metrics file for the report
    """
    logger = get_run_logger()
    started = time.time()
    logger.info("===== telco-churn-dataops start =====")
    logger.info("Project root: %s", PROJECT_ROOT)

    preprocess_report = preprocess_task()
    eda_report = eda_task()
    metrics = write_metrics_task(preprocess_report, eda_report, time.time() - started)

    logger.info("===== telco-churn-dataops finished in %.1fs =====", time.time() - started)
    return metrics


if __name__ == "__main__":
    telco_pipeline()
