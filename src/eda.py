"""
1.4 Exploratory Data Analysis for IBM Telco Customer Churn.

Covers the assignment list:
  - correlation coefficients
  - numeric and categorical associations with churn
  - binning
  - encoding
  - feature importance
  - univariate and bivariate charts
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

LOGGER = logging.getLogger(__name__)

LEAKAGE_OR_TARGET = [
    "Churn Label",
    "Churn Value",
    "Churn Score",   # already a model score — do not use as a predictor
    "Churn Reason",  # only filled for customers who already left
]
ID_OR_USELESS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
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


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved plot: %s", path)


def run_eda(processed_csv: Path, reports_dir: Path, plots_dir: Path) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_csv)
    LOGGER.info("EDA input shape: %s", df.shape)

    # ------------------------------------------------------------------
    # Correlation coefficients (numeric vs numeric, including target)
    # ------------------------------------------------------------------
    numeric_cols = [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "Churn Value",
        "Churn Score",
        "CLTV",
    ]
    corr = df[numeric_cols].corr(method="pearson")
    LOGGER.info("Pearson correlation matrix:\n%s", corr.round(3).to_string())
    corr.to_csv(reports_dir / "correlation_matrix.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Pearson correlation — numeric features")
    savefig(plots_dir / "01_correlation_heatmap.png")

    # ------------------------------------------------------------------
    # Categorical association with churn (mean of Churn Value = churn rate)
    # ------------------------------------------------------------------
    cat_cols = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
    ]
    cat_churn = {}
    for col in cat_cols:
        rates = (
            df.groupby(col)["Churn Value"]
            .mean()
            .mul(100)
            .round(2)
            .sort_values(ascending=False)
        )
        cat_churn[col] = rates.to_dict()
        LOGGER.info("Churn rate (%%) by %s:\n%s", col, rates.to_string())

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------
    df["Tenure Bin"] = pd.cut(
        df["Tenure Months"],
        bins=[-0.1, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
    )
    df["Monthly Charge Bin"] = pd.qcut(
        df["Monthly Charges"], q=4, labels=["Low", "Medium", "High", "Very High"]
    )
    df["CLTV Bin"] = pd.qcut(df["CLTV"], q=3, labels=["Low CLTV", "Medium CLTV", "High CLTV"])

    tenure_churn = (
        df.groupby("Tenure Bin", observed=False)["Churn Value"].mean().mul(100).round(2)
    )
    charge_churn = (
        df.groupby("Monthly Charge Bin", observed=False)["Churn Value"].mean().mul(100).round(2)
    )
    LOGGER.info("Churn %% by tenure bin:\n%s", tenure_churn.to_string())
    LOGGER.info("Churn %% by monthly charge bin:\n%s", charge_churn.to_string())

    # ------------------------------------------------------------------
    # Encoding (label + one-hot)
    # ------------------------------------------------------------------
    df_enc = df.copy()
    label_maps: dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[f"{col}_encoded"] = le.fit_transform(df_enc[col].astype(str))
        label_maps[col] = {cls: int(i) for i, cls in enumerate(le.classes_)}

    dummy_cols = ["Contract", "Internet Service", "Payment Method"]
    df_onehot = pd.get_dummies(df[dummy_cols], prefix=dummy_cols, drop_first=False)
    onehot_path = reports_dir / "encoded_onehot_sample.csv"
    pd.concat([df[["CustomerID", "Churn Value"]], df_onehot], axis=1).head(20).to_csv(
        onehot_path, index=False
    )
    LOGGER.info("Saved one-hot sample: %s", onehot_path)

    # ------------------------------------------------------------------
    # Feature importance (Random Forest on encoded non-leaky features)
    # ------------------------------------------------------------------
    feature_cols = [c for c in numeric_cols if c not in LEAKAGE_OR_TARGET]
    feature_cols += [f"{c}_encoded" for c in cat_cols]
    X = df_enc[feature_cols]
    y = df_enc["Churn Value"]
    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    rf.fit(X, y)
    importance = (
        pd.Series(rf.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
    )
    LOGGER.info("Feature importance:\n%s", importance.round(4).to_string())
    importance.to_csv(reports_dir / "feature_importance.csv")

    plt.figure(figsize=(9, 7))
    importance.head(15).sort_values().plot(kind="barh", color="#2E86AB")
    plt.xlabel("Importance")
    plt.title("Random Forest feature importance (no leakage features)")
    savefig(plots_dir / "02_feature_importance.png")

    # ------------------------------------------------------------------
    # Univariate charts
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn Label", hue="Churn Label", palette=["#4C9F38", "#C0392B"], legend=False)
    plt.title("Univariate: customer count by churn label")
    savefig(plots_dir / "03_univariate_churn_count.png")

    plt.figure(figsize=(7, 4))
    sns.histplot(df["Tenure Months"], bins=20, kde=True, color="#2E86AB")
    plt.title("Univariate: tenure (months)")
    savefig(plots_dir / "04_univariate_tenure.png")

    plt.figure(figsize=(7, 4))
    sns.histplot(df["Monthly Charges"], bins=20, kde=True, color="#E67E22")
    plt.title("Univariate: monthly charges")
    savefig(plots_dir / "05_univariate_monthly_charges.png")

    plt.figure(figsize=(8, 4))
    order = df["Contract"].value_counts().index
    sns.countplot(data=df, x="Contract", order=order, color="#2E86AB")
    plt.title("Univariate: contract type")
    savefig(plots_dir / "06_univariate_contract.png")

    # ------------------------------------------------------------------
    # Bivariate charts
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn Label", y="Tenure Months", hue="Churn Label",
                palette=["#4C9F38", "#C0392B"], legend=False)
    plt.title("Bivariate: tenure vs churn")
    savefig(plots_dir / "07_bivariate_tenure_vs_churn.png")

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn Label", y="Monthly Charges", hue="Churn Label",
                palette=["#4C9F38", "#C0392B"], legend=False)
    plt.title("Bivariate: monthly charges vs churn")
    savefig(plots_dir / "08_bivariate_charges_vs_churn.png")

    plt.figure(figsize=(7, 4))
    sns.barplot(
        data=df,
        x="Contract",
        y="Churn Value",
        estimator="mean",
        errorbar=None,
        color="#C0392B",
    )
    plt.ylabel("Churn rate")
    plt.title("Bivariate: churn rate by contract")
    savefig(plots_dir / "09_bivariate_churn_by_contract.png")

    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=df,
        x="Internet Service",
        y="Churn Value",
        estimator="mean",
        errorbar=None,
        color="#8E44AD",
    )
    plt.ylabel("Churn rate")
    plt.title("Bivariate: churn rate by internet service")
    savefig(plots_dir / "10_bivariate_churn_by_internet.png")

    plt.figure(figsize=(9, 4))
    sns.barplot(
        data=df,
        x="Payment Method",
        y="Churn Value",
        estimator="mean",
        errorbar=None,
        color="#16A085",
    )
    plt.ylabel("Churn rate")
    plt.xticks(rotation=15, ha="right")
    plt.title("Bivariate: churn rate by payment method")
    savefig(plots_dir / "11_bivariate_churn_by_payment.png")

    plt.figure(figsize=(7, 4))
    tenure_churn.plot(kind="bar", color="#2E86AB")
    plt.ylabel("Churn rate (%)")
    plt.title("Bivariate: churn rate by tenure bin")
    plt.xticks(rotation=15, ha="right")
    savefig(plots_dir / "12_bivariate_churn_by_tenure_bin.png")

    # Top churn reasons (only exists for churners)
    reason_counts = df["Churn Reason"].dropna().value_counts().head(10)
    plt.figure(figsize=(9, 5))
    reason_counts.sort_values().plot(kind="barh", color="#C0392B")
    plt.xlabel("Customers")
    plt.title("Top 10 churn reasons")
    savefig(plots_dir / "13_churn_reasons.png")

    report = {
        "step": "1.4_eda",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(df.shape[0]),
        "overall_churn_rate_pct": round(float(df["Churn Value"].mean() * 100), 2),
        "correlation_with_churn_value": corr["Churn Value"].sort_values(ascending=False).to_dict(),
        "churn_rate_by_category": cat_churn,
        "churn_rate_by_tenure_bin": tenure_churn.to_dict(),
        "churn_rate_by_monthly_charge_bin": charge_churn.to_dict(),
        "label_encoding_maps": label_maps,
        "feature_importance": importance.round(6).to_dict(),
        "plots": sorted(p.name for p in plots_dir.glob("*.png")),
    }
    report_path = reports_dir / "eda_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    LOGGER.info("Wrote EDA report: %s", report_path)
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    setup_logger(root / "reports" / "eda.log")
    run_eda(
        processed_csv=root / "data" / "processed" / "telco_preprocessed.csv",
        reports_dir=root / "reports",
        plots_dir=root / "reports" / "plots",
    )
