"""Run an Evidently data drift and target drift report for the churn pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evidently import Dataset, Report
from evidently.metrics import DriftedColumnsCount, MeanValue, ValueDrift


def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/telco_customer_churn.csv")
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(int)
    return df


def inject_drift(df: pd.DataFrame, fraction: float = 0.3) -> pd.DataFrame:
    current = df.copy()
    current["MonthlyCharges"] = current["MonthlyCharges"] + (current.index % 3) * 15
    current["Contract"] = current["Contract"].replace({
        "One year": "Month-to-month",
        "Two year": "Month-to-month",
    }).where(current["Contract"].isin(["Month-to-month", "One year", "Two year"]), current["Contract"])
    current.loc[current.index[: max(1, int(len(current) * fraction))], "Contract"] = "Month-to-month"
    current.loc[current.index[: max(1, int(len(current) * fraction))], "Churn"] = 1 - current.loc[current.index[: max(1, int(len(current) * fraction))], "Churn"]
    return current


def main() -> None:
    df = load_data()
    reference = df.sample(frac=0.7, random_state=42).reset_index(drop=True)
    current = inject_drift(df.drop(index=reference.index).reset_index(drop=True), fraction=0.25)
    report = Report(metrics=[
        ValueDrift(column="MonthlyCharges"),
        ValueDrift(column="Contract"),
        MeanValue(column="MonthlyCharges"),
        DriftedColumnsCount(),
    ])
    snapshot = report.run(current_data=current, reference_data=reference)

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    html_path = out_dir / "churn_drift_report.html"
    html_path.write_text(snapshot.get_html_str(as_iframe=False), encoding="utf-8")
    json_path = out_dir / "churn_drift_summary.json"
    json_path.write_text(json.dumps(snapshot.json(), indent=2), encoding="utf-8")
    print(snapshot.json())
    print(f"Saved drift report HTML to {html_path}")


if __name__ == "__main__":
    main()
