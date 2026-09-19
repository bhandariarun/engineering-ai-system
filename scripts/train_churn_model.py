"""Small churn-model training script with MLflow tracking and model registration."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def load_telco_data() -> pd.DataFrame:
    csv_path = Path("data/telco_customer_churn.csv")
    if not csv_path.exists():
        raise FileNotFoundError("The Telco churn dataset was not found at data/telco_customer_churn.csv")
    data = pd.read_csv(csv_path)
    if "Churn" in data.columns:
        churn = data["Churn"]
        if churn.dtype == "object":
            data["Churn"] = churn.map({"Yes": 1, "No": 0})
        else:
            data["Churn"] = churn.astype(int)
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)
    return data


def build_model(name: str, **kwargs):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=500, **kwargs)
    if name == "random_forest":
        return RandomForestClassifier(random_state=42, **kwargs)
    raise ValueError(f"Unsupported model: {name}")


def main() -> None:
    mlflow.set_tracking_uri("file:./mlruns")
    df = load_telco_data()
    drop_cols = ["Churn"]
    if "customerID" in df.columns:
        drop_cols.append("customerID")
    X = df.drop(columns=drop_cols)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )

    runs = [
        ("logistic_regression", {"C": 0.5, "penalty": "l2"}),
        ("random_forest", {"n_estimators": 150, "max_depth": 8}),
        ("random_forest", {"n_estimators": 250, "max_depth": 12}),
    ]
    run_metrics: list[dict] = []

    for model_name, params in runs:
        with mlflow.start_run(run_name=f"{model_name}-{params}") as run:
            model = build_model(model_name, **params)
            pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            cm = confusion_matrix(y_test, y_pred)
            cm_path = Path(f"artifacts/{run.info.run_id}_cm.json")
            cm_path.parent.mkdir(exist_ok=True)
            cm_path.write_text(json.dumps(cm.tolist()), encoding="utf-8")
            mlflow.log_artifact(str(cm_path), artifact_path="artifacts")

            report_path = Path(f"artifacts/{run.info.run_id}_report.txt")
            report_path.write_text(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]), encoding="utf-8")
            mlflow.log_artifact(str(report_path), artifact_path="artifacts")

            payload = {"run_id": run.info.run_id, **metrics, "model_name": model_name}
            run_metrics.append(payload)
            print(json.dumps(payload, indent=2))

    best_run = max(run_metrics, key=lambda item: (item["f1"], item["roc_auc"], item["recall"]))
    client = MlflowClient()
    model_name = "telco-churn-model"
    register_result = mlflow.register_model(model_uri=f"runs:/{best_run['run_id']}/model", name=model_name)
    version = int(register_result.version)
    client.transition_model_version_stage(name=model_name, version=version, stage="Staging", archive_existing_versions=True)
    client.transition_model_version_stage(name=model_name, version=version, stage="Production", archive_existing_versions=False)
    print(json.dumps({
        "best_run_id": best_run["run_id"],
        "best_model": best_run["model_name"],
        "registered_model": model_name,
        "version": version,
        "f1": best_run["f1"],
        "roc_auc": best_run["roc_auc"],
    }, indent=2))
    print("Training complete; compare runs in MLflow UI at file:./mlruns")


if __name__ == "__main__":
    main()
