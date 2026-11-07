"""Train and evaluate an XGBoost fraud model using a chronological split.

Rebuild features.parquet after changes to the feature-generation pipeline.
Categories are learned from training data; unseen validation categories are
treated as missing.
"""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import auc, f1_score, precision_recall_curve, precision_score, recall_score
from xgboost import XGBClassifier

if __package__:
    from .model_features import FEATURE_COLUMNS, MODEL_PATH, RAW_COLUMNS, ROOT, TEXT_COLUMNS
else:
    from model_features import FEATURE_COLUMNS, MODEL_PATH, RAW_COLUMNS, ROOT, TEXT_COLUMNS

FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
MERGED_PATH = ROOT / "data" / "processed" / "merged.parquet"
METRICS_PATH = ROOT / "reports" / "xgb_metrics.json"
TARGET = "isFraud"
THRESHOLD = 0.5


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def join_features(features: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """Join raw columns without changing the feature rows or their order."""
    for name, data in (("features", features), ("merged", merged)):
        if "TransactionID" not in data:
            raise ValueError(f"{name} data is missing TransactionID")
        if data["TransactionID"].isna().any() or data["TransactionID"].duplicated().any():
            raise ValueError(f"{name} data must have unique, non-missing TransactionID values")
    missing = set(RAW_COLUMNS) - set(merged.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")
    result = features.merge(
        merged[["TransactionID"] + RAW_COLUMNS],
        on="TransactionID", how="left", sort=False, validate="one_to_one", indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise ValueError("Some feature rows have no matching TransactionID in merged data")
    return result.drop(columns="_merge")


def split_by_time(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by event time, preserving input order for equal timestamps."""
    required = FEATURE_COLUMNS + [TARGET, "event_timestamp"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data["event_timestamp"].isna().any():
        raise ValueError("event_timestamp must not contain missing values")
    if not data[TARGET].isin([0, 1]).all():
        raise ValueError("isFraud must contain only binary labels 0 and 1")

    ordered = data.sort_values("event_timestamp", kind="stable")
    split = int(len(ordered) * 0.8)
    train, validation = ordered.iloc[:split], ordered.iloc[split:]
    if train[TARGET].nunique() != 2:
        raise ValueError("Training data must contain both fraud and non-fraud labels")
    if validation.empty or not validation[TARGET].eq(1).any():
        raise ValueError("Validation data must contain at least one fraud label")
    return train, validation


def prepare_features(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train[FEATURE_COLUMNS].copy()
    validation_x = validation[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        if column in TEXT_COLUMNS:
            train_x[column] = train_x[column].astype("category")
            known = validation_x[column].isin(train_x[column].cat.categories)
            validation_x[column] = validation_x[column].where(known).astype(train_x[column].dtype)
        else:
            train_x[column] = train_x[column].astype("float64")
            validation_x[column] = validation_x[column].astype("float64")
    return train_x, validation_x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depth", type=positive_int, default=6)
    parser.add_argument("--n-estimators", type=positive_int, default=100)
    parser.add_argument("--output-model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--output-metrics", type=Path,
        help="Metrics JSON path; defaults to xgb_metrics.json for the default model, "
             "otherwise to <output-model-stem>_metrics.json next to the model",
    )
    args = parser.parse_args()
    model_path = args.output_model
    metrics_path = args.output_metrics
    if metrics_path is None:
        metrics_path = (
            METRICS_PATH if model_path.resolve() == MODEL_PATH.resolve()
            else model_path.with_name(f"{model_path.stem}_metrics.json")
        )
    if model_path.resolve() == metrics_path.resolve():
        parser.error("model and metrics output paths must be different")
    if model_path.resolve() != MODEL_PATH.resolve() and metrics_path.resolve() == METRICS_PATH.resolve():
        parser.error("a non-default model must not overwrite the current model's metrics")

    features = pd.read_parquet(FEATURES_PATH)
    merged = pd.read_parquet(MERGED_PATH, columns=["TransactionID"] + RAW_COLUMNS)
    train, validation = split_by_time(join_features(features, merged))
    train_x, validation_x = prepare_features(train, validation)
    train_y = train[TARGET].astype("int64")
    validation_y = validation[TARGET].astype("int64")

    positives = int(train_y.sum())
    negatives = len(train_y) - positives
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": 0.1,
        "objective": "binary:logistic",
        "scale_pos_weight": negatives / positives,
        "random_state": 42,
        "n_jobs": 4,
        "enable_categorical": True,
        "tree_method": "hist",
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run():
        mlflow.log_params({**params, "threshold": THRESHOLD, "num_features": len(FEATURE_COLUMNS)})
        model = XGBClassifier(**params)
        model.fit(train_x, train_y)
        probabilities = model.predict_proba(validation_x)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(int)
        precision_curve, recall_curve, _ = precision_recall_curve(validation_y, probabilities)
        metrics = {
            "pr_auc": float(auc(recall_curve, precision_curve)),
            "precision": float(precision_score(validation_y, predictions, zero_division=0)),
            "recall": float(recall_score(validation_y, predictions, zero_division=0)),
            "f1": float(f1_score(validation_y, predictions, zero_division=0)),
        }
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
