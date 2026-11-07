"""Compare scored transactions with a sample from the XGBoost training period.

Run from the project root: .\\.venv\\Scripts\\python.exe src\\drift_report.py
Add --simulate-drift to triple current TransactionAmt values and add 10 to
current C1-C5 values. The simulation saves a separate HTML report; both modes
overwrite reports/drift_summary.json with the latest report's summary.
The current features are reconstructed from the saved parquet files, not from
the feature values actually used by the scoring API.
"""

import argparse
import json
import os
import sqlite3
from contextlib import closing

os.environ["DO_NOT_TRACK"] = "1"
os.environ["EVIDENTLY_DISABLE_TELEMETRY"] = "1"

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

if __package__:
    from . import train_xgboost
else:
    import train_xgboost


ROOT = train_xgboost.ROOT
DATABASE_PATH = ROOT / "data" / "processed" / "scored.db"
REPORT_PATH = ROOT / "reports" / "drift_report.html"
SIMULATED_REPORT_PATH = ROOT / "reports" / "drift_report_simulated.html"
SUMMARY_PATH = ROOT / "reports" / "drift_summary.json"
SAMPLE_SIZE = 20_000


def select_data(
    features: pd.DataFrame, merged: pd.DataFrame, scores: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = train_xgboost.join_features(features, merged)
    if "event_timestamp" not in data or data["event_timestamp"].isna().any():
        raise ValueError("features must have non-missing event_timestamp values")
    ordered = data.sort_values("event_timestamp", kind="stable")
    training = ordered.iloc[:int(len(ordered) * 0.8)]
    if training.empty:
        raise ValueError("The training period is empty")
    reference = training.sample(n=min(SAMPLE_SIZE, len(training)), random_state=42)

    if "TransactionID" not in scores or scores["TransactionID"].isna().any():
        raise ValueError("scores must have non-missing TransactionID values")
    if scores.empty:
        raise ValueError("The scores table is empty; score transactions first")
    current = scores[["TransactionID"]].merge(
        data[["TransactionID"] + train_xgboost.FEATURE_COLUMNS],
        on="TransactionID", how="left", sort=False, validate="many_to_one", indicator=True,
    )
    if not current["_merge"].eq("both").all():
        missing = current.loc[current["_merge"].ne("both"), "TransactionID"]
        raise ValueError(f"{len(missing)} scored rows have no matching features; IDs: {missing.head().tolist()}")
    return reference[train_xgboost.FEATURE_COLUMNS], current[train_xgboost.FEATURE_COLUMNS]


def make_report(reference: pd.DataFrame, current: pd.DataFrame):
    columns = train_xgboost.FEATURE_COLUMNS
    text_columns = train_xgboost.TEXT_COLUMNS
    definition = DataDefinition(
        numerical_columns=[column for column in columns if column not in text_columns],
        categorical_columns=text_columns,
    )
    reference = reference.copy()
    current = current.copy()
    for column in columns:
        if column not in text_columns:
            reference[column] = reference[column].astype("float64")
            current[column] = current[column].astype("float64")
    report = Report([DataDriftPreset(columns=columns)], include_tests=True)
    return report.run(
        current_data=Dataset.from_pandas(current, data_definition=definition),
        reference_data=Dataset.from_pandas(reference, data_definition=definition),
    )


def summarize(snapshot) -> dict:
    result = snapshot.dict()
    counts = [
        metric["value"] for metric in result["metrics"]
        if metric["config"]["type"] == "evidently:metric_v2:DriftedColumnsCount"
    ]
    tests = [
        test for test in result["tests"]
        if test["id"] == "drift"
        and test["metric_config"]["params"]["type"] == "evidently:metric_v2:ValueDrift"
    ]
    statuses = {test["metric_config"]["params"]["column"]: test["status"] for test in tests}
    columns = train_xgboost.FEATURE_COLUMNS
    if (
        len(counts) != 1 or len(tests) != len(columns) or set(statuses) != set(columns)
        or not set(statuses.values()).issubset({"SUCCESS", "FAIL"})
    ):
        raise ValueError("Evidently did not return a drift result for every feature")
    drifted = [column for column in columns if statuses[column] == "FAIL"]
    share = counts[0]["share"]
    if counts[0]["count"] != len(drifted) or abs(share - len(drifted) / len(columns)) > 1e-9:
        raise ValueError("Evidently's drifted-column count and individual results disagree")
    return {"share_of_drifted_columns": share, "drifted_columns": drifted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulate-drift", action="store_true",
        help="Triple current TransactionAmt and add 10 to current C1-C5 values",
    )
    args = parser.parse_args()

    features = pd.read_parquet(train_xgboost.FEATURES_PATH)
    merged = pd.read_parquet(
        train_xgboost.MERGED_PATH, columns=["TransactionID"] + train_xgboost.RAW_COLUMNS
    )
    with closing(sqlite3.connect(f"{DATABASE_PATH.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        scores = pd.read_sql_query("SELECT TransactionID FROM scores", connection)
    reference, current = select_data(features, merged, scores)
    if args.simulate_drift:
        current = current.copy()
        current["TransactionAmt"] = current["TransactionAmt"] * 3
        for column in (f"C{i}" for i in range(1, 6)):
            current[column] = current[column] + 10
    report_path = SIMULATED_REPORT_PATH if args.simulate_drift else REPORT_PATH
    snapshot = make_report(reference, current)
    summary = summarize(snapshot)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(report_path))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Compared {len(reference)} reference rows with {len(current)} scored rows.")
    print(json.dumps(summary, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
