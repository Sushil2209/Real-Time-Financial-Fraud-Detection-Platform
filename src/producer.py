"""Replay validation transactions to Kafka.

Run from the project root: python src/producer.py --count 1000 --delay 0.05
"""

import argparse
import math
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer


ROOT = Path(__file__).resolve().parent.parent
MERGED_PATH = ROOT / "data" / "processed" / "merged.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return number


def validation_transactions(count: int) -> pd.DataFrame:
    features = pd.read_parquet(FEATURES_PATH, columns=["TransactionID", "event_timestamp"])
    merged_ids = pd.read_parquet(MERGED_PATH, columns=["TransactionID"])
    for name, data in (("features", features), ("merged", merged_ids)):
        if data["TransactionID"].isna().any() or data["TransactionID"].duplicated().any():
            raise ValueError(f"{name} data must have unique, non-missing TransactionID values")
    if features["event_timestamp"].isna().any():
        raise ValueError("event_timestamp must not contain missing values")
    if len(features) != len(merged_ids) or not features["TransactionID"].isin(
        merged_ids["TransactionID"]
    ).all():
        raise ValueError("features and merged data must have matching TransactionID values")

    ordered = features.sort_values("event_timestamp", kind="stable")
    selected = ordered.iloc[int(len(ordered) * 0.8):].head(count)
    transactions = pd.read_parquet(
        MERGED_PATH, filters=[("TransactionID", "in", selected["TransactionID"].tolist())]
    )
    if "isFraud" not in transactions:
        raise ValueError("merged data is missing isFraud")
    return selected.merge(transactions, on="TransactionID", how="left", validate="one_to_one")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=positive_int, default=1000)
    parser.add_argument("--delay", type=nonnegative_float, default=0.05)
    args = parser.parse_args()

    transactions = validation_transactions(args.count)
    messages = transactions.to_json(orient="records", lines=True, date_format="iso", date_unit="ns")
    producer = KafkaProducer(bootstrap_servers="localhost:9092")
    try:
        for index, message in enumerate(messages.splitlines(), start=1):
            producer.send("transactions", message.encode("utf-8")).get(timeout=30)
            if index % 100 == 0:
                print(f"Sent {index} messages")
            if index < len(transactions) and args.delay:
                time.sleep(args.delay)
    finally:
        producer.close()


if __name__ == "__main__":
    main()
