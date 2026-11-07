"""Score Kafka transactions and save the results.

Run from the project root: python src/consumer.py --max-messages 1000
The idle timeout is 15 seconds. API failures count toward the message limit.
"""

import argparse
import json
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import requests
from kafka import KafkaConsumer, OffsetAndMetadata, TopicPartition


ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = ROOT / "data" / "processed" / "scored.db"
SCORE_URL = "http://127.0.0.1:8000/score"
IDLE_TIMEOUT = 15


def create_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE IF NOT EXISTS scores (
            TransactionID INTEGER,
            card1 INTEGER,
            TransactionAmt REAL,
            fraud_score REAL,
            risk_level TEXT,
            latency_ms REAL,
            scored_at TEXT,
            isFraud INTEGER,
            features_source TEXT
        )"""
    )
    try:
        connection.execute("ALTER TABLE scores ADD COLUMN features_source TEXT")
    except sqlite3.OperationalError as exc:
        if str(exc) != "duplicate column name: features_source":
            raise
    connection.commit()


def score_message(message: dict, session: requests.Session, connection: sqlite3.Connection) -> None:
    try:
        response = session.post(SCORE_URL, json=message, timeout=10)
        response.raise_for_status()
        result = response.json()
        fraud_score = result["fraud_score"]
        risk_level = result["risk_level"]
        latency_ms = result["latency_ms"]
        features_source = result["features_source"]
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        print(f"Warning: could not score transaction {message.get('TransactionID')}: {exc}", file=sys.stderr)
        return

    connection.execute(
        """INSERT INTO scores
        (TransactionID, card1, TransactionAmt, fraud_score, risk_level, latency_ms, scored_at, isFraud, features_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            message["TransactionID"], message["card1"], message["TransactionAmt"],
            fraud_score, risk_level, latency_ms, datetime.now(timezone.utc).isoformat(),
            message["isFraud"], features_source,
        ),
    )
    connection.commit()


def consume(max_messages: int, group_id: str = "fraud-scorer") -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DATABASE_PATH)) as connection, requests.Session() as session:
        create_table(connection)
        consumer = KafkaConsumer(
            "transactions",
            bootstrap_servers="localhost:9092",
            group_id=group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        )
        try:
            count = 0
            deadline = time.monotonic() + IDLE_TIMEOUT
            while count < max_messages:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print("Stopped after 15 seconds without a new message.")
                    break
                batches = consumer.poll(timeout_ms=max(1, int(remaining * 1000)), max_records=1)
                for records in batches.values():
                    for record in records:
                        score_message(record.value, session, connection)
                        consumer.commit({
                            TopicPartition(record.topic, record.partition):
                                OffsetAndMetadata(record.offset + 1, "")
                        })
                        count += 1
                        deadline = time.monotonic() + IDLE_TIMEOUT
            print(f"Processed {count} messages.")
        finally:
            consumer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-messages", type=int, default=1000)
    parser.add_argument("--group-id", default="fraud-scorer", help="Kafka consumer group ID")
    args = parser.parse_args()
    if args.max_messages <= 0:
        parser.error("--max-messages must be a positive integer")
    consume(args.max_messages, args.group_id)


if __name__ == "__main__":
    main()
