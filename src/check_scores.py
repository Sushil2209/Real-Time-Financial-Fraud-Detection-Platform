"""Summarize saved scores. Run from any directory: python src/check_scores.py."""

import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from statistics import mean, median


DATABASE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "scored.db"


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Calculate a percentile using linear interpolation."""
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def print_counts(connection: sqlite3.Connection, column: str, expected: tuple[str, ...]) -> None:
    # The column name is selected by this script, not supplied by a user.
    counts = dict(connection.execute(f'SELECT "{column}", COUNT(*) FROM scores GROUP BY "{column}"'))
    for value in expected:
        print(f"  {value}: {counts.pop(value, 0)}")
    for value, count in counts.items():
        print(f"  {value if value is not None else 'NULL'}: {count}")


def report(connection: sqlite3.Connection) -> None:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(scores)")}
    if not columns:
        raise ValueError("The database does not contain a scores table.")
    missing = {"latency_ms", "risk_level"} - columns
    if missing:
        raise ValueError(f"scores is missing required columns: {', '.join(sorted(missing))}")

    total = connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    latencies = sorted(row[0] for row in connection.execute(
        "SELECT latency_ms FROM scores WHERE latency_ms IS NOT NULL"
    ))
    print(f"Total rows: {total}")
    print(f"Latency samples: {len(latencies)} (excluding NULL)")
    if latencies:
        print(f"Average latency: {mean(latencies):.2f} ms")
        print(f"Median latency: {median(latencies):.2f} ms")
        print(f"95th percentile latency: {percentile(latencies, 0.95):.2f} ms (linear interpolation)")
    else:
        print("Latency statistics: unavailable (no non-NULL latency values)")

    print("Risk levels:")
    print_counts(connection, "risk_level", ("low", "medium", "high"))

    if "features_source" in columns:
        print("Feature sources:")
        print_counts(connection, "features_source", ("feast", "default"))
    else:
        print(
            "Feature sources: unavailable (features_source is not stored).\n"
            "To store future sources in consumer.py:\n"
            "  1. Add features_source TEXT to CREATE TABLE and migrate the existing table "
            "once with ALTER TABLE scores ADD COLUMN features_source TEXT.\n"
            "  2. In score_message, read result['features_source'] and add "
            "features_source to the INSERT columns, placeholders, and values.\n"
            "  Existing rows will remain NULL; their sources cannot be recovered from this table."
        )

    if "isFraud" in columns:
        frauds, detected = connection.execute(
            "SELECT COUNT(*), COUNT(CASE WHEN risk_level IN ('medium', 'high') THEN 1 END) "
            "FROM scores WHERE isFraud = 1"
        ).fetchone()
        missing_labels = connection.execute(
            "SELECT COUNT(*) FROM scores WHERE isFraud IS NULL"
        ).fetchone()[0]
        print(f"Actual frauds (isFraud = 1): {frauds}")
        print(f"Actual frauds scored medium or high: {detected}")
        if missing_labels:
            print(f"Rows without an actual label: {missing_labels}")
    else:
        print("Actual fraud counts: unavailable (isFraud is not stored).")


def main() -> int:
    try:
        with closing(sqlite3.connect(DATABASE_PATH.as_uri() + "?mode=ro", uri=True)) as connection:
            report(connection)
    except (sqlite3.Error, ValueError) as exc:
        print(f"Error reading scores: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
