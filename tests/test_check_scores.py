import io
import sqlite3
import unittest
from contextlib import closing, redirect_stdout

from src import check_scores


class CheckScoresTests(unittest.TestCase):
    def test_report_with_sources_labels_and_nulls(self):
        with closing(sqlite3.connect(":memory:")) as connection:
            connection.execute(
                "CREATE TABLE scores (latency_ms REAL, risk_level TEXT, "
                "features_source TEXT, isFraud INTEGER)"
            )
            connection.executemany(
                "INSERT INTO scores VALUES (?, ?, ?, ?)",
                [
                    (10, "low", "feast", 1),
                    (20, "medium", "default", 1),
                    (30, "high", "feast", 1),
                    (40, "high", None, 0),
                    (None, None, "other", None),
                ],
            )
            with redirect_stdout(io.StringIO()) as output:
                check_scores.report(connection)
        text = output.getvalue()
        for expected in (
            "Total rows: 5", "Latency samples: 4", "Average latency: 25.00 ms",
            "Median latency: 25.00 ms", "95th percentile latency: 38.50 ms",
            "low: 1", "medium: 1", "high: 2", "feast: 2", "default: 1",
            "NULL: 1", "other: 1", "Actual frauds (isFraud = 1): 3",
            "Actual frauds scored medium or high: 2", "Rows without an actual label: 1",
        ):
            self.assertIn(expected, text)

    def test_empty_table_and_missing_optional_columns(self):
        with closing(sqlite3.connect(":memory:")) as connection:
            connection.execute("CREATE TABLE scores (latency_ms REAL, risk_level TEXT)")
            with redirect_stdout(io.StringIO()) as output:
                check_scores.report(connection)
        text = output.getvalue()
        self.assertIn("Total rows: 0", text)
        self.assertIn("Latency statistics: unavailable", text)
        self.assertIn("features_source is not stored", text)
        self.assertIn("ALTER TABLE scores ADD COLUMN features_source TEXT", text)
        self.assertIn("isFraud is not stored", text)

    def test_missing_table_or_required_columns(self):
        with closing(sqlite3.connect(":memory:")) as connection:
            with self.assertRaisesRegex(ValueError, "does not contain a scores table"):
                check_scores.report(connection)
            connection.execute("CREATE TABLE scores (risk_level TEXT)")
            with self.assertRaisesRegex(ValueError, "latency_ms"):
                check_scores.report(connection)


if __name__ == "__main__":
    unittest.main()
