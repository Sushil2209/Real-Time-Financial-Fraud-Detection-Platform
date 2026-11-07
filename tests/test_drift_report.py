import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import drift_report, train_xgboost


def sample_inputs(size=10):
    features = pd.DataFrame({
        "TransactionID": range(size),
        "event_timestamp": pd.to_datetime("2020-01-01") + pd.to_timedelta(
            list(reversed(range(size))), unit="s"
        ),
        "txn_count_24h": range(size),
        "amount_vs_avg": [1.0] * size,
        "device_changed": [0] * size,
    })
    raw = {"TransactionID": list(reversed(range(size)))}
    for column in train_xgboost.RAW_COLUMNS:
        raw[column] = (
            ["category"] * size if column in train_xgboost.TEXT_COLUMNS
            else list(reversed(range(size)))
        )
    return features, pd.DataFrame(raw)


class DriftReportTests(unittest.TestCase):
    def test_selects_first_80_percent_and_preserves_scored_rows(self):
        features, raw = sample_inputs()
        scores = pd.DataFrame({"TransactionID": [0, 9, 0]})
        reference, current = drift_report.select_data(features, raw, scores)
        self.assertEqual(len(reference), 8)
        self.assertEqual(set(reference["txn_count_24h"]), set(range(2, 10)))
        self.assertEqual(current["txn_count_24h"].tolist(), [0, 9, 0])
        self.assertEqual(current["TransactionAmt"].tolist(), [0, 9, 0])
        self.assertEqual(list(current.columns), train_xgboost.FEATURE_COLUMNS)

    def test_reference_sample_is_capped_and_reproducible(self):
        features, raw = sample_inputs(25_010)
        scores = pd.DataFrame({"TransactionID": [0]})
        first, _ = drift_report.select_data(features, raw, scores)
        second, _ = drift_report.select_data(features, raw, scores)
        self.assertEqual(len(first), 20_000)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue((first["txn_count_24h"] >= 5002).all())

    def test_rejects_unmatched_empty_and_null_scores_and_null_timestamps(self):
        features, raw = sample_inputs()
        with self.assertRaisesRegex(ValueError, "no matching features"):
            drift_report.select_data(features, raw, pd.DataFrame({"TransactionID": [42]}))
        with self.assertRaisesRegex(ValueError, "scores table is empty"):
            drift_report.select_data(features, raw, pd.DataFrame({"TransactionID": []}))
        with self.assertRaisesRegex(ValueError, "non-missing TransactionID"):
            drift_report.select_data(features, raw, pd.DataFrame({"TransactionID": [None]}))
        features.loc[0, "event_timestamp"] = pd.NaT
        with self.assertRaisesRegex(ValueError, "event_timestamp"):
            drift_report.select_data(features, raw, pd.DataFrame({"TransactionID": [0]}))

    def test_evidently_summary_matches_actual_drift_tests(self):
        rng = np.random.default_rng(42)
        reference = pd.DataFrame({"shifted": rng.normal(size=500), "stable": rng.normal(size=500)})
        current = reference.copy()
        current["shifted"] += 10
        with patch.object(train_xgboost, "FEATURE_COLUMNS", ["shifted", "stable"]), \
             patch.object(train_xgboost, "TEXT_COLUMNS", []):
            snapshot = drift_report.make_report(reference, current)
            self.assertEqual(
                drift_report.summarize(snapshot),
                {"share_of_drifted_columns": 0.5, "drifted_columns": ["shifted"]},
            )

    def test_cli_simulates_only_current_data_and_selects_report_path(self):
        features, raw = sample_inputs()
        raw.loc[raw["TransactionID"] == 0, "C1"] = np.nan
        scores = pd.DataFrame({"TransactionID": [0, 9]})
        expected_reference, expected_current = drift_report.select_data(features, raw, scores)
        summary = {"share_of_drifted_columns": 0.5, "drifted_columns": ["C1"]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "scored.db"
            with closing(sqlite3.connect(database)) as connection:
                scores.to_sql("scores", connection, index=False)
            normal_report = root / "drift_report.html"
            simulated_report = root / "drift_report_simulated.html"
            summary_path = root / "drift_summary.json"
            normal_report.write_text("existing normal report", encoding="utf-8")

            class Snapshot:
                def save_html(self, path):
                    Path(path).write_text("report", encoding="utf-8")

            for simulate in (False, True):
                with self.subTest(simulate=simulate), \
                     patch.object(sys, "argv", ["drift_report.py"] + (["--simulate-drift"] if simulate else [])), \
                     patch.object(drift_report, "DATABASE_PATH", database), \
                     patch.object(drift_report, "REPORT_PATH", normal_report), \
                     patch.object(drift_report, "SIMULATED_REPORT_PATH", simulated_report), \
                     patch.object(drift_report, "SUMMARY_PATH", summary_path), \
                     patch.object(drift_report.pd, "read_parquet", side_effect=[features, raw]), \
                     patch.object(drift_report, "make_report", return_value=Snapshot()) as make_report, \
                     patch.object(drift_report, "summarize", return_value=summary):
                    drift_report.main()

                actual_reference, actual_current = make_report.call_args.args
                pd.testing.assert_frame_equal(actual_reference, expected_reference)
                expected = expected_current.copy()
                if simulate:
                    expected["TransactionAmt"] *= 3
                    for column in (f"C{i}" for i in range(1, 6)):
                        expected[column] += 10
                pd.testing.assert_frame_equal(actual_current, expected)
                self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8")), summary)
                self.assertEqual(normal_report.read_text(encoding="utf-8"), "report")
                if simulate:
                    self.assertEqual(simulated_report.read_text(encoding="utf-8"), "report")

    def test_simulated_report_with_evidently(self):
        features, raw = sample_inputs(500)
        rng = np.random.default_rng(42)
        raw["TransactionAmt"] = rng.uniform(10, 20, len(raw))
        for column in (f"C{i}" for i in range(1, 6)):
            raw[column] = rng.uniform(0, 1, len(raw))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "scored.db"
            with closing(sqlite3.connect(database)) as connection:
                pd.DataFrame({"TransactionID": range(100, 500)}).to_sql(
                    "scores", connection, index=False
                )
            report_path = root / "drift_report_simulated.html"
            summary_path = root / "drift_summary.json"
            with patch.object(sys, "argv", ["drift_report.py", "--simulate-drift"]), \
                 patch.object(drift_report, "DATABASE_PATH", database), \
                 patch.object(drift_report, "SIMULATED_REPORT_PATH", report_path), \
                 patch.object(drift_report, "SUMMARY_PATH", summary_path), \
                 patch.object(drift_report.pd, "read_parquet", side_effect=[features, raw]):
                drift_report.main()

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(report_path.is_file())
            self.assertGreater(report_path.stat().st_size, 0)
            self.assertTrue(
                {"TransactionAmt", "C1", "C2", "C3", "C4", "C5"}.issubset(summary["drifted_columns"])
            )


if __name__ == "__main__":
    unittest.main()
