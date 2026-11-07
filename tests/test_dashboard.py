import shutil
import sqlite3
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src import dashboard


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        (root / "src").mkdir()
        (root / "data" / "processed").mkdir(parents=True)
        self.app_path = root / "src" / "dashboard.py"
        shutil.copy2(dashboard.__file__, self.app_path)
        self.database = root / "data" / "processed" / "scored.db"
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute(
                "CREATE TABLE scores (TransactionID INTEGER, card1 INTEGER, "
                "TransactionAmt REAL, fraud_score REAL, risk_level TEXT)"
            )
            connection.executemany(
                "INSERT INTO scores VALUES (?, ?, ?, ?, ?)",
                [
                    (1, 101, 10.0, 0.9, "high"),
                    (2, 102, 20.0, 0.8, "medium"),
                    (3, 103, 30.0, 0.1, "low"),
                    (1, 101, 10.0, 0.85, "high"),
                ],
            )

    def test_dashboard_and_feedback_round_trip(self):
        app = AppTest.from_file(str(self.app_path), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual([metric.value for metric in app.metric], ["4", "2", "1", "1", "2"])
        self.assertEqual(app.dataframe[0].value["TransactionID"].tolist(), [1, 1])
        self.assertEqual(app.dataframe[1].value["TransactionID"].tolist(), [1, 1, 2])

        app.selectbox[1].select("Under Review")
        app.text_area[0].input("Investigating")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[4].value, "1")
        self.assertEqual(app.selectbox[1].value, "Under Review")
        self.assertEqual(app.text_area[0].value, "Investigating")
        self.assertEqual(app.success[0].value, "Feedback saved.")

        app.selectbox[1].select("Fraud")
        app.text_area[0].input("Confirmed")
        app.button[0].click().run()
        with closing(sqlite3.connect(self.database)) as connection:
            rows = connection.execute(
                "SELECT TransactionID, status, updated_at, note FROM feedback"
            ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0][0], rows[0][1], rows[0][3]), (1, "Fraud", "Confirmed"))
        self.assertTrue(rows[0][2].endswith("+00:00"))

        app.slider[0].set_value(0.9).run()
        self.assertEqual(app.info[1].value, "No transactions exceed the alert threshold.")
        app.selectbox[0].select(2).run()
        self.assertEqual(app.selectbox[1].value, "New")
        self.assertEqual(app.text_area[0].value, "")

    def test_empty_scores_and_missing_database(self):
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("DELETE FROM scores")
        app = AppTest.from_file(str(self.app_path), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual([metric.value for metric in app.metric], ["0", "0", "0", "0", "0"])
        self.assertIn("no flagged transactions", app.info[2].value.lower())

        self.database.unlink()
        app = AppTest.from_file(str(self.app_path), default_timeout=30).run()
        self.assertTrue(app.error)
        self.assertFalse(self.database.exists())

    def test_invalid_status_is_rejected(self):
        with closing(sqlite3.connect(self.database)) as connection:
            dashboard.create_feedback_table(connection)
            with self.assertRaisesRegex(ValueError, "Unknown case status"):
                dashboard.save_feedback(connection, 1, "Invalid", "")
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0], 0)

    def test_scores_cache_expires_after_five_seconds(self):
        dashboard.read_scores.clear()
        self.assertEqual(len(dashboard.read_scores(self.database)), 4)
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("INSERT INTO scores VALUES (4, 104, 40.0, 0.7, 'medium')")
        self.assertEqual(len(dashboard.read_scores(self.database)), 4)
        time.sleep(5.1)
        self.assertEqual(len(dashboard.read_scores(self.database)), 5)


if __name__ == "__main__":
    unittest.main()
