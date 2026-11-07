import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from kafka import OffsetAndMetadata, TopicPartition

from src import consumer


class ConsumerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / "processed" / "scored.db"
        self.path_patch = patch.object(consumer, "DATABASE_PATH", self.database)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.message = {
            "TransactionID": 123, "card1": 456, "TransactionAmt": None, "isFraud": 1,
        }
        self.response = Mock()
        self.response.json.return_value = {
            "TransactionID": 123, "fraud_score": 0.9, "risk_level": "high", "latency_ms": 12.5,
            "features_source": "feast",
        }
        self.session = Mock()
        self.session.post.return_value = self.response

    def test_saves_api_result_and_actual_label(self):
        with closing(sqlite3.connect(":memory:")) as connection:
            consumer.create_table(connection)
            consumer.create_table(connection)
            consumer.score_message(self.message, self.session, connection)
            row = connection.execute(
                "SELECT TransactionID, card1, TransactionAmt, fraud_score, risk_level, "
                "latency_ms, scored_at, isFraud, features_source FROM scores"
            ).fetchone()
        self.session.post.assert_called_once_with(consumer.SCORE_URL, json=self.message, timeout=10)
        self.assertEqual(row[:6], (123, 456, None, 0.9, "high", 12.5))
        self.assertIsNotNone(datetime.fromisoformat(row[6]).tzinfo)
        self.assertEqual(row[7], 1)
        self.assertEqual(row[8], "feast")

    def test_existing_table_is_migrated_without_changing_old_rows(self):
        with closing(sqlite3.connect(":memory:")) as connection:
            connection.execute(
                "CREATE TABLE scores (TransactionID INTEGER, card1 INTEGER, TransactionAmt REAL, "
                "fraud_score REAL, risk_level TEXT, latency_ms REAL, scored_at TEXT, isFraud INTEGER)"
            )
            connection.execute("INSERT INTO scores (TransactionID) VALUES (42)")
            consumer.create_table(connection)
            consumer.create_table(connection)
            consumer.score_message(self.message, self.session, connection)
            rows = connection.execute(
                "SELECT TransactionID, features_source FROM scores ORDER BY TransactionID"
            ).fetchall()
        self.assertEqual(rows, [(42, None), (123, "feast")])

    def test_migration_propagates_other_database_errors(self):
        connection = Mock()
        connection.execute.side_effect = [None, sqlite3.OperationalError("disk error")]
        with self.assertRaisesRegex(sqlite3.OperationalError, "disk error"):
            consumer.create_table(connection)

    def test_missing_feature_source_warns_without_saving(self):
        del self.response.json.return_value["features_source"]
        with closing(sqlite3.connect(":memory:")) as connection:
            consumer.create_table(connection)
            with redirect_stderr(io.StringIO()) as output:
                consumer.score_message(self.message, self.session, connection)
            self.assertIn("features_source", output.getvalue())
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0], 0)

    def test_api_errors_warn_without_saving(self):
        with closing(sqlite3.connect(":memory:")) as connection:
            consumer.create_table(connection)
            for error in (requests.Timeout("timeout"), requests.HTTPError("bad status")):
                self.session.post.side_effect = error
                with redirect_stderr(io.StringIO()) as output:
                    consumer.score_message(self.message, self.session, connection)
                self.assertIn("Warning:", output.getvalue())
            self.session.post.side_effect = None
            self.response.json.return_value = {}
            with redirect_stderr(io.StringIO()) as output:
                consumer.score_message(self.message, self.session, connection)
            self.assertIn("Warning:", output.getvalue())
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0], 0)

    def test_consumes_up_to_limit_and_commits_after_each(self):
        records = [
            SimpleNamespace(value=self.message, topic="transactions", partition=0, offset=i)
            for i in range(2)
        ]
        kafka = Mock()
        kafka.poll.side_effect = [{0: [record]} for record in records]
        session_context = Mock()
        session_context.__enter__ = Mock(return_value=self.session)
        session_context.__exit__ = Mock(return_value=False)
        with patch.object(consumer, "KafkaConsumer", return_value=kafka) as factory, \
             patch.object(consumer.requests, "Session", return_value=session_context), \
             redirect_stdout(io.StringIO()):
            consumer.consume(2)
        options = factory.call_args.kwargs
        self.assertEqual(factory.call_args.args, ("transactions",))
        self.assertEqual(options["bootstrap_servers"], "localhost:9092")
        self.assertEqual(options["group_id"], "fraud-scorer")
        self.assertEqual(options["auto_offset_reset"], "earliest")
        self.assertFalse(options["enable_auto_commit"])
        self.assertEqual(options["value_deserializer"](json.dumps(self.message).encode()), self.message)
        self.assertEqual(kafka.poll.call_count, 2)
        self.assertEqual(kafka.commit.call_count, 2)
        kafka.commit.assert_called_with({TopicPartition("transactions", 0): OffsetAndMetadata(2, "")})
        kafka.close.assert_called_once()
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0], 2)

    def test_api_failure_does_not_stop_later_messages(self):
        records = [
            SimpleNamespace(value=self.message, topic="transactions", partition=0, offset=i)
            for i in range(2)
        ]
        kafka = Mock()
        kafka.poll.side_effect = [{0: [record]} for record in records]
        self.session.post.side_effect = [requests.Timeout("timeout"), self.response]
        session_context = Mock()
        session_context.__enter__ = Mock(return_value=self.session)
        session_context.__exit__ = Mock(return_value=False)
        with patch.object(consumer, "KafkaConsumer", return_value=kafka), \
             patch.object(consumer.requests, "Session", return_value=session_context), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            consumer.consume(2)
        self.assertIn("Warning:", errors.getvalue())
        self.assertEqual(kafka.commit.call_count, 2)
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0], 1)

    def test_database_failure_does_not_commit_offset(self):
        record = SimpleNamespace(value=self.message, topic="transactions", partition=0, offset=0)
        kafka = Mock()
        kafka.poll.return_value = {0: [record]}
        session_context = Mock()
        session_context.__enter__ = Mock(return_value=self.session)
        session_context.__exit__ = Mock(return_value=False)
        with patch.object(consumer, "KafkaConsumer", return_value=kafka), \
             patch.object(consumer.requests, "Session", return_value=session_context), \
             patch.object(consumer, "score_message", side_effect=sqlite3.OperationalError("disk error")):
            with self.assertRaisesRegex(sqlite3.OperationalError, "disk error"):
                consumer.consume(1)
        kafka.commit.assert_not_called()
        kafka.close.assert_called_once()

    def test_idle_timeout_and_close(self):
        kafka = Mock()
        kafka.poll.return_value = {}
        session_context = Mock()
        session_context.__enter__ = Mock(return_value=self.session)
        session_context.__exit__ = Mock(return_value=False)
        output = io.StringIO()
        with patch.object(consumer, "KafkaConsumer", return_value=kafka), \
             patch.object(consumer.requests, "Session", return_value=session_context), \
             patch.object(consumer.time, "monotonic", side_effect=[100, 100, 115]), \
             redirect_stdout(output):
            consumer.consume(5)
        kafka.poll.assert_called_once_with(timeout_ms=15000, max_records=1)
        kafka.close.assert_called_once()
        self.assertIn("15 seconds", output.getvalue())

    def test_command_line_rejects_nonpositive_limit(self):
        result = subprocess.run(
            [sys.executable, str(Path(consumer.__file__)), "--max-messages", "0"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("positive integer", result.stderr)

    def test_command_line_passes_group_id_to_kafka(self):
        record = SimpleNamespace(value=self.message, topic="transactions", partition=0, offset=0)
        kafka = Mock()
        kafka.poll.return_value = {0: [record]}
        session_context = Mock()
        session_context.__enter__ = Mock(return_value=self.session)
        session_context.__exit__ = Mock(return_value=False)
        with patch.object(sys, "argv", ["consumer.py", "--max-messages", "1", "--group-id", "test-group"]), \
             patch.object(consumer, "KafkaConsumer", return_value=kafka) as factory, \
             patch.object(consumer.requests, "Session", return_value=session_context), \
             redirect_stdout(io.StringIO()):
            consumer.main()
        self.assertEqual(factory.call_args.kwargs["group_id"], "test-group")
        kafka.commit.assert_called_once()

    def test_command_line_uses_default_group_id(self):
        with patch.object(sys, "argv", ["consumer.py"]), \
             patch.object(consumer, "consume") as consume:
            consumer.main()
        consume.assert_called_once_with(1000, "fraud-scorer")


if __name__ == "__main__":
    unittest.main()
