import json
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import producer


class ProducerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.merged_path = root / "merged.parquet"
        self.features_path = root / "features.parquet"
        pd.DataFrame({
            "TransactionID": list(range(10)),
            "isFraud": [0] * 8 + [1, 0],
            "TransactionAmt": [float("nan")] * 10,
        }).to_parquet(self.merged_path, index=False)
        pd.DataFrame({
            "TransactionID": list(range(10)),
            "event_timestamp": pd.to_datetime([
                "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04",
                "2020-01-05", "2020-01-06", "2020-01-07", "2020-01-08",
                "2020-01-10", "2020-01-09",
            ]),
        }).to_parquet(self.features_path, index=False)
        self.paths = patch.multiple(
            producer, MERGED_PATH=self.merged_path, FEATURES_PATH=self.features_path
        )
        self.paths.start()
        self.addCleanup(self.paths.stop)

    def test_selects_first_transactions_in_chronological_validation_period(self):
        result = producer.validation_transactions(1000)
        self.assertEqual(result["TransactionID"].tolist(), [9, 8])
        self.assertEqual(result["isFraud"].tolist(), [0, 1])
        self.assertEqual(producer.validation_transactions(1)["TransactionID"].tolist(), [9])

    def test_sends_json_and_waits_for_each_delivery(self):
        with patch.object(producer, "KafkaProducer") as kafka, \
             patch.object(producer.time, "sleep") as sleep, \
             patch.object(sys, "argv", ["producer.py", "--count", "2", "--delay", "0.1"]):
            producer.main()
        client = kafka.return_value
        kafka.assert_called_once_with(bootstrap_servers="localhost:9092")
        self.assertEqual(client.send.call_count, 2)
        payloads = [json.loads(call.args[1]) for call in client.send.call_args_list]
        self.assertEqual([payload["TransactionID"] for payload in payloads], [9, 8])
        self.assertIsNone(payloads[0]["TransactionAmt"])
        self.assertEqual(payloads[1]["isFraud"], 1)
        self.assertIsInstance(payloads[0]["event_timestamp"], str)
        client.send.return_value.get.assert_called_with(timeout=30)
        self.assertEqual(client.send.return_value.get.call_count, 2)
        sleep.assert_called_once_with(0.1)
        client.close.assert_called_once()

    def test_mismatched_ids_fail(self):
        features = pd.read_parquet(self.features_path)
        features.loc[0, "TransactionID"] = 100
        features.to_parquet(self.features_path, index=False)
        with self.assertRaisesRegex(ValueError, "matching TransactionID"):
            producer.validation_transactions(1)

    def test_progress_reports_only_confirmed_messages(self):
        transactions = pd.concat([producer.validation_transactions(2)] * 101, ignore_index=True)
        output = io.StringIO()
        with patch.object(producer, "validation_transactions", return_value=transactions), \
             patch.object(producer, "KafkaProducer"), \
             patch.object(sys, "argv", ["producer.py", "--count", "202", "--delay", "0"]), \
             redirect_stdout(output):
            producer.main()
        self.assertEqual(output.getvalue().splitlines(), ["Sent 100 messages", "Sent 200 messages"])

    def test_delivery_failure_is_raised_and_producer_closed(self):
        with patch.object(producer, "KafkaProducer") as kafka, \
             patch.object(sys, "argv", ["producer.py", "--count", "1", "--delay", "0"]):
            kafka.return_value.send.return_value.get.side_effect = RuntimeError("delivery failed")
            with self.assertRaisesRegex(RuntimeError, "delivery failed"):
                producer.main()
        kafka.return_value.close.assert_called_once()

    def test_command_line_rejects_invalid_values(self):
        for option, value in (("--count", "0"), ("--delay", "-1"), ("--delay", "nan")):
            result = subprocess.run(
                [sys.executable, str(Path(producer.__file__)), option, value],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("error:", result.stderr)


if __name__ == "__main__":
    unittest.main()
