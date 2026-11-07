import unittest

import pandas as pd

from src.build_features import START_DATE, build_features


class BuildFeaturesTests(unittest.TestCase):
    def test_features_use_card_history_and_restore_input_order(self):
        transactions = pd.DataFrame(
            {
                "TransactionID": [3, 1, 2, 4, 5, 6, 7],
                "TransactionDT": [86400, 0, 0, 86401, 2, 3, 4],
                "TransactionAmt": [30, 10, 20, 40, 0, None, 10],
                "card1": [1, 1, 1, 1, 2, 2, None],
                "DeviceInfo": ["b", "a", "b", None, None, "c", "z"],
                "isFraud": [0, 0, 1, 0, 0, None, 0],
            }
        )

        result = build_features(transactions)

        self.assertEqual(
            list(result.columns),
            [
                "TransactionID", "card1", "isFraud", "event_timestamp",
                "txn_count_24h", "amount_vs_avg", "device_changed",
            ],
        )
        self.assertEqual(result["TransactionID"].tolist(), [3, 1, 2, 4, 5, 6, 7])
        self.assertEqual(result["txn_count_24h"].tolist(), [0, 0, 1, 1, 0, 1, 0])
        self.assertEqual(result["device_changed"].tolist(), [0, 0, 1, 0, 0, 0, 0])
        self.assertEqual(result["event_timestamp"].iloc[1], START_DATE)
        self.assertEqual(result["amount_vs_avg"].tolist(), [2.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0])
        self.assertEqual(result["card1"].iloc[-1], -1)
        self.assertEqual(result["isFraud"].iloc[5], -1)
        self.assertFalse(result.isna().any().any())

    def test_future_transactions_do_not_change_earlier_features(self):
        transactions = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3, 4, 5],
                "TransactionDT": [0, 0, 86399, 86400, 86401],
                "TransactionAmt": [10, 30, 20, 40, 100],
                "card1": [1, 1, 1, 1, 1],
                "DeviceInfo": ["a"] * 5,
                "isFraud": [0] * 5,
            }
        )
        result = build_features(transactions)
        self.assertEqual(result["txn_count_24h"].tolist(), [0, 1, 2, 1, 2])
        for actual, expected in zip(
            result["amount_vs_avg"], [1, 3, 1, 2, 4]
        ):
            self.assertAlmostEqual(actual, expected)

        updated = transactions.copy()
        updated.loc[4, "TransactionAmt"] = 1000000
        previous = build_features(updated)
        pd.testing.assert_frame_equal(result.iloc[:4], previous.iloc[:4])

    def test_missing_and_zero_amounts_in_history(self):
        transactions = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3, 4],
                "TransactionDT": [1, 2, 3, 4],
                "TransactionAmt": [None, 0, 10, 20],
                "card1": [1, 1, 1, 1],
                "DeviceInfo": ["a"] * 4,
                "isFraud": [0] * 4,
            }
        )
        result = build_features(transactions)
        self.assertEqual(result["amount_vs_avg"].tolist(), [1.0, 0.0, 0.0, 4.0])
        self.assertEqual(result["txn_count_24h"].tolist(), [0, 1, 2, 3])

    def test_all_missing_card_ids_do_not_share_history(self):
        transactions = pd.DataFrame(
            {
                "TransactionID": [1, 2],
                "TransactionDT": [0, 1],
                "TransactionAmt": [10.0, 20.0],
                "card1": pd.Series([None, None], dtype="int64[pyarrow]"),
                "DeviceInfo": ["a", "b"],
                "isFraud": [0, 1],
            }
        )
        result = build_features(transactions)
        self.assertEqual(result["txn_count_24h"].tolist(), [0, 0])
        self.assertEqual(result["amount_vs_avg"].tolist(), [1.0, 1.0])

    def test_empty_input(self):
        data = pd.DataFrame(
            {
                "TransactionID": pd.Series(dtype="int64"),
                "TransactionDT": pd.Series(dtype="int64"),
                "TransactionAmt": pd.Series(dtype="float64"),
                "card1": pd.Series(dtype="float64"),
                "DeviceInfo": pd.Series(dtype="string"),
                "isFraud": pd.Series(dtype="int64"),
            }
        )
        self.assertTrue(build_features(data).empty)


if __name__ == "__main__":
    unittest.main()
