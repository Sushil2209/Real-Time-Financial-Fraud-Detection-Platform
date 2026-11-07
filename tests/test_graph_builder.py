import unittest

import pandas as pd
import torch

from src.graph_builder import build_graph


def make_data(rows):
    features = pd.DataFrame(
        [
            {
                "TransactionID": row[0],
                "event_timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=row[1]),
                "txn_count_24h": row[2],
                "amount_vs_avg": row[3],
                "device_changed": row[4],
            }
            for row in rows
        ]
    )
    merged = pd.DataFrame(
        [
            {
                "TransactionID": row[0],
                "card1": row[5],
                "isFraud": row[6],
                "TransactionAmt": row[7],
                "DeviceInfo": row[8],
                "P_emaildomain": row[9],
            }
            for row in rows
        ]
    )
    return features, merged


class GraphBuilderTests(unittest.TestCase):
    def test_features_labels_edges_and_missing_accounts(self):
        features, merged = make_data([
            (3, 3, 4, 3, 1, 10, 1, 30, "phone", "email"),
            (1, 1, 0, 1, 0, 10, 0, 10, "phone", "email"),
            (2, 2, 2, 2, 0, 20, 0, 20, "phone", "email"),
            (4, 4, 0, 1, 0, 30, 0, 40, None, None),
            (5, 5, 0, 1, 0, None, 1, 50, "phone", "email"),
        ])
        graph, mapping, excluded = build_graph(features, merged)
        self.assertEqual(mapping, {10: 0, 20: 1, 30: 2})
        self.assertEqual(excluded, 1)
        torch.testing.assert_close(graph.x, torch.tensor([
            [2, 2, 0.5, 2, 20], [2, 2, 0, 1, 20], [0, 1, 0, 1, 40],
        ], dtype=torch.float))
        self.assertEqual(graph.y.tolist(), [1, 0, 0])
        self.assertEqual(graph.edge_index.tolist(), [[0, 1], [1, 0]])

    def test_group_cap_and_empty_edges(self):
        rows = [(i, i, 0, 1, 0, i, 0, 1, "shared", None) for i in range(21)]
        graph, _, _ = build_graph(*make_data(rows))
        self.assertEqual(graph.num_edges, 20 * 19)
        self.assertNotIn(20, graph.edge_index.flatten().tolist())
        graph, _, _ = build_graph(*make_data([rows[0]]))
        self.assertEqual(tuple(graph.edge_index.shape), (2, 0))

    def test_arrow_backed_columns(self):
        features, merged = make_data([(1, 1, 0, 1, 0, 1, 1, 10, None, None)])
        graph, _, _ = build_graph(
            features.convert_dtypes(dtype_backend="pyarrow"),
            merged.convert_dtypes(dtype_backend="pyarrow"),
        )
        self.assertEqual(graph.x.tolist(), [[0, 1, 0, 1, 10]])
        self.assertEqual(graph.y.tolist(), [1])

    def test_chronological_window_before_account_filtering(self):
        rows = [(i, 50_000 - i, 0, 1, 0, i, 0, 1, None, None) for i in range(50_001)]
        graph, mapping, _ = build_graph(*make_data(rows))
        self.assertEqual(graph.num_nodes, 50_000)
        self.assertNotIn(0, mapping)
        self.assertEqual(mapping[50_000], 0)

    def test_rejects_bad_join_labels_and_numeric_values(self):
        features, merged = make_data([(1, 1, 0, 1, 0, 1, 0, 1, None, None)])
        with self.assertRaisesRegex(ValueError, "matching TransactionID"):
            build_graph(features, merged.assign(TransactionID=2))
        with self.assertRaisesRegex(ValueError, "binary isFraud"):
            build_graph(features, merged.assign(isFraud=-1))
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            build_graph(features, merged.assign(TransactionAmt=float("nan")))


if __name__ == "__main__":
    unittest.main()
