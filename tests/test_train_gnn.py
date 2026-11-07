import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src import train_gnn


def sample_graph():
    return Data(
        x=torch.tensor([[i, i % 3, 1] for i in range(20)], dtype=torch.float),
        edge_index=torch.tensor([[i for i in range(19)], [i + 1 for i in range(19)]]),
        y=torch.tensor([0] * 16 + [1] * 4),
    )


class TrainGNNTests(unittest.TestCase):
    def test_stratified_split_is_disjoint_and_reproducible(self):
        graph = sample_graph()
        train, test = train_gnn.split_nodes(graph)
        self.assertEqual((len(train), len(test)), (14, 6))
        self.assertEqual(len(set(train.tolist()) & set(test.tolist())), 0)
        self.assertEqual(set(train.tolist()) | set(test.tolist()), set(range(20)))
        self.assertEqual(int(graph.y[train].sum()), 3)
        self.assertEqual(int(graph.y[test].sum()), 1)
        self.assertTrue(torch.equal(train, train_gnn.split_nodes(graph)[0]))

    def test_invalid_labels_and_mapping_are_rejected(self):
        graph = sample_graph()
        graph.y[0] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            train_gnn.split_nodes(graph)
        with tempfile.TemporaryDirectory() as directory:
            mapping = Path(directory) / "mapping.json"
            mapping.write_text(json.dumps({"10": 0, "20": 0}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique node"):
                train_gnn.cards_in_node_order(mapping, 2)

    def test_training_saves_reloadable_model_scores_and_mlflow_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(sample_graph(), root / "graph.pt")
            (root / "mapping.json").write_text(
                json.dumps({str(1000 + i): i for i in reversed(range(20))}),
                encoding="utf-8",
            )
            script = """
import os
import sys
from pathlib import Path
import mlflow
from src import train_gnn

root = Path(sys.argv[1])
os.chdir(root)
train_gnn.GRAPH_PATH = root / "graph.pt"
train_gnn.MAPPING_PATH = root / "mapping.json"
train_gnn.MODEL_PATH = root / "models" / "fraud_gcn.pt"
train_gnn.SCORES_PATH = root / "scores.parquet"
train_gnn.EPOCHS = 2
train_gnn.main()
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("fraud-detection")
runs = client.search_runs([experiment.experiment_id])
assert len(runs) == 1 and 0 <= runs[0].data.metrics["pr_auc"] <= 1
assert runs[0].data.params["epochs"] == "2"
"""
            result = subprocess.run(
                [sys.executable, "-c", script, str(root)], check=True,
                capture_output=True, text=True, cwd=Path(train_gnn.ROOT),
                env={**os.environ, "PYTHONPATH": str(train_gnn.ROOT)},
            )
            self.assertIn("Test PR-AUC:", result.stdout)
            scores = pd.read_parquet(root / "scores.parquet")
            self.assertEqual(scores.columns.tolist(), ["card1", "gnn_score"])
            self.assertEqual(scores["card1"].tolist(), list(range(1000, 1020)))
            self.assertTrue(scores["gnn_score"].between(0, 1).all())
            checkpoint = torch.load(root / "models" / "fraud_gcn.pt", weights_only=True)
            model = train_gnn.FraudGCN(
                checkpoint["input_channels"], checkpoint["hidden_channels"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            graph = sample_graph()
            with torch.no_grad():
                x = (graph.x - checkpoint["feature_mean"]) / checkpoint["feature_std"]
                restored = torch.sigmoid(model(x, graph.edge_index)).numpy()
            np.testing.assert_allclose(restored, scores["gnn_score"].to_numpy())
            train, test = train_gnn.split_nodes(graph)
            precision, recall, _ = train_gnn.precision_recall_curve(
                graph.y[test].numpy(), restored[test.numpy()],
            )
            self.assertIn(f"{train_gnn.auc(recall, precision):.6f}", result.stdout)
            torch.testing.assert_close(checkpoint["feature_mean"], graph.x[train].mean(dim=0))


if __name__ == "__main__":
    unittest.main()
