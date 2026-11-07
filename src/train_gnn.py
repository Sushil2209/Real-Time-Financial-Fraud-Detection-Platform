"""Train a two-layer account GCN and save its scores.

Run from the project root after building graph.pt and card1_to_node.json:
    .\\.venv\\Scripts\\python.exe src\\train_gnn.py

The split is transductive: train and test accounts share the same graph, but
only training labels are used for fitting. Scores are not calibrated
probabilities, and training-node scores are not out-of-sample predictions.
Only load graph.pt from a trusted source; torch.load uses pickle for PyG data.
"""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
GRAPH_PATH = PROCESSED / "graph.pt"
MAPPING_PATH = PROCESSED / "card1_to_node.json"
SCORES_PATH = PROCESSED / "gnn_scores.parquet"
MODEL_PATH = ROOT / "models" / "fraud_gcn.pt"
SEED = 42
EPOCHS = 100
HIDDEN_CHANNELS = 32


class FraudGCN(nn.Module):
    """Combine each account's features with those of its neighbors twice."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index).squeeze(-1)


def split_nodes(graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Split node indices while keeping both label proportions similar."""
    if graph.x is None or graph.y is None or graph.edge_index is None:
        raise ValueError("Graph must contain x, y, and edge_index")
    if graph.x.ndim != 2 or graph.x.shape[1] == 0:
        raise ValueError("Graph x must be a non-empty 2D feature matrix")
    if not torch.isfinite(graph.x).all():
        raise ValueError("Graph features must be finite")
    if graph.y.ndim != 1 or len(graph.y) != len(graph.x):
        raise ValueError("Graph must have one label per node")
    if not torch.all((graph.y == 0) | (graph.y == 1)):
        raise ValueError("Graph labels must be binary (0 or 1)")
    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError("Graph edge_index must have shape [2, number of edges]")

    nodes = np.arange(len(graph.y))
    train, test = train_test_split(
        nodes, test_size=0.3, random_state=SEED, stratify=graph.y.numpy(),
    )
    return torch.as_tensor(train, dtype=torch.long), torch.as_tensor(test, dtype=torch.long)


def cards_in_node_order(mapping_path: Path, num_nodes: int) -> list[int]:
    """Invert the graph builder's card-to-node mapping without changing order."""
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or len(mapping) != num_nodes:
        raise ValueError("Card mapping must contain exactly one entry per graph node")
    cards: list[int | None] = [None] * num_nodes
    for card, node in mapping.items():
        if type(node) is not int or not 0 <= node < num_nodes or cards[node] is not None:
            raise ValueError("Card mapping must contain unique node indices in graph range")
        cards[node] = int(card)
    if len(set(cards)) != num_nodes:
        raise ValueError("Card mapping must contain unique card1 values")
    return [card for card in cards if card is not None]


def main() -> None:
    torch.manual_seed(SEED)
    graph = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    train_nodes, test_nodes = split_nodes(graph)
    cards = cards_in_node_order(MAPPING_PATH, len(graph.y))

    # Learn feature scaling and class weight from training nodes only.
    x = graph.x.float()
    mean = x[train_nodes].mean(dim=0)
    std = x[train_nodes].std(dim=0, unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)
    x = (x - mean) / std
    labels = graph.y.float()
    positives = labels[train_nodes].sum()
    negatives = len(train_nodes) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Training nodes must contain both fraud and non-fraud labels")
    pos_weight = negatives / positives

    model = FraudGCN(x.shape[1], HIDDEN_CHANNELS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run():
        mlflow.log_params({
            "model": "GCN", "epochs": EPOCHS, "hidden_channels": HIDDEN_CHANNELS,
            "learning_rate": 0.01, "weight_decay": 5e-4, "seed": SEED,
            "test_size": 0.3, "pos_weight": float(pos_weight),
        })
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            logits = model(x, graph.edge_index)
            loss = loss_fn(logits[train_nodes], labels[train_nodes])
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS}: loss={loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(x, graph.edge_index)).numpy()
        precision, recall, _ = precision_recall_curve(
            labels[test_nodes].numpy(), scores[test_nodes.numpy()],
        )
        pr_auc = float(auc(recall, precision))
        mlflow.log_metric("pr_auc", pr_auc)
        print(f"Test PR-AUC: {pr_auc:.6f}")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_channels": x.shape[1],
            "hidden_channels": HIDDEN_CHANNELS,
            "feature_mean": mean,
            "feature_std": std,
        }, MODEL_PATH)
        SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"card1": cards, "gnn_score": scores}).to_parquet(
            SCORES_PATH, index=False,
        )


if __name__ == "__main__":
    main()
