"""Build an account graph from the earliest 50,000 transactions.

Each shared device or email-domain group uses at most its first 20 distinct
accounts in timestamp order. Missing account IDs are excluded; missing device
or email values do not create links. Graph features are, in order, the means of
txn_count_24h, amount_vs_avg, and device_changed, the transaction count, and
the mean TransactionAmt. Each undirected edge is stored in both directions.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FEATURE_COLUMNS = ["txn_count_24h", "amount_vs_avg", "device_changed"]
RAW_COLUMNS = ["card1", "isFraud", "TransactionAmt", "DeviceInfo", "P_emaildomain"]
TRANSACTION_LIMIT = 50_000
GROUP_LIMIT = 20


def build_graph(features: pd.DataFrame, merged: pd.DataFrame) -> tuple[Data, dict[int, int], int]:
    """Return the graph, card-to-index mapping, and excluded transaction count."""
    for name, frame, required in (
        ("features", features, ["TransactionID", "event_timestamp"] + FEATURE_COLUMNS),
        ("merged", merged, ["TransactionID"] + RAW_COLUMNS),
    ):
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} data is missing columns: {sorted(missing)}")
        if frame["TransactionID"].isna().any() or frame["TransactionID"].duplicated().any():
            raise ValueError(f"{name} data must have unique, non-missing TransactionID values")
    if features["event_timestamp"].isna().any():
        raise ValueError("event_timestamp must not contain missing values")

    ordered_features = features[["TransactionID", "event_timestamp"] + FEATURE_COLUMNS].copy()
    ordered_features["_feature_order"] = np.arange(len(ordered_features))
    transactions = ordered_features.merge(
        merged[["TransactionID"] + RAW_COLUMNS],
        on="TransactionID", how="outer", sort=False, validate="one_to_one", indicator=True,
    )
    if not transactions["_merge"].eq("both").all():
        raise ValueError("features and merged data must have matching TransactionID values")
    transactions = transactions.sort_values(
        ["event_timestamp", "_feature_order"], kind="stable",
    ).head(TRANSACTION_LIMIT)
    if not transactions["isFraud"].isin([0, 1]).all():
        raise ValueError("Selected transactions must have binary isFraud labels")
    numeric = transactions[FEATURE_COLUMNS + ["TransactionAmt"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Selected transactions must have finite numeric feature and amount values")

    known_card = transactions["card1"].notna() & transactions["card1"].ne(-1)
    excluded = int((~known_card).sum())
    transactions = transactions.loc[known_card].copy()
    if transactions.empty:
        raise ValueError("No transactions with a known card1 account in the selected window")
    transactions["node"] = pd.factorize(transactions["card1"], sort=False)[0]
    cards = transactions.drop_duplicates("node")["card1"].tolist()
    mapping = {int(card): index for index, card in enumerate(cards)}

    grouped = transactions.groupby("node", sort=True)
    node_features = grouped[FEATURE_COLUMNS].mean()
    node_features["transaction_count"] = grouped.size()
    node_features["average_TransactionAmt"] = grouped["TransactionAmt"].mean()
    labels = grouped["isFraud"].max()

    pairs: set[tuple[int, int]] = set()
    for column in ("DeviceInfo", "P_emaildomain"):
        memberships = transactions.dropna(subset=[column]).drop_duplicates([column, "node"])
        for _, group in memberships.groupby(column, sort=False):
            nodes = sorted(group["node"].head(GROUP_LIMIT).tolist())
            pairs.update(combinations(nodes, 2))
    directed_edges = [(a, b) for a, b in sorted(pairs) for a, b in ((a, b), (b, a))]
    edge_index = torch.tensor(directed_edges, dtype=torch.long).reshape(-1, 2).t().contiguous()
    graph = Data(
        x=torch.tensor(node_features.to_numpy(dtype=np.float32), dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels.to_numpy(dtype=np.int64), dtype=torch.long),
    )
    return graph, mapping, excluded


def main() -> None:
    features = pd.read_parquet(
        PROCESSED / "features.parquet",
        columns=["TransactionID", "event_timestamp"] + FEATURE_COLUMNS,
    )
    merged = pd.read_parquet(
        PROCESSED / "merged.parquet", columns=["TransactionID"] + RAW_COLUMNS,
    )
    graph, mapping, excluded = build_graph(features, merged)
    torch.save(graph, PROCESSED / "graph.pt")
    (PROCESSED / "card1_to_node.json").write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Nodes: {graph.num_nodes}")
    print(f"Edges: {graph.num_edges // 2} undirected ({graph.num_edges} directed entries)")
    print(f"Fraud accounts: {int(graph.y.sum().item())}")
    print(f"Transactions excluded for missing card1: {excluded}")


if __name__ == "__main__":
    main()
