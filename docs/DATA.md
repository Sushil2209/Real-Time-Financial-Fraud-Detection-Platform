# Data Guide

## Dataset

The project uses the **IEEE-CIS Fraud Detection** dataset from Kaggle:
<https://www.kaggle.com/c/ieee-fraud-detection/data> (a Kaggle account and accepting the
competition rules are required). Put these files in `data/raw/`:

| File | Content |
|------|---------|
| `train_transaction.csv` | One row per transaction: `TransactionID`, `isFraud`, `TransactionDT`, `TransactionAmt`, `card1`…, `C1`–`C14`, `D1`–`D15`, etc. |
| `train_identity.csv` | Device/network info keyed by `TransactionID` (`DeviceType`, `DeviceInfo`, …) |

The test files are not used (they have no labels).
Fraud is rare: about **3.5%** of transactions are fraud, so models are judged by **PR-AUC**, not accuracy.

## Pipeline

1. `notebooks/01_eda.ipynb` joins the two CSVs on `TransactionID` and writes `data/processed/merged.parquet`.
2. [src/build_features.py](../src/build_features.py) writes `features.parquet`.
3. [src/train_xgboost.py](../src/train_xgboost.py) joins both and trains the model.

## Model features (41)

Defined in [src/model_features.py](../src/model_features.py) (`FEATURE_COLUMNS`):

**3 engineered** (per `card1`, from `build_features.py`):

- `txn_count_24h`: number of the card's earlier transactions in the previous 24 hours
- `amount_vs_avg`: current amount divided by the card's average past amount
- `device_changed`: 1 if `DeviceInfo` differs from the card's previous transaction

**38 raw columns** (`RAW_COLUMNS`):

- `TransactionAmt`, `addr1`, `addr2`
- 6 categorical: `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `DeviceType`
- `C1`–`C14` (14 columns) and `D1`–`D15` (15 columns)

Total: 3 engineered + 3 (amount/addr) + 6 categorical + 14 C + 15 D = 41.

## Time-based split

`TransactionDT` is seconds from a reference point; it is converted to `event_timestamp`
(start date 2017-12-01 + `TransactionDT`). `split_by_time` sorts by `event_timestamp`
(stable, so ties keep file order) and takes the **first 80% for training** and the **last 20% for
validation**. No random shuffling, so the model never trains on the future.
The producer replays the same last-20% validation rows, and the drift report compares them with training data.

## Leakage fix: past transactions only

Computing history features with the current transaction included would leak the target
period into the features. `build_features.py` therefore uses past data only:

- `txn_count_24h`: rolling 24h count **minus the current row**, so it counts earlier transactions only.
- `amount_vs_avg`: expanding mean of the card's amounts **shifted by one row**, so the current amount is not in its own average. A card's first transaction gets 1.0; a missing or zero average gives 0.
- `device_changed` also compares only with the previous transaction.

Rows with the same timestamp are handled in input order. Missing `card1` values do not share a history.
At serving time Feast provides the same features (see [API.md](API.md)). Rebuild `features.parquet`
after any change to this logic.

## data/processed contents

| File | Created by | Contents |
|------|-----------|----------|
| `merged.parquet` | `notebooks/01_eda.ipynb` | Transactions joined with identity data (all raw columns plus `isFraud`, `TransactionDT`, `DeviceInfo`) |
| `features.parquet` | `src/build_features.py` | One row per transaction: `TransactionID`, `card1`, `isFraud`, `event_timestamp`, `txn_count_24h`, `amount_vs_avg`, `device_changed` (missing IDs/labels stored as -1) |
| `graph.pt` | `src/graph_builder.py` | PyTorch Geometric account graph built from the earliest 50,000 transactions (nodes = `card1`; edges link accounts sharing a device or email domain) |
| `card1_to_node.json` | `src/graph_builder.py` | Mapping from `card1` to graph node index |
| `gnn_scores.parquet` | `src/train_gnn.py` | `card1`, `gnn_score` (uncalibrated GCN score per account) |
| `scored.db` | `src/consumer.py`, `src/dashboard.py` | SQLite with `scores` and `feedback` tables (see [API.md](API.md)) |

Only load `graph.pt` from a trusted source (it uses pickle).
