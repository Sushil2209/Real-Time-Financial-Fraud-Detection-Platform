# Architecture

## System diagram

```mermaid
flowchart TD
    DS[("IEEE-CIS dataset<br/>train_transaction.csv + train_identity.csv")]
    EDA["EDA<br/>notebooks/01_eda.ipynb<br/>-> merged.parquet"]
    FE["Feature engineering<br/>src/build_features.py<br/>-> features.parquet"]
    DS --> EDA --> FE

    subgraph Offline["Offline training"]
        XGB["XGBoost training<br/>src/train_xgboost.py"]
        MLF[("MLflow tracking<br/>sqlite:///mlflow.db")]
        MODEL[["models/xgb_model.joblib"]]
        GG["Graph builder<br/>src/graph_builder.py"]
        GNN["GNN training (PyTorch Geometric)<br/>src/train_gnn.py"]
        BLEND["GNN blend (OPTIONAL, weight 0)<br/>src/blend.py - offline comparison only"]
        XGB --> MLF
        XGB --> MODEL
        GG --> GNN --> MLF
        GNN -.-> BLEND
        XGB -.-> BLEND
    end
    FE --> XGB
    FE --> GG

    subgraph Feast["Feast (feature_repo/)"]
        OFF[("Offline store<br/>features.parquet")]
        ON[("Online store<br/>SQLite")]
        FS["Feast feature server :6566"]
        OFF -->|materialize| ON --> FS
    end
    FE --> OFF

    subgraph RT["Real-time path"]
        PROD["Producer<br/>src/producer.py"]
        KAFKA{{"Kafka topic: transactions"}}
        CONS["Consumer<br/>src/consumer.py"]
        API["FastAPI /score<br/>src/api.py"]
        DB[("SQLite scored.db<br/>scores table")]
        DASH["Streamlit analyst dashboard<br/>src/dashboard.py"]
        FB[("feedback table")]
        PROD --> KAFKA --> CONS --> API
        API -->|"POST response"| CONS
        CONS --> DB --> DASH --> FB
    end
    API <-->|"get-online-features"| FS
    MODEL --> API

    subgraph Mon["Monitoring and retraining"]
        EV["Evidently drift report<br/>src/drift_report.py"]
        REP[/"reports/drift_report.html<br/>reports/drift_summary.json"/]
        RT2["Retrain trigger<br/>src/retrain_trigger.py"]
        EV --> REP --> RT2
    end
    DB --> EV
    FE -->|"reference sample"| EV
    RT2 -->|"re-runs training"| XGB
    RT2 -->|"promote if PR-AUC gain"| MODEL
```

The dotted GNN blend is optional: `src/blend.py` sweeps GNN weights and logs the PR-AUC to MLflow, but the default weight is 0.0 and the API scores with XGBoost only.

## Data flow

```mermaid
flowchart LR
    A["Raw CSVs"] --> B["merged.parquet"] --> C["features.parquet"]
    C --> D["Feast online store"]
    C --> E["xgb_model.joblib"]
    P["Producer"] --> K["Kafka"] --> Co["Consumer"] --> S["/score"]
    D --> S
    E --> S
    S --> R["scored.db"] --> U["Dashboard feedback"]
    R --> V["Drift report"] --> T["Retrain trigger"] --> E
```

## Components

| Component | File | Role |
|---|---|---|
| EDA | `notebooks/01_eda.ipynb` | Joins transaction and identity CSVs into `merged.parquet`. |
| Feature engineering | `src/build_features.py` | Builds card history features into `features.parquet`. |
| Feast | `feature_repo/` | Offline parquet source, SQLite online store, feature server on port 6566. |
| XGBoost training | `src/train_xgboost.py` | Trains the model and logs PR-AUC and the model to MLflow. |
| GNN training | `src/graph_builder.py`, `src/train_gnn.py` | Builds a card-account graph and trains a GCN. |
| GNN blend (optional) | `src/blend.py` | Compares weighted XGBoost + GNN ranks; default GNN weight 0. |
| Producer | `src/producer.py` | Replays validation transactions to the Kafka topic `transactions`. |
| Consumer | `src/consumer.py` | Reads the topic, calls `/score`, writes results to `scored.db`. |
| FastAPI | `src/api.py` | `/score` fetches Feast features and scores with XGBoost. |
| Dashboard | `src/dashboard.py` | Shows scores; analysts save feedback to the `feedback` table. |
| Drift report | `src/drift_report.py` | Evidently compares `scored.db` with a training-period reference sample. |
| Retrain trigger | `src/retrain_trigger.py` | If drift exceeds the threshold in `config/retrain.json`, trains a candidate and promotes it only with enough PR-AUC gain. |

## Data and training

`notebooks/01_eda.ipynb` reads the IEEE-CIS transaction and identity CSV files, joins them on `TransactionID`, and writes `data/processed/merged.parquet`. `src/build_features.py` creates transaction history features and writes `features.parquet`.

`src/train_xgboost.py` combines those features with raw transaction columns, trains on the earlier 80% of the timeline, evaluates on the later 20%, and logs metrics and the model to local MLflow tracking. The separate graph pipeline builds a sampled card-account graph, trains a PyTorch Geometric GCN, and writes account-level scores. `src/blend.py` compares percentile-ranked XGBoost and GNN scores.

## Online scoring and monitoring

The Feast repository uses the feature parquet as an offline file source and SQLite as its online store. Its feature server provides the latest materialized features for a card. FastAPI combines those values with request fields and scores the transaction using the saved XGBoost model. If the Feast lookup is unavailable, the API records that it used default feature values.

The Kafka producer replays validation transactions to the `transactions` topic. The consumer calls the scoring API and stores responses in `data/processed/scored.db`. Streamlit reads the score database and lets reviewers record feedback. Evidently compares scored transactions with a sample from the XGBoost training period; the retraining trigger can train a candidate and promote it when its PR-AUC meets the configured gain.

## Local deployment

`docker/docker-compose.yml` defines the Kafka container. The API, Feast feature server, and Streamlit dashboard can be started as local processes by `scripts/start_all.ps1`; `scripts/stop_all.ps1` stops the services by their listening ports and can also stop Kafka. Dockerfiles for API, dashboard, and Feast are present. The `k8s/` directory currently contains no Kubernetes manifests.
