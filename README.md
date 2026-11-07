# Real-Time Financial Fraud Detection Platform

## 1. Overview

Fraudulent card transactions are rare, so a model must find suspicious activity without treating every payment as fraud. This project uses the IEEE-CIS transaction data to train and evaluate fraud models, then demonstrates scoring transactions as they arrive through a local Kafka stream.

The demo combines an XGBoost transaction model, an experimental graph neural network (GNN), a Feast feature store, a FastAPI scoring service, and a Streamlit review dashboard. It is a local development project, not a production payment service.

## 2. Architecture

The EDA notebook joins the IEEE-CIS transaction and identity files. A feature script builds per-card history features; XGBoost trains on a chronological split, while a separate graph pipeline builds account links and evaluates a GNN. Feast serves materialized card features to the API. Kafka carries replayed transactions to a consumer, which sends them to the API and saves scores in SQLite for the dashboard. Evidently compares scored rows with training-period data, and a retraining script can evaluate and promote a candidate model.

See [docs/architecture.md](docs/architecture.md) for the component flow and current local setup.

## 3. Tech stack

- **Kafka** — local transaction topic and replay/consumer demo.
- **XGBoost** — primary transaction model; tracked with **MLflow** using SQLite.
- **PyTorch Geometric** — experimental account-level graph neural network.
- **Feast** — local offline file source and SQLite online store.
- **FastAPI** — transaction scoring API.
- **Evidently AI** — data drift reports and drift summary.
- **Streamlit** — fraud review dashboard with case feedback.
- **Docker / Docker Compose** — Kafka container configuration and API, dashboard, and Feast Dockerfiles.
- **Kubernetes** — a `k8s/` folder exists, but it currently contains no manifests.

## 4. Project structure

```text
.
├── config/          # Drift and retraining thresholds
├── data/
│   ├── raw/         # Downloaded IEEE-CIS CSV files (not included in the repository)
│   └── processed/   # Merged data, features, graph, scores, and SQLite results
├── docker/          # Dockerfiles and Kafka Docker Compose configuration
├── docs/            # Architecture documentation
├── feature_repo/    # Feast feature definitions and local store configuration
├── k8s/             # Empty; Kubernetes manifests are not present
├── models/          # Saved XGBoost and GNN models, plus retraining backups
├── notebooks/       # EDA and data merge notebook
├── reports/         # Model metrics, drift reports, and retraining log
├── scripts/         # PowerShell start/stop scripts
├── src/             # Feature building, model training, API, stream, and monitoring code
└── tests/           # Python unittest suite
```

`data/processed/`, model files, and local MLflow tracking data are generated or local runtime artifacts. The IEEE-CIS dataset is not included; download it separately as described below.

## 5. Setup (Windows, PowerShell)

Run commands from the project root. Python 3.12 is required.

### Create the main environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements file pins CPU-only PyTorch; install that wheel from the PyTorch index before installing the remaining requirements.

### Create the Feast environment

Feast is installed separately from the main application dependencies:

```powershell
py -3.12 -m venv .venv-feast
.\.venv-feast\Scripts\python.exe -m pip install --upgrade pip
.\.venv-feast\Scripts\python.exe -m pip install -r requirements-feast.txt
```

### Download the IEEE-CIS data

The Kaggle account must have access to the IEEE-CIS Fraud Detection competition and the Kaggle CLI must be authenticated. Install the CLI into the project environment if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install kaggle
New-Item -ItemType Directory -Force data\raw | Out-Null
.\.venv\Scripts\kaggle.exe competitions download -c ieee-fraud-detection -p data\raw
Expand-Archive -Path data\raw\ieee-fraud-detection.zip -DestinationPath data\raw -Force
```

The notebook expects these files:

```text
data\raw\train_transaction.csv
data\raw\train_identity.csv
```

## 6. How to run

Run each command from the project root unless the step says otherwise. Do the data and model preparation steps before starting the live demo.

### 1. Run EDA and create the merged dataset

The notebook reads both raw CSV files, explores class balance and missing values, then saves `data\processed\merged.parquet`. Jupyter, Matplotlib, and Seaborn are imported by the notebook but are not listed in `requirements.txt`; install them into `.venv` if they are missing.

```powershell
.\.venv\Scripts\python.exe -m pip install jupyter matplotlib seaborn
.\.venv\Scripts\python.exe -m jupyter lab notebooks\01_eda.ipynb
```

Run the notebook cells in order and save the notebook after the final cell.

### 2. Build transaction features

```powershell
.\.venv\Scripts\python.exe src\build_features.py
```

This reads `merged.parquet` and writes `data\processed\features.parquet`.

### 3. Apply and materialize Feast features

Run Feast commands from the feature repository so its local registry and SQLite online store use the configured paths:

```powershell
Push-Location feature_repo
..\.venv-feast\Scripts\feast.exe apply
..\.venv-feast\Scripts\feast.exe materialize 2017-12-01T00:00:00 2019-01-01T00:00:00
Pop-Location
```

### 4. Train XGBoost with MLflow tracking

```powershell
.\.venv\Scripts\python.exe src\train_xgboost.py
```

The script uses a chronological 80/20 split, saves the model under `models\`, writes metrics to `reports\xgb_metrics.json`, and logs the run to the local `sqlite:///mlflow.db` tracking store.

### 5. Build the account graph and train the GNN

```powershell
.\.venv\Scripts\python.exe src\graph_builder.py
.\.venv\Scripts\python.exe src\train_gnn.py
```

The graph builder uses at most the earliest 50,000 transactions. The GNN training script saves an account-level model and scores under `models\` and `data\processed\`.

### 6. Evaluate the XGBoost/GNN blend

```powershell
.\.venv\Scripts\python.exe src\blend.py
```

The script compares rank-based blends and logs the results to MLflow. The current default GNN weight is `0.0`.

### 7. Start Feast, API, dashboard, and Kafka

Docker Desktop must be running for Kafka. This starts Kafka, creates the `transactions` topic if needed, and opens separate PowerShell windows for Feast, the API, and Streamlit:

```powershell
.\scripts\start_all.ps1 -WithKafka
```

Check the API and request a sample score from another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
.\.venv\Scripts\python.exe src\api_test_request.py
```

The dashboard is available at [http://127.0.0.1:8501](http://127.0.0.1:8501). The Feast feature server listens on port `6566`; the API listens on `8000`.

### 8. Replay transactions through Kafka

Open a separate PowerShell window and start the consumer first:

```powershell
.\.venv\Scripts\python.exe src\consumer.py --max-messages 1000
```

Then, from another window, replay up to 1,000 validation transactions:

```powershell
.\.venv\Scripts\python.exe src\producer.py --count 1000 --delay 0.05
```

The consumer scores messages through the API and saves results to `data\processed\scored.db`. Summarize the saved results with:

```powershell
.\.venv\Scripts\python.exe src\check_scores.py
```

### 9. Generate a drift report and run the retraining trigger

Run after at least one transaction has been scored:

```powershell
.\.venv\Scripts\python.exe src\drift_report.py
.\.venv\Scripts\python.exe src\retrain_trigger.py
```

The drift report writes an HTML report and `reports\drift_summary.json`. The retraining trigger reads the configured drift threshold and PR-AUC gain, trains a candidate, and promotes it only when its PR-AUC meets the configured requirement. `--simulate-drift` is available for testing the report path; it triples `TransactionAmt` and adds 10 to `C1`–`C5` in the current sample:

```powershell
.\.venv\Scripts\python.exe src\drift_report.py --simulate-drift
```

### 10. Stop the local demo

```powershell
.\scripts\stop_all.ps1 -WithKafka
```

This stops processes listening on the Feast, API, and dashboard ports; `-WithKafka` also runs `docker compose down` for the Kafka service.

## 7. Results

| Experiment | Measured result |
|---|---|
| XGBoost, 3 features | PR-AUC **0.058** |
| XGBoost, 41 features | PR-AUC **0.458**; precision **0.19**, recall **0.73** at threshold **0.5** |
| Account-level GNN | PR-AUC **0.463** |
| XGBoost/GNN blend | Did not improve ranking; GNN weight **0.0** was best |
| Real-time replay, 1,000 transactions | Average latency **53 ms**, p95 **78 ms**; **47 of 59** frauds scored medium or high |
| Drift report | **14 of 41** columns drifted (share **0.341**) |

PR-AUC is used because the fraud class is highly imbalanced. The saved XGBoost metrics are in `reports/xgb_metrics.json`; the current drift summary and HTML reports are in `reports/`.

## 8. Limitations and next steps

- XGBoost validation uses a time-based split, but the online Feast store keeps each card's latest value. Replaying historical transactions can therefore use a slightly future feature value.
- The GNN covers only a sample of transactions, and its account-level score did not improve the blend's ranking.
- The retraining trigger currently trains on the same data and split as the existing pipeline; it is not a new-data retraining process.
- The `k8s/` directory is currently empty, so Kubernetes manifests are not available in this repository. The demo runs as local processes because Minikube was not reliable on the 8 GB development laptop.
- The Docker Compose configuration starts Kafka only. The API, Feast service, and dashboard are started as local processes by `scripts/start_all.ps1`.

## 9. Testing

The tests use Python's built-in `unittest` runner and do not require the dataset:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
