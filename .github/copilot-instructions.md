# Project context
Real-Time Financial Fraud Detection Platform (AIML-01), fresher-level version.
- Windows, PowerShell, VS Code. Everything runs locally, no cloud.
- Python 3.12, virtual environment .venv in C:\fraud-platform.
- Dataset: IEEE-CIS (data/raw/train_transaction.csv, train_identity.csv).
- Stack: XGBoost, MLflow (sqlite:///mlflow.db), Feast, PyTorch Geometric,
  Kafka, FastAPI, Evidently AI, Streamlit, Kubernetes (Minikube).
# Rules
- I am a fresher: explain simply, keep code short and commented.
- One small change at a time, and tell me how to test it.
- Use PowerShell commands.
- Evaluate with PR-AUC, not accuracy, because fraud is under 1% of the data.
- Always use a virtual environment and install dependencies from requirements.txt.