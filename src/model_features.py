from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "xgb_model.joblib"
TEXT_COLUMNS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType"]
RAW_COLUMNS = [
    "TransactionAmt", "ProductCD", "card4", "card6", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "DeviceType",
] + [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]
FEATURE_COLUMNS = ["txn_count_24h", "amount_vs_avg", "device_changed"] + RAW_COLUMNS
