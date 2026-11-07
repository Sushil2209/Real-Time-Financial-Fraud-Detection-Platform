import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src import train_xgboost


def sample_data():
    return pd.DataFrame(
        {
            "TransactionID": list(range(10)),
            "event_timestamp": pd.to_datetime(
                ["2020-01-05", "2020-01-01", "2020-01-10", "2020-01-02",
                 "2020-01-09", "2020-01-03", "2020-01-08", "2020-01-04",
                 "2020-01-07", "2020-01-06"]
            ),
            "isFraud": [0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
            "txn_count_24h": list(range(10)),
            "amount_vs_avg": [1.0] * 10,
            "device_changed": [0, 1] * 5,
        }
    )


def sample_raw():
    data = {"TransactionID": list(range(9, -1, -1))}
    for column in train_xgboost.RAW_COLUMNS:
        if column in train_xgboost.TEXT_COLUMNS:
            data[column] = ["known"] * 10
        else:
            data[column] = [float(i) for i in range(9, -1, -1)]
    data["ProductCD"][7] = "unseen"
    data["card4"][5] = None
    data["D1"][5] = None
    return pd.DataFrame(data)


class TrainXGBoostTests(unittest.TestCase):
    def test_split_sorts_by_time_and_uses_first_eighty_percent(self):
        data = train_xgboost.join_features(sample_data(), sample_raw())
        train, validation = train_xgboost.split_by_time(data)
        self.assertEqual((len(train), len(validation)), (8, 2))
        self.assertTrue(train["event_timestamp"].is_monotonic_increasing)
        self.assertLessEqual(train["event_timestamp"].max(), validation["event_timestamp"].min())

    def test_invalid_labels_and_single_class_training_fail(self):
        data = train_xgboost.join_features(sample_data(), sample_raw())
        data.loc[0, "isFraud"] = -1
        with self.assertRaisesRegex(ValueError, "binary labels"):
            train_xgboost.split_by_time(data)
        data["isFraud"] = 0
        with self.assertRaisesRegex(ValueError, "both fraud and non-fraud"):
            train_xgboost.split_by_time(data)

    def test_join_and_feature_preparation(self):
        data = train_xgboost.join_features(sample_data(), sample_raw())
        self.assertEqual(data["TransactionID"].tolist(), list(range(10)))
        self.assertEqual(data.loc[0, "TransactionAmt"], 0.0)
        train, validation = train_xgboost.split_by_time(data)
        train_x, validation_x = train_xgboost.prepare_features(train, validation)
        self.assertEqual(len(train_x.columns), 41)
        self.assertNotIn("isFraud", train_x.columns)
        self.assertNotIn("TransactionID", train_x.columns)
        for column in train_xgboost.TEXT_COLUMNS:
            self.assertIsInstance(train_x[column].dtype, pd.CategoricalDtype)
            self.assertEqual(train_x[column].dtype, validation_x[column].dtype)
        self.assertTrue(pd.isna(validation_x.loc[2, "ProductCD"]))
        self.assertTrue(pd.isna(validation_x.loc[4, "card4"]))
        self.assertTrue(pd.isna(validation_x.loc[4, "D1"]))

    def test_join_rejects_missing_or_duplicate_keys_and_missing_columns(self):
        features, raw = sample_data(), sample_raw()
        with self.assertRaisesRegex(ValueError, "no matching TransactionID"):
            train_xgboost.join_features(features, raw.iloc[:-1])
        with self.assertRaisesRegex(ValueError, "unique, non-missing"):
            train_xgboost.join_features(features, pd.concat([raw, raw.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, "Missing raw columns"):
            train_xgboost.join_features(features, raw.drop(columns="D15"))

    def test_command_line_rejects_nonpositive_options(self):
        script = Path(train_xgboost.__file__)
        for option in ("--max-depth", "--n-estimators"):
            result = subprocess.run(
                [sys.executable, str(script), option, "0"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be a positive integer", result.stderr)

    def test_training_saves_and_logs_model_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "features.parquet"
            merged_path = root / "merged.parquet"
            model_path = root / "models" / "xgb_model.joblib"
            metrics_path = root / "reports" / "xgb_metrics.json"
            sample_data().to_parquet(data_path, index=False)
            sample_raw().to_parquet(merged_path, index=False)
            script = """
import json
import os
import sys
from pathlib import Path
import mlflow
import mlflow.xgboost
from src import train_xgboost

root = Path(sys.argv[1])
os.chdir(root)
sys.argv = [sys.argv[0], "--max-depth", "4", "--n-estimators", "12"]
train_xgboost.FEATURES_PATH = root / "features.parquet"
train_xgboost.MERGED_PATH = root / "merged.parquet"
train_xgboost.MODEL_PATH = root / "models" / "xgb_model.joblib"
train_xgboost.METRICS_PATH = root / "reports" / "xgb_metrics.json"
train_xgboost.main()
original_model = train_xgboost.MODEL_PATH.read_bytes()
original_metrics = train_xgboost.METRICS_PATH.read_bytes()
candidate = root / "models" / "candidate.joblib"
sys.argv = [sys.argv[0], "--max-depth", "4", "--n-estimators", "12",
            "--output-model", str(candidate)]
train_xgboost.main()
assert candidate.is_file()
assert (candidate.parent / "candidate_metrics.json").is_file()
assert train_xgboost.MODEL_PATH.read_bytes() == original_model
assert train_xgboost.METRICS_PATH.read_bytes() == original_metrics
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("fraud-detection")
runs = client.search_runs([experiment.experiment_id])
assert len(runs) == 2
assert runs[0].data.metrics == json.loads(train_xgboost.METRICS_PATH.read_text())
assert float(runs[0].data.params["scale_pos_weight"]) == 3.0
assert runs[0].data.params["max_depth"] == "4"
assert runs[0].data.params["n_estimators"] == "12"
assert runs[0].data.params["num_features"] == "41"
assert runs[0].data.params["enable_categorical"] == "True"
assert runs[0].data.params["tree_method"] == "hist"
model = mlflow.xgboost.load_model(f"runs:/{runs[0].info.run_id}/model")
features = train_xgboost.pd.read_parquet(train_xgboost.FEATURES_PATH)
raw = train_xgboost.pd.read_parquet(train_xgboost.MERGED_PATH)
train, validation = train_xgboost.split_by_time(train_xgboost.join_features(features, raw))
_, validation_x = train_xgboost.prepare_features(train, validation)
assert model.predict_proba(validation_x).shape == (2, 2)
"""
            subprocess.run([sys.executable, "-c", script, str(root)], check=True)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(set(metrics), {"pr_auc", "precision", "recall", "f1"})
            model = joblib.load(model_path)
            self.assertAlmostEqual(model.get_params()["scale_pos_weight"], 3.0)
            self.assertEqual(model.get_params()["max_depth"], 4)
            self.assertEqual(model.get_params()["n_estimators"], 12)


if __name__ == "__main__":
    unittest.main()
