import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import retrain_trigger


class RetrainTriggerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config = root / "config" / "retrain.json"
        self.drift = root / "reports" / "drift_summary.json"
        self.metrics = root / "reports" / "xgb_metrics.json"
        self.model = root / "models" / "xgb_model.joblib"
        self.backup = root / "models" / "backup"
        self.log = root / "reports" / "retrain_log.txt"
        for directory in (self.config.parent, self.drift.parent, self.model.parent):
            directory.mkdir(parents=True, exist_ok=True)
        self.config.write_text(json.dumps({
            "drift_share_threshold": 0.3, "min_pr_auc_gain": 0.0,
        }), encoding="utf-8")
        self.metrics.write_text(json.dumps({"pr_auc": 0.5}), encoding="utf-8")
        self.model.write_bytes(b"old model")
        self.paths = patch.multiple(
            retrain_trigger, ROOT=root, CONFIG_PATH=self.config, DRIFT_PATH=self.drift,
            METRICS_PATH=self.metrics, MODEL_PATH=self.model, BACKUP_DIR=self.backup,
            LOG_PATH=self.log, TRAIN_SCRIPT=root / "src" / "train_xgboost.py",
        )
        self.paths.start()
        self.addCleanup(self.paths.stop)

    def set_drift(self, share):
        self.drift.write_text(json.dumps({"share_of_drifted_columns": share}), encoding="utf-8")

    def candidate_training(self, score):
        def train(command, *, cwd, check):
            self.assertTrue(check)
            self.assertNotEqual(Path(command[command.index("--output-model") + 1]), self.model)
            self.assertNotEqual(Path(command[command.index("--output-metrics") + 1]), self.metrics)
            Path(command[command.index("--output-model") + 1]).write_bytes(b"new model")
            Path(command[command.index("--output-metrics") + 1]).write_text(
                json.dumps({"pr_auc": score}), encoding="utf-8"
            )
        return train

    def test_equal_threshold_does_not_train_and_is_logged(self):
        self.set_drift(0.3)
        with patch.object(retrain_trigger.subprocess, "run") as train:
            retrain_trigger.main()
        train.assert_not_called()
        self.assertIn("No retraining needed", self.log.read_text(encoding="utf-8"))
        self.assertEqual(self.model.read_bytes(), b"old model")

    def test_rejected_candidate_keeps_current_files(self):
        self.set_drift(0.31)
        with patch.object(retrain_trigger.subprocess, "run", side_effect=self.candidate_training(0.49)):
            retrain_trigger.main()
        self.assertEqual(self.model.read_bytes(), b"old model")
        self.assertEqual(json.loads(self.metrics.read_text())["pr_auc"], 0.5)
        self.assertFalse(self.backup.exists())
        self.assertIn("Candidate rejected", self.log.read_text(encoding="utf-8"))

    def test_equal_score_promotes_and_updates_baseline_for_next_run(self):
        self.set_drift(0.31)
        with patch.object(retrain_trigger.subprocess, "run", side_effect=self.candidate_training(0.5)):
            retrain_trigger.main()
        self.assertEqual(self.model.read_bytes(), b"new model")
        self.assertEqual(json.loads(self.metrics.read_text())["pr_auc"], 0.5)
        backups = list(self.backup.glob("xgb_model_*.joblib"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"old model")
        with patch.object(retrain_trigger.subprocess, "run", side_effect=self.candidate_training(0.6)):
            retrain_trigger.main()
        with patch.object(retrain_trigger.subprocess, "run", side_effect=self.candidate_training(0.55)):
            retrain_trigger.main()
        self.assertEqual(json.loads(self.metrics.read_text())["pr_auc"], 0.6)
        self.assertEqual(len(list(self.backup.glob("xgb_model_*.joblib"))), 2)

    def test_minimum_gain_is_enforced(self):
        self.set_drift(0.4)
        self.config.write_text(json.dumps({
            "drift_share_threshold": 0.3, "min_pr_auc_gain": 0.1,
        }), encoding="utf-8")
        with patch.object(retrain_trigger.subprocess, "run", side_effect=self.candidate_training(0.59)):
            retrain_trigger.main()
        self.assertEqual(self.model.read_bytes(), b"old model")
        with patch.object(retrain_trigger.subprocess, "run", side_effect=self.candidate_training(0.6)):
            retrain_trigger.main()
        self.assertEqual(self.model.read_bytes(), b"new model")

    def test_training_failure_is_logged_and_preserves_current_files(self):
        self.set_drift(0.4)
        with patch.object(
            retrain_trigger.subprocess, "run",
            side_effect=subprocess.CalledProcessError(1, "training"),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                retrain_trigger.main()
        self.assertEqual(self.model.read_bytes(), b"old model")
        self.assertEqual(json.loads(self.metrics.read_text())["pr_auc"], 0.5)
        self.assertIn("Retraining failed", self.log.read_text(encoding="utf-8"))

    def test_runs_candidate_training_as_a_real_subprocess(self):
        self.set_drift(0.4)
        script = retrain_trigger.TRAIN_SCRIPT
        script.parent.mkdir()
        script.write_text(
            "import argparse, json\n"
            "from pathlib import Path\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--output-model', required=True)\n"
            "p.add_argument('--output-metrics', required=True)\n"
            "args = p.parse_args()\n"
            "Path(args.output_model).write_bytes(b'subprocess model')\n"
            "Path(args.output_metrics).write_text(json.dumps({'pr_auc': 0.7}))\n",
            encoding="utf-8",
        )
        retrain_trigger.main()
        self.assertEqual(self.model.read_bytes(), b"subprocess model")
        self.assertEqual(json.loads(self.metrics.read_text())["pr_auc"], 0.7)
        self.assertEqual(next(self.backup.iterdir()).read_bytes(), b"old model")

    def test_invalid_share_is_logged(self):
        self.set_drift(float("nan"))
        with patch.object(retrain_trigger.subprocess, "run") as train:
            with self.assertRaises(ValueError):
                retrain_trigger.main()
        train.assert_not_called()
        self.assertIn("Retraining failed", self.log.read_text(encoding="utf-8"))

    def test_metrics_promotion_failure_restores_old_model(self):
        self.set_drift(0.4)
        real_replace = retrain_trigger.os.replace

        def replace(source, destination):
            if destination == self.metrics:
                raise OSError("metrics write failed")
            real_replace(source, destination)

        with patch.object(retrain_trigger.subprocess, "run", side_effect=self.candidate_training(0.6)), \
             patch.object(retrain_trigger.os, "replace", side_effect=replace):
            with self.assertRaisesRegex(OSError, "metrics write failed"):
                retrain_trigger.main()
        self.assertEqual(self.model.read_bytes(), b"old model")
        self.assertEqual(json.loads(self.metrics.read_text())["pr_auc"], 0.5)
        self.assertEqual(len(list(self.backup.glob("xgb_model_*.joblib"))), 1)


if __name__ == "__main__":
    unittest.main()
