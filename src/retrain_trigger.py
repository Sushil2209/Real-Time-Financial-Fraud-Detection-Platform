"""Retrain on drift and promote only a model meeting the PR-AUC requirement.

Run from the project root: .\\.venv\\Scripts\\python.exe src\\retrain_trigger.py
Candidate files are temporary. On promotion, the old model is backed up and
the current metrics are updated to match the new model.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "retrain.json"
DRIFT_PATH = ROOT / "reports" / "drift_summary.json"
METRICS_PATH = ROOT / "reports" / "xgb_metrics.json"
MODEL_PATH = ROOT / "models" / "xgb_model.joblib"
BACKUP_DIR = ROOT / "models" / "backup"
LOG_PATH = ROOT / "reports" / "retrain_log.txt"
TRAIN_SCRIPT = ROOT / "src" / "train_xgboost.py"


def log(message: str) -> None:
    entry = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    print(entry, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(entry + "\n")


def read_number(path: Path, key: str, minimum: float, maximum: float | None = None) -> float:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    value = data[key]
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{path}: {key} must be a finite number in the allowed range")
    return float(value)


def run() -> None:
    threshold = read_number(CONFIG_PATH, "drift_share_threshold", 0.0, 1.0)
    gain = read_number(CONFIG_PATH, "min_pr_auc_gain", 0.0)
    share = read_number(DRIFT_PATH, "share_of_drifted_columns", 0.0, 1.0)
    if share <= threshold:
        log(f"No retraining needed: drift share {share:.6f} is not above {threshold:.6f}.")
        return

    log(f"Retraining needed: drift share {share:.6f} exceeds {threshold:.6f}.")
    current_score = read_number(METRICS_PATH, "pr_auc", 0.0, 1.0)
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Current model does not exist: {MODEL_PATH}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="retrain_", dir=MODEL_PATH.parent) as directory:
        candidate = Path(directory) / "candidate.joblib"
        candidate_metrics = Path(directory) / "candidate_metrics.json"
        log("Starting candidate training without changing the current model or metrics.")
        subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT), "--output-model", str(candidate),
             "--output-metrics", str(candidate_metrics)],
            cwd=ROOT, check=True,
        )
        new_score = read_number(candidate_metrics, "pr_auc", 0.0, 1.0)
        if not candidate.is_file():
            raise FileNotFoundError(f"Training did not create a candidate model: {candidate}")
        required = current_score + gain
        if new_score < required:
            log(
                f"Candidate rejected: PR-AUC {new_score:.6f} is below required "
                f"{required:.6f} (current {current_score:.6f}, gain {gain:.6f})."
            )
            return

        log(
            f"Candidate accepted: PR-AUC {new_score:.6f} meets required "
            f"{required:.6f} (current {current_score:.6f}, gain {gain:.6f})."
        )
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = BACKUP_DIR / f"xgb_model_{timestamp}.joblib"
        with MODEL_PATH.open("rb") as source, backup.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        log(f"Backed up current model to {backup}.")
        os.replace(candidate, MODEL_PATH)
        try:
            os.replace(candidate_metrics, METRICS_PATH)
        except OSError:
            shutil.copyfile(backup, candidate)
            os.replace(candidate, MODEL_PATH)
            raise
        log(f"Promoted candidate model and updated current metrics to PR-AUC {new_score:.6f}.")


def main() -> None:
    try:
        run()
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        log(f"Retraining failed; inspect the error before retrying: {exc}")
        raise


if __name__ == "__main__":
    main()
