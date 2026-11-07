"""Evaluate a weighted XGBoost and account-GNN blend on validation transactions.

Run from the project root: .\\.venv\\Scripts\\python.exe src\\blend.py
Only load the joblib model from a trusted source.

GNN scores are transductive and include scores for GNN training accounts.
They are not calibrated probabilities or independent account-level predictions.
Percentiles are computed over validation transactions, with average ranks for
ties; missing GNN scores remain zero and are excluded from GNN ranking.
"""

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

if __package__:
    from . import train_xgboost
else:
    import train_xgboost


GNN_SCORES_PATH = train_xgboost.ROOT / "data" / "processed" / "gnn_scores.parquet"
GNN_WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.3)
DEFAULT_GNN_WEIGHT = 0.0


def join_gnn_scores(validation: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Attach one score per known card without changing transaction order."""
    if "card1" not in validation:
        raise ValueError("Validation data is missing card1")
    if not {"card1", "gnn_score"}.issubset(scores.columns):
        raise ValueError("GNN scores must contain card1 and gnn_score")
    if scores["card1"].isna().any() or scores["card1"].eq(-1).any():
        raise ValueError("GNN scores must have known card1 values")
    if scores["card1"].duplicated().any():
        raise ValueError("GNN scores must have unique card1 values")
    numeric_scores = pd.to_numeric(scores["gnn_score"], errors="raise")
    if numeric_scores.isna().any() or not np.isfinite(numeric_scores).all():
        raise ValueError("GNN scores must be finite and non-missing")
    if not numeric_scores.between(0, 1).all():
        raise ValueError("GNN scores must be between 0 and 1")
    score_table = scores[["card1"]].copy()
    score_table["gnn_score"] = numeric_scores
    result = validation.merge(
        score_table, on="card1", how="left", sort=False,
        validate="many_to_one", indicator=True,
    )
    result["has_gnn_score"] = result.pop("_merge").eq("both")
    result["gnn_score"] = result["gnn_score"].fillna(0.0)
    return result


def pr_auc(labels: pd.Series, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def percentile_scores(xgb_score: np.ndarray, scored: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Rank validation transactions, leaving unmatched GNN scores at zero."""
    xgb_rank = pd.Series(xgb_score).rank(method="average", pct=True).to_numpy()
    gnn_rank = scored["gnn_score"].where(scored["has_gnn_score"]).rank(
        method="average", pct=True,
    ).fillna(0.0).to_numpy()
    return xgb_rank, gnn_rank


def sweep_weights(
    labels: pd.Series, xgb_rank: np.ndarray, gnn_rank: np.ndarray,
) -> dict[float, float]:
    return {
        weight: pr_auc(labels, (1.0 - weight) * xgb_rank + weight * gnn_rank)
        for weight in GNN_WEIGHTS
    }


def main() -> None:
    features = pd.read_parquet(train_xgboost.FEATURES_PATH)
    merged = pd.read_parquet(
        train_xgboost.MERGED_PATH, columns=["TransactionID"] + train_xgboost.RAW_COLUMNS,
    )
    train, validation = train_xgboost.split_by_time(
        train_xgboost.join_features(features, merged)
    )
    _, validation_x = train_xgboost.prepare_features(train, validation)
    model = joblib.load(train_xgboost.MODEL_PATH)
    xgb_score = model.predict_proba(validation_x)[:, 1]
    scored = join_gnn_scores(validation, pd.read_parquet(GNN_SCORES_PATH))
    labels = scored[train_xgboost.TARGET].astype("int64")
    xgb_pr_auc = pr_auc(labels, xgb_score)
    xgb_rank, gnn_rank = percentile_scores(xgb_score, scored)
    results = sweep_weights(labels, xgb_rank, gnn_rank)
    best_weight = max(results, key=results.get)
    best_pr_auc = results[best_weight]
    default_pr_auc = results[DEFAULT_GNN_WEIGHT]
    coverage = float(scored["has_gnn_score"].mean())

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run():
        mlflow.log_params({
            "xgboost_weight": 1.0 - DEFAULT_GNN_WEIGHT,
            "gnn_weight": DEFAULT_GNN_WEIGHT,
            "best_gnn_weight": best_weight,
        })
        mlflow.log_metrics({
            "xgboost_pr_auc": xgb_pr_auc,
            "blend_pr_auc": default_pr_auc,
            "best_blend_pr_auc": best_pr_auc,
            "gnn_score_coverage": coverage,
            **{
                f"pr_auc_gnn_weight_{weight:.2f}".replace(".", "_"): value
                for weight, value in results.items()
            },
        })

    print(f"XGBoost validation PR-AUC: {xgb_pr_auc:.6f}")
    print(f"Validation transactions with a GNN score: {coverage:.2%}")
    print("GNN weight | XGBoost weight | PR-AUC")
    for weight, value in results.items():
        print(f"    {weight:.2f}   |      {1.0 - weight:.2f}      | {value:.6f}")
    print(f"Best GNN weight: {best_weight:.2f} (PR-AUC: {best_pr_auc:.6f})")
    if best_weight != DEFAULT_GNN_WEIGHT:
        print(
            f"Configured default GNN weight is {DEFAULT_GNN_WEIGHT:.2f}; "
            "update it if the new sweep should become the default."
        )
    difference = best_pr_auc - xgb_pr_auc
    if difference > 0:
        comparison = "improved"
    elif difference < 0:
        comparison = "decreased"
    else:
        comparison = "did not change"
    print(
        f"The best blend {comparison} PR-AUC by {abs(difference):.6f}. "
        "Higher PR-AUC means a better precision-recall tradeoff across thresholds."
    )
    print(
        "The weight was chosen on the validation set, so this result is slightly "
        "optimistic; evaluate the chosen weight on a separate test set."
    )
    print(
        "Caution: GNN training-account scores are in-sample and GNN scores are not "
        "calibrated probabilities; the blend is not a fully independent evaluation."
    )


if __name__ == "__main__":
    main()
