import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import blend, train_xgboost


class BlendTests(unittest.TestCase):
    def test_join_preserves_order_and_counts_zero_scores_as_covered(self):
        validation = pd.DataFrame({"card1": [2, 1, 2, -1, None], "isFraud": [0, 1, 1, 0, 0]})
        scores = pd.DataFrame({"card1": [1, 2], "gnn_score": [0.0, 0.8]})
        result = blend.join_gnn_scores(validation, scores)
        self.assertEqual(result["isFraud"].tolist(), validation["isFraud"].tolist())
        self.assertEqual(result["gnn_score"].tolist(), [0.8, 0.0, 0.8, 0.0, 0.0])
        self.assertEqual(result["has_gnn_score"].tolist(), [True, True, True, False, False])

    def test_join_rejects_bad_scores_and_keys(self):
        validation = pd.DataFrame({"card1": [1]})
        for scores in (
            pd.DataFrame({"card1": [1, 1], "gnn_score": [0.1, 0.2]}),
            pd.DataFrame({"card1": [None], "gnn_score": [0.1]}),
            pd.DataFrame({"card1": [-1], "gnn_score": [0.1]}),
            pd.DataFrame({"card1": [1], "gnn_score": [np.nan]}),
            pd.DataFrame({"card1": [1], "gnn_score": [np.inf]}),
            pd.DataFrame({"card1": [1], "gnn_score": [1.1]}),
        ):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                blend.join_gnn_scores(validation, scores)

    def test_percentiles_handle_ties_missing_scores_and_zero_coverage(self):
        scored = pd.DataFrame({
            "gnn_score": [0.0, 0.8, 0.8, 0.0],
            "has_gnn_score": [True, True, True, False],
        })
        xgb_rank, gnn_rank = blend.percentile_scores(np.array([0.3, 0.1, 0.1, 0.9]), scored)
        np.testing.assert_allclose(xgb_rank, [0.75, 0.375, 0.375, 1.0])
        np.testing.assert_allclose(gnn_rank, [1 / 3, 5 / 6, 5 / 6, 0.0])
        scored["has_gnn_score"] = False
        _, gnn_rank = blend.percentile_scores(np.array([0.3, 0.1, 0.1, 0.9]), scored)
        np.testing.assert_array_equal(gnn_rank, [0.0] * 4)

    def test_sweep_uses_ranked_scores_and_complementary_weights(self):
        labels = pd.Series([1, 0, 1, 0])
        xgb_rank = np.array([0.25, 0.5, 0.75, 1.0])
        gnn_rank = np.array([1.0, 0.25, 0.75, 0.5])
        results = blend.sweep_weights(labels, xgb_rank, gnn_rank)
        self.assertEqual(list(results), [0.0, 0.05, 0.1, 0.2, 0.3])
        for weight, result in results.items():
            expected = blend.pr_auc(labels, (1 - weight) * xgb_rank + weight * gnn_rank)
            self.assertAlmostEqual(result, expected)
        self.assertGreater(results[0.3], results[0.0])

    def test_main_uses_training_preparation_and_logs_computed_metrics(self):
        features = pd.DataFrame({
            "TransactionID": list(range(10)),
            "card1": [1] * 8 + [2, 3],
            "isFraud": [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            "event_timestamp": pd.date_range("2020-01-01", periods=10),
            "txn_count_24h": [0] * 10,
            "amount_vs_avg": [1.0] * 10,
            "device_changed": [0] * 10,
        })
        raw = pd.DataFrame({"TransactionID": list(range(10))})
        for column in train_xgboost.RAW_COLUMNS:
            raw[column] = "known" if column in train_xgboost.TEXT_COLUMNS else 1.0
        raw.loc[8, "ProductCD"] = "unseen"
        scores = pd.DataFrame({"card1": [2], "gnn_score": [0.0]})

        class Model:
            def predict_proba(self, validation_x):
                self_test.assertEqual(list(validation_x.columns), train_xgboost.FEATURE_COLUMNS)
                self_test.assertEqual(len(validation_x.columns), 41)
                self_test.assertEqual(validation_x.index.tolist(), [8, 9])
                self_test.assertTrue(pd.isna(validation_x.loc[8, "ProductCD"]))
                return np.array([[0.2, 0.8], [0.8, 0.2]])

        self_test = self
        with (
            patch.object(blend.pd, "read_parquet", side_effect=[features, raw, scores]),
            patch.object(blend.joblib, "load", return_value=Model()),
            patch.object(blend.mlflow, "set_tracking_uri"),
            patch.object(blend.mlflow, "set_experiment") as experiment,
            patch.object(blend.mlflow, "start_run"),
            patch.object(blend.mlflow, "log_params") as params,
            patch.object(blend.mlflow, "log_metrics") as metrics,
        ):
            blend.main()
        experiment.assert_called_once_with("fraud-detection")
        params.assert_called_once_with({
            "xgboost_weight": 1 - blend.DEFAULT_GNN_WEIGHT,
            "gnn_weight": blend.DEFAULT_GNN_WEIGHT,
            "best_gnn_weight": 0.0,
        })
        metrics.assert_called_once_with({
            "xgboost_pr_auc": blend.pr_auc(pd.Series([1, 0]), np.array([0.8, 0.2])),
            "blend_pr_auc": 1.0,
            "best_blend_pr_auc": 1.0,
            "gnn_score_coverage": 0.5,
            "pr_auc_gnn_weight_0_00": 1.0,
            "pr_auc_gnn_weight_0_05": 1.0,
            "pr_auc_gnn_weight_0_10": 1.0,
            "pr_auc_gnn_weight_0_20": 1.0,
            "pr_auc_gnn_weight_0_30": 1.0,
        })


if __name__ == "__main__":
    unittest.main()
