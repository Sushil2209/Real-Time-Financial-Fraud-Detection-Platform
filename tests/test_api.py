import asyncio
import os
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

from src import api


class OnlineFeatureTests(unittest.TestCase):
    def setUp(self):
        self.session = Mock()
        self.state_patch = patch.dict(api.state, {"session": self.session})
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)

    def test_maps_response_by_name_and_sends_card(self):
        response = Mock()
        response.json.return_value = {
            "metadata": {
                "feature_names": ["card1", "device_changed", "txn_count_24h", "amount_vs_avg"]
            },
            "results": [
                {"values": [13926]}, {"values": [1]}, {"values": [7]}, {"values": [1.5]}
            ],
        }
        self.session.post.return_value = response
        with (
            patch.object(api.time, "perf_counter", side_effect=[10.0, 10.025]),
            self.assertLogs(api.log, level="INFO") as logs,
        ):
            values, source = api.get_online_features(13926)
        self.assertEqual(values, {"txn_count_24h": 7, "amount_vs_avg": 1.5, "device_changed": 1})
        self.assertEqual(source, "feast")
        self.session.post.assert_called_once_with(
            api.FEAST_URL,
            json={"features": api.FEAST_FEATURES, "entities": {"card1": [13926]}},
            timeout=1,
        )
        response.raise_for_status.assert_called_once()
        self.assertIn("Feast call took 25.0 ms", logs.output[0])

    def test_unavailable_server_and_timeout_use_defaults(self):
        for error in (requests.ConnectionError("down"), requests.Timeout("timeout")):
            self.session.post.side_effect = error
            with self.subTest(error=error), self.assertLogs(api.log, level="INFO") as logs:
                self.assertEqual(api.get_online_features(1), (api.DEFAULTS, "default"))
            self.assertIn("Feast call took", logs.output[0])

    def test_bad_status_uses_defaults(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("server error")
        self.session.post.return_value = response
        self.assertEqual(api.get_online_features(1), (api.DEFAULTS, "default"))

    def test_configured_url_is_used(self):
        with (
            patch.object(api, "FEAST_URL", "http://localhost:9000/features"),
            patch.object(self.session, "post", side_effect=requests.Timeout) as post,
        ):
            api.get_online_features(1)
        self.assertEqual(post.call_args.args[0], "http://localhost:9000/features")

    def test_null_or_malformed_response_uses_defaults(self):
        valid = {
            "metadata": {"feature_names": ["txn_count_24h", "amount_vs_avg", "device_changed"]},
            "results": [{"values": [7]}, {"values": [1.5]}, {"values": [1]}],
        }
        cases = [
            {**valid, "results": [{"values": [7]}, {"values": [None]}, {"values": [1]}]},
            {**valid, "results": [{"values": [7]}]},
            {**valid, "results": [{"values": [7]}, {"values": []}, {"values": [1]}]},
            {**valid, "metadata": {"feature_names": ["txn_count_24h"] * 3}},
            {},
        ]
        for result in cases:
            response = Mock()
            response.json.return_value = result
            self.session.post.return_value = response
            with self.subTest(result=result):
                self.assertEqual(api.get_online_features(1), (api.DEFAULTS, "default"))

    def test_score_returns_feature_source_and_passes_features_to_model(self):
        model = Mock()
        model.predict_proba.return_value = np.array([[0.2, 0.8]])
        txn = api.Transaction(TransactionID=123, card1=456)
        for source in ("feast", "default"):
            with (
                self.subTest(source=source),
                patch.dict(api.state, {"model": model}),
                patch.object(api, "get_online_features", return_value=(api.DEFAULTS, source)),
                patch.object(api, "build_row", return_value="row") as build_row,
                patch.object(api.time, "perf_counter", side_effect=[20.0, 30.0, 30.125, 40.0]),
                self.assertLogs(api.log, level="INFO") as logs,
            ):
                result = api.score(txn)
            self.assertEqual(result["features_source"], source)
            self.assertEqual(result["TransactionID"], 123)
            self.assertEqual(result["risk_level"], "high")
            build_row.assert_called_once_with(txn.model_dump(), api.DEFAULTS)
            model.predict_proba.assert_called_with("row")
            self.assertIn("Model scoring took 125.0 ms", logs.output[0])

    def test_default_url_uses_ipv4(self):
        env = os.environ.copy()
        env.pop("FEAST_URL", None)
        url = subprocess.check_output(
            [sys.executable, "-c", "from src.api import FEAST_URL; print(FEAST_URL)"],
            env=env,
            text=True,
        ).strip()
        self.assertEqual(url, "http://127.0.0.1:6566/get-online-features")

    def test_lifespan_reuses_and_closes_session(self):
        model = Mock()
        model.get_booster.return_value.get_categories.return_value.to_arrow.return_value = []
        session = Mock()
        session_context = Mock()
        session_context.__enter__ = Mock(return_value=session)
        session_context.__exit__ = Mock(return_value=False)

        async def run_lifespan():
            async with api.lifespan(api.app):
                self.assertIs(api.state["session"], session)
                self.assertIs(api.state["session"], session)
            self.assertNotIn("session", api.state)

        with (
            patch.object(api.joblib, "load", return_value=model),
            patch.object(api.requests, "Session", return_value=session_context) as session_factory,
        ):
            asyncio.run(run_lifespan())
        session_factory.assert_called_once_with()
        session_context.__exit__.assert_called_once_with(None, None, None)


if __name__ == "__main__":
    unittest.main()
