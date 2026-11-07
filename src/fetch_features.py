"""Fetch the latest materialized features for a card."""

import argparse
from importlib import import_module
from pathlib import Path

FeatureStore = import_module("feast").FeatureStore


FEATURE_REPO = Path(__file__).resolve().parent.parent / "feature_repo"
FEATURES = [
    "card_transaction_features:txn_count_24h",
    "card_transaction_features:amount_vs_avg",
    "card_transaction_features:device_changed",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("card1", type=int, help="Card ID to look up")
    args = parser.parse_args()

    store = FeatureStore(repo_path=str(FEATURE_REPO))
    result = store.get_online_features(
        features=FEATURES, entity_rows=[{"card1": args.card1}]
    ).to_dict()
    if any(result[name][0] is None for name in (
        "txn_count_24h", "amount_vs_avg", "device_changed"
    )):
        raise SystemExit(
            f"No complete online features found for card1={args.card1}. "
            "Check the card ID and materialize the feature view."
        )
    print(result)


if __name__ == "__main__":
    main()
