"""Build transaction features from the merged dataset.

The 24-hour count excludes the current transaction and transactions exactly
24 hours earlier. Equal timestamps are processed in input order. The amount
average uses only earlier transactions for the card. First transactions have
an amount ratio of 1.0; otherwise missing or zero averages and missing amounts
produce 0. Missing card IDs do not share a history; missing device values are
unknown, not changes. Missing transaction times use the start date. Missing
IDs and labels are represented by -1 in the output.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "processed" / "merged.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "features.parquet"
START_DATE = pd.Timestamp("2017-12-01")


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return one feature row per input transaction, in input order."""
    data = transactions[
        ["TransactionID", "TransactionDT", "TransactionAmt", "card1", "DeviceInfo", "isFraud"]
    ].copy().reset_index(drop=True)
    data["event_timestamp"] = START_DATE + pd.to_timedelta(
        data["TransactionDT"].fillna(0), unit="s"
    )
    data = data.sort_values(["card1", "event_timestamp"], kind="stable")
    known_card = data["card1"].notna()
    data["txn_count_24h"] = 0
    data["_transaction"] = 1
    counts = (
        data.loc[known_card]
        .groupby("card1", sort=False)
        .rolling("24h", on="event_timestamp")["_transaction"]
        .sum()
    )
    data.loc[known_card, "txn_count_24h"] = counts.to_numpy().astype("int64") - 1

    card_groups = data.groupby("card1", sort=False)
    data["amount_vs_avg"] = 1.0
    if known_card.any():
        average = (
            card_groups["TransactionAmt"]
            .expanding()
            .mean()
            .groupby(level=0, sort=False)
            .shift(1)
            .reset_index(level=0, drop=True)
            .replace(0, float("nan"))
        )
        data.loc[known_card, "amount_vs_avg"] = (
            data.loc[known_card, "TransactionAmt"] / average
        ).fillna(0)
        data.loc[card_groups.cumcount().eq(0), "amount_vs_avg"] = 1.0
    previous_device = card_groups["DeviceInfo"].shift()
    data["device_changed"] = (
        data["DeviceInfo"].notna()
        & previous_device.notna()
        & data["DeviceInfo"].ne(previous_device).fillna(False)
    ).astype("int64")

    for column in ("TransactionID", "card1", "isFraud"):
        data[column] = data[column].fillna(-1)

    return data.sort_index()[
        [
            "TransactionID",
            "card1",
            "isFraud",
            "event_timestamp",
            "txn_count_24h",
            "amount_vs_avg",
            "device_changed",
        ]
    ]


def main() -> None:
    transactions = pd.read_parquet(INPUT_PATH)
    features = build_features(transactions)
    features.to_parquet(OUTPUT_PATH, index=False)
    print(features.head().to_string(index=False))


if __name__ == "__main__":
    main()
