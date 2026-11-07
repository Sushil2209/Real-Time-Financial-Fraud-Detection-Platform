from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64


features_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "features.parquet"

card1 = Entity(
    name="card1",
    join_keys=["card1"],
    value_type=ValueType.INT64,
)

transaction_source = FileSource(
    name="transaction_features",
    path=str(features_path),
    timestamp_field="event_timestamp",
)

card_transaction_features = FeatureView(
    name="card_transaction_features",
    entities=[card1],
    # Include the 2017-2018 sample data in the first incremental materialization.
    ttl=timedelta(days=36500),
    schema=[
        Field(name="txn_count_24h", dtype=Int64),
        Field(name="amount_vs_avg", dtype=Float64),
        Field(name="device_changed", dtype=Int64),
    ],
    source=transaction_source,
    online=True,
)
