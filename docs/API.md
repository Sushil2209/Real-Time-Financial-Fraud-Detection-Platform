# API Reference

The scoring service lives in [src/api.py](../src/api.py). Start it from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --reload
```

It listens on `http://127.0.0.1:8000`. FastAPI also serves interactive docs at `/docs`.
On startup it loads `models/xgb_model.joblib` (the app will not start without it).

## GET /health

Returns `{"status": "ok"}` with HTTP 200. It does not check Feast or the model.

## POST /score

Scores one transaction.

### Request (JSON)

The model is built from `RAW_COLUMNS` in [src/model_features.py](../src/model_features.py).

| Field | Type | Required |
|-------|------|----------|
| `TransactionID` | integer | **yes** |
| `card1` | integer | **yes** |
| `TransactionAmt` | number | no |
| `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `DeviceType` | string | no |
| `addr1`, `addr2` | number | no |
| `C1` … `C14` | number | no |
| `D1` … `D15` | number | no |

Optional fields that are missing or `null` become NaN, which XGBoost handles.
Text values not seen during training also become NaN. Extra fields (for example `isFraud`,
which the Kafka producer sends) are ignored.

### Response (JSON)

| Field | Meaning |
|-------|---------|
| `TransactionID` | Echo of the request ID |
| `fraud_score` | Fraud probability from the model, 0 to 1 |
| `risk_level` | `low` if score < 0.3; `medium` if 0.3 ≤ score ≤ 0.7; `high` if score > 0.7 |
| `features_source` | `feast` if the 3 history features came from Feast, `default` if the fallback was used |
| `latency_ms` | Time spent inside `/score`, rounded to 2 decimals |

Example response:

```json
{"TransactionID": 2987000, "fraud_score": 0.0123, "risk_level": "low",
 "features_source": "feast", "latency_ms": 18.4}
```

### Examples

PowerShell:

```powershell
$body = @{ TransactionID = 2987000; card1 = 13926; TransactionAmt = 68.5; ProductCD = "W" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/score -Method Post -ContentType "application/json" -Body $body
```

curl (use `curl.exe` in PowerShell):

```powershell
curl.exe -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d '{\"TransactionID\":2987000,\"card1\":13926,\"TransactionAmt\":68.5}'
```

A ready-made script is also available: `python src/api_test_request.py`.

### Error behavior

- Missing `TransactionID` or `card1`, or a wrong type: HTTP **422** with FastAPI's validation details.
- Feast problems never cause an error (see below); the API falls back to defaults and returns 200.
- Model or feature-building errors are not caught and produce HTTP **500**.

## Feast feature server call

`get_online_features()` sends one request per score:

- URL: env var `FEAST_URL`, default `http://127.0.0.1:6566/get-online-features`
- Timeout: 1 second
- Body:

```json
{
  "features": [
    "card_transaction_features:txn_count_24h",
    "card_transaction_features:amount_vs_avg",
    "card_transaction_features:device_changed"
  ],
  "entities": {"card1": [13926]}
}
```

It reads `metadata.feature_names` and `results[i].values[0]` from the reply.

### Fallback to default features

If Feast is down, slow (over 1 s), returns an HTTP error, a malformed reply, or any null value,
the API logs a warning and uses these defaults, with `features_source = "default"`:

| Feature | Default |
|---------|---------|
| `txn_count_24h` | 0 |
| `amount_vs_avg` | 1.0 |
| `device_changed` | 0 |

Scores made with defaults ignore the card's history, so they can be less accurate.

## Kafka message format

[src/producer.py](../src/producer.py) replays the last 20% (by time) of transactions to topic
`transactions` on `localhost:9092`. Run: `python src/producer.py --count 1000 --delay 0.05`.

Each message is one UTF-8 JSON object (one row). It holds:

- `TransactionID`, `card1`, `isFraud` (from `features.parquet` / `merged.parquet`, `isFraud` is the true label)
- `event_timestamp` (ISO 8601 string, nanoseconds)
- `txn_count_24h`, `amount_vs_avg`, `device_changed` (computed features)
- all other columns of `merged.parquet` (raw IEEE-CIS columns, missing values as `null`)

[src/consumer.py](../src/consumer.py) posts each message to `/score` unchanged and stores the result.
If scoring fails it prints a warning and skips the message.

## SQLite database

File: `data/processed/scored.db`.

### Table `scores` (created by `src/consumer.py`)

One row per scored message; no primary key.

| Column | Type |
|--------|------|
| `TransactionID` | INTEGER |
| `card1` | INTEGER |
| `TransactionAmt` | REAL |
| `fraud_score` | REAL |
| `risk_level` | TEXT |
| `latency_ms` | REAL |
| `scored_at` | TEXT (UTC ISO 8601) |
| `isFraud` | INTEGER (true label) |
| `features_source` | TEXT (`feast` or `default`; added by migration for older databases) |

### Table `feedback` (created by `src/dashboard.py`)

One row per reviewed transaction (upserted when an analyst saves a case).

| Column | Type | Constraint |
|--------|------|-----------|
| `TransactionID` | INTEGER | PRIMARY KEY |
| `status` | TEXT | NOT NULL; one of `New`, `Under Review`, `Fraud`, `Not Fraud` |
| `updated_at` | TEXT | NOT NULL (UTC ISO 8601) |
| `note` | TEXT | optional |
