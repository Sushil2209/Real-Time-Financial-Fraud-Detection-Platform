"""Send one sample transaction to the running API and print the answer."""

import json
import urllib.request

URL = "http://127.0.0.1:8000/score"
SAMPLE = {
    "TransactionID": 3000001,
    "card1": 13926,
    "TransactionAmt": 59.0,
    "ProductCD": "W",
    "card4": "visa",
    "card6": "debit",
    "addr1": 315,
    "addr2": 87,
    "P_emaildomain": "gmail.com",
    "C1": 1, "C2": 1, "D1": 0,
}

request = urllib.request.Request(
    URL, data=json.dumps(SAMPLE).encode(), headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(request) as response:
    print(json.dumps(json.load(response), indent=2))
