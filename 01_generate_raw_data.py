import json
import random
from datetime import datetime, timedelta

categories = ["GROCERY", "ELECTRONICS", "MONEY_TRANSFER", "LUXURY", "TRAVEL"]
devices = ["iOS", "Android", "Web"]

records = []
start_time = datetime.now()

for i in range(2000):
    tx_time = start_time - timedelta(minutes=random.randint(0, 10000))
    amount = round(random.uniform(5.0, 5000.0), 2)
    # High amounts or random chance trigger synthetic fraud labels
    is_fraud = 1 if (amount > 3200 or random.random() < 0.04) else 0

    doc = {
        "transaction_id": f"TXN_{10000 + i}",
        "sender": {
            "account_id": f"ACC_{random.randint(100, 200)}", 
            "risk_rating": random.choice(["LOW", "MED", "HIGH"])
        },
        "receiver": {"account_id": f"ACC_{random.randint(201, 300)}"},
        "amount": amount,
        "category": random.choice(categories),
        "device_footprint": {
            "device_os": random.choice(devices), 
            "ip_address": f"192.168.1.{random.randint(1, 255)}"
        },
        "location": {
            "type": "Point",
            "coordinates": [round(random.uniform(72.0, 78.0), 4), round(random.uniform(12.0, 19.0), 4)]
        },
        "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
        "is_fraud_label": is_fraud
    }
    records.append(doc)

# Save to data folder
with open("data/raw_transactions.json", "w") as f:
    json.dump(records, f, indent=2)

print("Success: Generated 2,000 raw JSON records in data/raw_transactions.json")
