from __future__ import annotations

import json
import sys
import urllib.request


API_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/predict"

payload = {
    "query_id": "T2_LOCAL_001",
    "type": "type2",
    "query": "An LC circuit consists of a 50 mH inductor and a 20 uF capacitor. If the maximum voltage across the capacitor is 12 V, what is the maximum current in the circuit?",
    "premises": [],
    "options": [],
    "debug": True,
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    API_URL,
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req, timeout=60) as resp:
    raw = resp.read().decode("utf-8")
    status = resp.status

print("HTTP", status)
print(raw)

parsed = json.loads(raw)
assert isinstance(parsed, list), "Response must be a JSON list."
assert len(parsed) == 1, "Single query must return a list with exactly one object."
item = parsed[0]
for key in ["query_id", "answer", "unit", "explanation", "premises_used", "reasoning"]:
    assert key in item, f"Missing key: {key}"
assert item["query_id"] == payload["query_id"]
assert item["unit"] == "A"
assert item["premises_used"] == []
assert isinstance(item["explanation"], str) and item["explanation"].strip()
print("Schema check passed.")
