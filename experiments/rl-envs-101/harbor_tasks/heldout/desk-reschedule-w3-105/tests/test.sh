#!/bin/bash
# Grades the calendar the agent left, never its words. Writes every key on every path (a missing key averages as 0).
mkdir -p /logs/verifier
python3 /tests/verify.py /app/clinic.db > /logs/verifier/reward.json 2> /logs/verifier/verify_error.txt \
  || echo '{"reward": 0.0, "hidden": 0.0, "collateral": -1.0}' > /logs/verifier/reward.json
cat /logs/verifier/reward.json
