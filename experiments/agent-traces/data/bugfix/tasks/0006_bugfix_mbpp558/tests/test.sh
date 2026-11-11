#!/usr/bin/env bash
set -u
mkdir -p /logs/verifier
cd /home/user/input
if python3 /tests/hidden_tests.py; then echo "1.0" > /logs/verifier/reward.txt; else echo "0.0" > /logs/verifier/reward.txt; fi
exit 0
