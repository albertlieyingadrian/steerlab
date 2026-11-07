#!/usr/bin/env bash
# Rerun the pilot: every rollout still missing from results/rollouts.jsonl, then the leaderboard and heatmap.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "$PY run.py"; echo "$PY analyze.py"; exit 0
fi
"$PY" run.py
"$PY" analyze.py
