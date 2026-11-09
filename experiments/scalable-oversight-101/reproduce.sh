#!/usr/bin/env bash
# Rerun the note: the dataset (cached samples are reused), activations, weak-to-strong, sandwiching (cached arguments
# are reused) and the figures. Local only, no API keys. The automated researcher's search is a record, kept in
# aar/results.tsv and aar/runs/; rerun any run with aar/run.py.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
STEPS=(data.py features.py w2s.py w2s_lora.py "w2s_lora.py --shuffled" random_labels.py sandwich.py make_figures.py)
if [[ "${1:-}" == "--dry-run" ]]; then
  for step in "${STEPS[@]}"; do echo "$PY $step"; done
  exit 0
fi
for step in "${STEPS[@]}"; do
  # shellcheck disable=SC2086
  "$PY" $step
done
