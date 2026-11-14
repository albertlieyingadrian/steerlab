#!/usr/bin/env bash
# Rerun the note. Everything trains and evaluates on Modal (one A10G per job, at most three at once); nothing trains
# locally. Steps: download the data and released checkpoints into a Modal volume; reproduce the released baselines with
# the official code; the exploration runs (validation split); the pre-registered finals (test split, seeds 1-3); the
# frames for the cover; then the summary and the figures, which only read results/. About 25 A10G-hours, about $40.
# With --scores, only the steps that read results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
STEPS=(
  "$MODAL run modal_app.py"
  "$PY run.py repro_trl"
  "$PY run.py repro2"
  "$PY run.py jobs configs/explore.json"
  "$PY run.py jobs configs/finals.json"
  "$PY run.py hero"
  "$PY analyse.py"
  "$PY make_figures.py"
)
[[ "${1:-}" == "--scores" ]] && STEPS=("$PY analyse.py" "$PY make_figures.py")
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' "${STEPS[@]}"
  exit 0
fi
for step in "${STEPS[@]}"; do
  eval "$step"
done
