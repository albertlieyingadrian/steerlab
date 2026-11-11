#!/usr/bin/env bash
# Rerun the note. The GPU steps run on Modal H100s (`modal setup` first; about $4.6 per H100-hour at list price) and
# cost $13.86 in all on 2026-11-10/11 (the six main runs $10.63). Activations are copied down to work/ (about 1 GB, not shipped).
# With --scores, only the steps that read the shipped results/: the lead-time table and the figures.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
STEPS=("$PY env.py"
  "$MODAL run modal_app.py --job screen"
  "$MODAL run modal_app.py --job probe_data"
  "$MODAL volume get rl-hack-monitor probe work/"
  "$PY analyze.py --probe-only"
  "$MODAL run --detach modal_app.py --job runs --seeds 0,1,2 --steps 50 --lr 2e-5 --eval-every 2 --tag main"
  "$MODAL volume get rl-hack-monitor runs work/"
  "$PY analyze.py"
  "$PY posthoc.py"
  "$PY export_results.py")
SCORES=("$PY make_figures.py" "$PY make_explorer.py" "$PY make_cover.py")
if [[ "${1:-}" == "--scores" || "${2:-}" == "--scores" ]]; then STEPS=(); fi
STEPS+=("${SCORES[@]}")
if [[ "${1:-}" == "--dry-run" ]]; then printf '%s\n' "${STEPS[@]}"; exit 0; fi
for step in "${STEPS[@]}"; do echo "+ $step"; eval "$step"; done
