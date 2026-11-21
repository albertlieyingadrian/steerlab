#!/usr/bin/env bash
# Rebuild every number in the The Taste Stack, Episode 1 note. Costs about $40-60 of Modal (H100/L40S); caps in budget.py.
#   bash reproduce.sh            # run everything
#   bash reproduce.sh --dry-run  # print the commands only
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
MODAL=../../.venv/bin/modal
# Sounds Like Me data: set SLM_DIR to a checkout whose experiments/sounds-like-me/data/ was rebuilt by that episode.
export MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian}
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1
run() { echo "+ $*"; [[ "$DRY" == "1" ]] || "$@"; }

run $MODAL deploy modal_app.py
run $MODAL deploy modal_pigment.py
run $MODAL deploy modal_voice.py
# M0: freeze the text sets (hashes -> results/data_manifest.json)
run $PY build_data.py humans
run $PY build_data.py generate
run $PY build_data.py humanize
run $PY build_data.py assemble
# M1-M3: tests 1, 2, 3, 6 (leaderboard, shortcut edits, learning curves, cost)
run $PY probes.py extract
run $PY probes.py luar
run $PY probes.py score
run $PY probes.py fit
run $PY probes.py curves
# M4: test 4 (presence vs use)
run $PY causal.py units
run $PY causal.py tune
run $PY causal.py run
run $PY causal.py score
# M4b-c: tests 9 and 10 (voice: vector vs LoRA vs few-shot)
run $PY voice.py tune
run $PY voice.py run
run $PY voice.py score
# M4d, M5a: tests 7 and 5 (reading probes; planted confounds and hacks)
run $PY explain.py grader-check
run $PY explain.py readouts
run $PY explain.py judge
# M4e: test 8 (Pigment Gym checkpoints)
run $PY pigment.py run
run $PY pigment.py score
# predictions and figures
run $PY analyze.py
run $PY make_figures.py
run $PY budget.py
