#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/monitor-evasion/reproduce.sh --dry-run    print the commands only
#   experiments/monitor-evasion/reproduce.sh              run them (about 2 hours on an M-series laptop)
#
# Downloads Qwen2.5-1.5B-Instruct and the published lock's adapters (albertlieadrian/qwen2.5-1.5b-sandbagging-lock)
# from Hugging Face. The monitors, the evasion LoRAs and the baseline's answer distributions are written to a work
# folder (the lab's outputs folder or ~/.cache/steerlab); results/ holds every number and per-item score.
set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-../../.venv/bin/python}

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

echo "# 1: the three monitors (layers 10, 17, 23) on the untouched organism, thresholds at 5% false positives"
run "$PY" evade.py baseline
echo "# 2: the pre-registered arms, 300 steps each: probe only (P), probe + keep answers (PKL), random direction (R)"
run "$PY" evade.py train --arm P
run "$PY" evade.py train --arm PKL
run "$PY" evade.py train --arm R
echo "# 3: per-item scores for every arm, and how far each arm moved the activations"
run "$PY" scores.py
echo "# 4 (exploratory): the first 25 steps of arm P, checked after every step"
run "$PY" evade.py train --arm P --steps 25 --eval-every 1 --tag _first25 --curve-only
echo "# 5: figures (they print every plotted number) and the cover"
run "$PY" make_figures.py
run "$PY" make_cover.py
