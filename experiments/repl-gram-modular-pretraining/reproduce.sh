#!/usr/bin/env bash
# Rerun every number in report/index.html, then redraw its figures.
#
#   experiments/repl-gram-modular-pretraining/reproduce.sh --dry-run    print the commands only
#   experiments/repl-gram-modular-pretraining/reproduce.sh              run them
#
# Steps 1-3 run on Modal (one H100 per seed, about 1.5 hours and $6 each; never more than two at a time).
# Steps 4-5 need only results/ and run on a laptop in seconds.
set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
PY=${PYTHON:-../../.venv/bin/python}
MODAL=${MODAL:-../../.venv/bin/modal}
export MODAL_PROFILE=${MODAL_PROFILE:-albertlieyingadrian}

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

echo "# 1: tokenise Simple Stories (48 topics, 10% test split, seed 42) into the Modal volume"
run "$MODAL" run modal_app.py::prep
echo "# 2: one job per seed: baseline, GRAM, five filtered models, MaxEnt, and every relearning attack"
run "$MODAL" run modal_app.py::launch --seeds 1,2
run "$MODAL" run modal_app.py::launch --seeds 3
echo "# 3: copy the result JSON (not checkpoints) into results/runs/"
run "$MODAL" run modal_app.py::pull
echo "# 4: compute ratios, seed means and CIs, the pre-registered tests"
run "$PY" analyze.py
echo "# 5: figures (they print every plotted number)"
run "$PY" make_figures.py
echo "# 6: the cover and research-index thumbnail"
run "$PY" make_cover.py
