#!/usr/bin/env bash
# Reproduce every number in the Episode 2 note (Spectrum Tuning + Artificial Hivemind replications).
# COSTS MONEY: Modal GPU time (hard cap $50 in modal_app.py) and OpenRouter (hard stop at $9.50 in
# openrouter_client.py). Needs a Modal profile and OPENROUTER_API_KEY in .env.
#
#   bash experiments/pluralism-replications/reproduce.sh            # run everything
#   bash experiments/pluralism-replications/reproduce.sh --dry-run  # print the commands only
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
MODAL=".venv/bin/modal"
EXP=experiments/pluralism-replications
export MODAL_PROFILE="${MODAL_PROFILE:-albertlieyingadrian}"

steps=(
  "$PY -m pytest $EXP/tests -q"                                   # known-answer tests of the measures
  "$MODAL run $EXP/modal_app.py::fetch"                           # Spectrum Suite sources -> Modal volume
  "$MODAL run $EXP/modal_app.py::full --max-eval 300"             # A, B, open-model C/D samples, open judges
  "$PY $EXP/run_hivemind.py checks"                               # embedding + judge known-answer checks
  "$PY $EXP/run_hivemind.py gen --run full"                       # frontier samples (C, D), 3 models x 100 x 10
  "$PY $EXP/run_hivemind.py judge --run full"                     # GPT-4o and Claude Haiku 5.5 judges (E)
  "$PY $EXP/analyse.py --run full300"                             # every number + verdict -> results/summary.json
  "$PY $EXP/make_figures.py"                                      # static figures + page data
)

for s in "${steps[@]}"; do
  echo "+ $s"
  if [[ "${1:-}" != "--dry-run" ]]; then eval "$s"; fi
done
